using UnityEngine;
using System.Collections;
using System.IO;

public class SweepBatchRunner : MonoBehaviour
{
    const string ChungValidationFlag = "-validateChungEval6";
    const int EvalFormatVersion = 3;

    [System.Serializable]
    class EvalDatasetFormat
    {
        public int formatVersion;
        public string laterality;
        public int whiskerCount;
        public int sourceFrameCount;
        public string tactileOrdering;
        public string[] tactileColumns;
        public string[] whiskerNames;
        public string[] conditions;
        public int repeatsPerCondition;
    }

    public enum GenMode { TRAIN_AUG, EVAL_6COND, TRAIN_SUPERVISED }
    public WhiskerManager whiskerManager;

    [Header("Generation mode")]
    public GenMode mode = GenMode.EVAL_6COND;

    [Header("Eval profile (only used in EVAL_6COND)")]
    public Eval6CondProfile evalProfile;
    public Eval6CondPlan eval6CondPlan;
    public ArcStimulusSpawner arcSpawner;

    [Header("Supervised profile (only used in TRAIN_SUPERVISED)")]
    public SupervisedProfile supervisedProfile;
    public SupervisedPlan supervisedPlan;

    [Header("Scene refs")]
    public PassiveSweepLogger logger;
    public Material defaultMaterial;


    [Header("ShapeNet root")]
    public string shapenetRoot = "Assets/ShapeNet";  // ShapeNet/<synset>/<model_id>/models/model.obj

    [Header("Global scale")]
    public float sceneUnitsPerMm = 0.10f;        // e.g., 0.10 → 40 mm = 4 scene units

    [Header("Paper parameters (Inspector)")]
    public float speed_mm_s   = 30f;             // sweep speed
    public float[] height_mm  = { -5f, 0f };     // object height (Y)
    public float[] yaw_deg    = { 0f, 30f, 90f, 120f };
    public float[] H_mm       = { 5f, 8f };      // standoff distance (Z)
    public float sweep_len_mm = 40f;             // total travel along Z

    [Header("Timing (visibility)")]
    public float preHold = 0.5f, postHold = 0.5f;

    [Header("Output")]
    public string outDir = "Data/whisker_logs";

    [Header("CLI / Resuming")]
    public int startIndex = 0;               // first model index to process
    public int count = 300;                   // how many models to process (-1 = all)

    string[] objPaths;
    bool isChungValidationRun;
    int originalEvalRepeats;
    string originalEvalOutputDir;

    ITrialPlan BuildPlan()
    {
        if (mode == GenMode.EVAL_6COND)
        {
            return eval6CondPlan;
        }
        else if (mode == GenMode.TRAIN_SUPERVISED)
        {
            return supervisedPlan;
        }
        else
        {
            string[] paths;
            if (File.Exists("obj_list_order.txt"))
                paths = File.ReadAllLines("obj_list_order.txt");
            else
            {
                paths = GetObjPaths(shapenetRoot);
                File.WriteAllLines(Path.Combine(outDir, "obj_list_order.txt"), paths);
            }

            int start = Mathf.Clamp(startIndex, 0, paths.Length);
            if (start != startIndex)
                Debug.LogWarning($"[TRAIN_AUG] startIndex {startIndex} is outside the available range 0..{paths.Length}; using {start}.");

            int available = paths.Length - start;
            int take = (count < 0) ? available : Mathf.Min(count, available);
            if (take == 0)
                Debug.LogWarning($"[TRAIN_AUG] No OBJ trials selected (paths={paths.Length}, start={start}, count={count}).");

            var window = new System.Collections.Generic.List<string>();
            for (int i = 0; i < take; i++) window.Add(paths[start + i]);

            return new TrainAugPlan(
                window.ToArray(), outDir,
                H_mm, height_mm, yaw_deg,
                sweep_len_mm, sceneUnitsPerMm,
                new Vector3(25f, -4f, -20f)   // Default head offset in scene units.
            );
        }
    }

    void Start()
    {
        isChungValidationRun =
            Application.isBatchMode
            && System.Array.Exists(
                System.Environment.GetCommandLineArgs(),
                arg => arg == ChungValidationFlag
            );

        if (isChungValidationRun)
        {
            mode = GenMode.EVAL_6COND;
            originalEvalRepeats = evalProfile.repeatsPerCond;
            originalEvalOutputDir = evalProfile.outputDir;
            evalProfile.repeatsPerCond = 1;
            evalProfile.outputDir = Path.Combine(
                Application.temporaryCachePath,
                "chung_eval6_validation"
            );
            Debug.Log($"[Runner] Chung validation output: {evalProfile.outputDir}");
        }

        Debug.Log($"[Runner] Generation mode: {mode}");
        System.IO.Directory.CreateDirectory(outDir);
        StartCoroutine(RunBatch());
    }

    IEnumerator RunBatch()
    {
        yield return null;                        // 1 rendered frame
        yield return new WaitForFixedUpdate();    // 1 physics step
        yield return null;

        var plan = BuildPlan();
        if (plan == null)
        {
            Debug.LogError($"[Runner] No trial plan is assigned for mode {mode}. Check the SweepBatchRunner references in the Inspector.");
            yield break;
        }
        int completedSupervisedTrials = 0;
        string supervisedCompletionMarker = null;
        int completedEvalTrials = 0;
        string evalCompletionMarker = null;
        if (mode == GenMode.EVAL_6COND && evalProfile != null)
        {
            Directory.CreateDirectory(evalProfile.outputDir);
            evalCompletionMarker = Path.Combine(
                evalProfile.outputDir,
                "dataset_complete.txt"
            );
            if (File.Exists(evalCompletionMarker))
                File.Delete(evalCompletionMarker);
        }
        if (mode == GenMode.TRAIN_SUPERVISED && supervisedProfile != null)
        {
            supervisedCompletionMarker = Path.Combine(
                supervisedProfile.outputDir,
                "dataset_complete.txt"
            );
            if (File.Exists(supervisedCompletionMarker))
                File.Delete(supervisedCompletionMarker);
        }

        foreach (var trial in plan.GenerateTrials())
        {
            if (mode == GenMode.TRAIN_AUG)
            {
                string objPath = FindObjPathByIds(trial.synset, trial.modelId);
                whiskerManager.animate = false;
                var carrier = ShapeFactory.CreateCarrierWithObj(
                    objPath, defaultMaterial,
                    addKinematicRB: true,
                    sceneUnitsPerMm: sceneUnitsPerMm
                );
                foreach (Transform t in carrier.GetComponentsInChildren<Transform>(true))
                {
                    t.gameObject.tag = "Stimulus";
                    int stimLayer = LayerMask.NameToLayer("Stimulus");
                    if (stimLayer != -1) t.gameObject.layer = stimLayer;
                }

                // Force Rigidbody + collider settings on the collider object (model_normalized)
                var mc = carrier.GetComponentInChildren<MeshCollider>(true);
                if (mc != null)
                {
                    mc.convex = true;
                    mc.isTrigger = false;

                    var rbStim = mc.GetComponent<Rigidbody>();
                    if (rbStim == null) rbStim = mc.gameObject.AddComponent<Rigidbody>();
                    rbStim.isKinematic = true;
                    rbStim.useGravity = false;

                    rbStim.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
                }
                else
                {
                    Debug.LogError("[TRAIN_AUG] No MeshCollider found under carrier.");
                }

                var sweep = carrier.AddComponent<PassiveSweep>();

                logger.SetOutput(trial.outFile);
                logger.BeginTrial();

                sweep.Configure(
                    sceneUnitsPerMm,
                    new Vector3(25f, -4f, -20f),
                    trial.H_mm,
                    trial.y_mm,
                    trial.startZ_mm,
                    trial.endZ_mm,
                    speed_mm_s,
                    trial.yawDeg,
                    preHold,
                    postHold,
                    logger
                );

                yield return new WaitWhile(() => sweep != null && sweep.IsRunning);
                logger.EndTrial();
                if (carrier) Destroy(carrier);
            }
            else if (mode == GenMode.EVAL_6COND)
            {
                arcSpawner.SetAllActive(false);
                Transform root = arcSpawner.transform.Find("__GeneratedStimuli__");
                GameObject stim = root.Find(trial.label).gameObject;
                stim.SetActive(true);
                stim.tag = "Stimulus";

                yield return CapturePoseControlledTrial(
                    trial,
                    evalProfile.maxFrames,
                    evalProfile.preHold,
                    evalProfile.postHold,
                    "EVAL_6COND"
                );

                if (logger.CapturedFrames == evalProfile.maxFrames)
                    completedEvalTrials++;
                else
                    Debug.LogError(
                        $"[EVAL_6COND] Incomplete trial {trial.label} rep " +
                        $"{trial.repeatIndex}: captured {logger.CapturedFrames} frames."
                    );

                stim.SetActive(false);

            }
            else if (mode == GenMode.TRAIN_SUPERVISED)
            {
                arcSpawner.SetAllActive(false);

                bool isConcave = trial.label.StartsWith("concave");
                var curvature = isConcave
                    ? ArcStimulusSpawner.Curvature.Concave
                    : ArcStimulusSpawner.Curvature.Convex;

                float distanceUnits = trial.startZ_mm;
                float yUnits = trial.y_mm;

                GameObject stim = arcSpawner.SpawnTrialStimulus(
                    trial.label,
                    curvature,
                    distanceUnits,
                    yUnits,
                    trial.yawDeg
                );

                stim.SetActive(true);
                stim.tag = "Stimulus";

                yield return CapturePoseControlledTrial(
                    trial,
                    supervisedProfile.maxFrames,
                    supervisedProfile.preHold,
                    supervisedProfile.postHold,
                    "TRAIN_SUPERVISED"
                );

                if (logger.CapturedFrames == supervisedProfile.maxFrames)
                    completedSupervisedTrials++;
                else
                    Debug.LogError($"[TRAIN_SUPERVISED] Incomplete trial {trial.label} rep {trial.repeatIndex}: captured {logger.CapturedFrames} frames.");

                arcSpawner.DestroyStimulus(stim);
            }
            yield return new WaitForSeconds(0.1f);
        }

        if (mode == GenMode.EVAL_6COND && evalProfile != null)
        {
            int expectedTrials =
                evalProfile.repeatsPerCond * evalProfile.labels.Length;
            if (completedEvalTrials == expectedTrials)
            {
                var format = new EvalDatasetFormat
                {
                    formatVersion = EvalFormatVersion,
                    laterality = "right",
                    whiskerCount = WhiskerManager.RightWhiskerCount,
                    sourceFrameCount = WhiskerManager.SourceFrameCount,
                    tactileOrdering = "frame-major, then whiskerNames order",
                    tactileColumns = new[] { "s", "theta_deg" },
                    whiskerNames = whiskerManager.whiskerNames.ToArray(),
                    conditions = (string[])evalProfile.labels.Clone(),
                    repeatsPerCondition = evalProfile.repeatsPerCond
                };
                File.WriteAllText(
                    Path.Combine(evalProfile.outputDir, "dataset_format.json"),
                    JsonUtility.ToJson(format, prettyPrint: true)
                );
                File.WriteAllText(
                    evalCompletionMarker,
                    $"format_version={EvalFormatVersion}\n" +
                    $"completed_trials={completedEvalTrials}\n" +
                    $"conditions={evalProfile.labels.Length}\n" +
                    $"repeats={evalProfile.repeatsPerCond}\n" +
                    $"whiskers={WhiskerManager.RightWhiskerCount}\n" +
                    "laterality=right\n"
                );
                Debug.Log(
                    $"[EVAL_6COND] RIGHT_ONLY_EVAL6_COMPLETE: " +
                    $"{completedEvalTrials} trials."
                );
            }
            else
            {
                Debug.LogError(
                    $"[EVAL_6COND] Dataset incomplete: " +
                    $"{completedEvalTrials}/{expectedTrials}. " +
                    "No completion marker was written."
                );
            }
        }
        else if (mode == GenMode.TRAIN_SUPERVISED && supervisedProfile != null)
        {
            int expectedTrials = supervisedProfile.repeatsPerCond
                * supervisedProfile.labels.Length
                * supervisedProfile.distances.Length;

            if (completedSupervisedTrials == expectedTrials)
            {
                File.WriteAllText(
                    supervisedCompletionMarker,
                    $"completed_trials={completedSupervisedTrials}\nseed={supervisedProfile.randomSeed}\n"
                );
                Debug.Log($"[TRAIN_SUPERVISED] Complete dataset: {completedSupervisedTrials} trials.");
            }
            else
            {
                Debug.LogError($"[TRAIN_SUPERVISED] Dataset incomplete: {completedSupervisedTrials}/{expectedTrials} trials. No completion marker was written.");
            }
        }

        Debug.Log("[Runner] Done.");

        if (isChungValidationRun)
        {
            evalProfile.repeatsPerCond = originalEvalRepeats;
            evalProfile.outputDir = originalEvalOutputDir;
            Debug.Log("[Runner] CHUNG_EVAL6_VALIDATION_COMPLETE");

        #if UNITY_EDITOR
            UnityEditor.EditorApplication.Exit(0);
        #else
            Application.Quit(0);
        #endif
        }
    }

    IEnumerator CapturePoseControlledTrial(
        TrialSpec trial,
        int sourceFrameCount,
        float preTrialHold,
        float postTrialHold,
        string modeLabel
    )
    {
        if (sourceFrameCount != 51)
        {
            Debug.LogError($"[{modeLabel}] Exact theta capture requires 51 source frames (-10 through 40); got {sourceFrameCount}.");
            yield break;
        }

        // Pose control guarantees one CSV block for every source angle and
        // removes rendered-frame timing from the tactile input.
        whiskerManager.animate = false;
        whiskerManager.SetAllWhiskersToFrame(0);
        Physics.SyncTransforms();

        if (preTrialHold > 0f)
            yield return new WaitForSeconds(preTrialHold);

        logger.SetOutput(trial.outFile);
        logger.maxFrames = sourceFrameCount;
        logger.BeginTrial(manualCapture: true);

        for (int sourceFrame = 0; sourceFrame < sourceFrameCount; sourceFrame++)
        {
            logger.ResetContacts();
            whiskerManager.SetAllWhiskersToFrame(sourceFrame);
            Physics.SyncTransforms();

            // Query the fixed pose directly instead of depending on trigger
            // callback timing. This makes identical EVAL_6COND runs produce
            // byte-identical tactile CSVs.
            whiskerManager.EvaluateContactsAgainstStimuli();

            if (!logger.CaptureFrame())
            {
                Debug.LogError($"[{modeLabel}] Failed to capture source frame {sourceFrame} for {trial.label}.");
                break;
            }

            yield return null;
        }

        logger.EndTrial();

        if (postTrialHold > 0f)
            yield return new WaitForSeconds(postTrialHold);
    }

    // Helper (only for resolving OBJ paths by IDs inside TRAIN_AUG branch)
    string FindObjPathByIds(string synset, string modelId)
    {
        var modelsPath = System.IO.Path.Combine(shapenetRoot, synset, modelId, "models");
        var normalized = System.IO.Path.Combine(modelsPath, "model_normalized.obj");
        var plain      = System.IO.Path.Combine(modelsPath, "model.obj");
        if (System.IO.File.Exists(normalized)) return normalized;
        if (System.IO.File.Exists(plain)) return plain;
        throw new System.IO.FileNotFoundException($"OBJ not found for {synset}/{modelId}");
    }


    string[] GetObjPaths(string shapenetRoot)
    {
        // Accept both "model.obj" and "model_normalized.obj"
        var paths = new System.Collections.Generic.List<string>();

        // Enumerate all synsets
        foreach (var synsetDir in System.IO.Directory.GetDirectories(shapenetRoot))
        {
            // Enumerate all model_ids
            foreach (var modelDir in System.IO.Directory.GetDirectories(synsetDir))
            {
                var modelsPath = System.IO.Path.Combine(modelDir, "models");
                if (!System.IO.Directory.Exists(modelsPath)) continue;

                var normalized = System.IO.Path.Combine(modelsPath, "model_normalized.obj");
                var plain      = System.IO.Path.Combine(modelsPath, "model.obj");

                if (System.IO.File.Exists(normalized)) paths.Add(normalized);
                else if (System.IO.File.Exists(plain)) paths.Add(plain);
            }
        }

        return paths.ToArray();
    }

}


public static class ShapeFactory
{
    public static GameObject CreateCarrierWithObj(
        string objPath, Material defaultMat = null,
        float scale = 1f, bool addKinematicRB = true,
        float sceneUnitsPerMm = 0.1f
    )
    {
        var carrier = new GameObject("ShapeCarrier");
        carrier.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

        var child = new GameObject(System.IO.Path.GetFileNameWithoutExtension(objPath));
        child.transform.SetParent(carrier.transform, worldPositionStays: false);

        // Load mesh
        var mesh = RuntimeObjLoader.LoadMesh(objPath, true, scale);

        var mf = child.AddComponent<MeshFilter>();
        mf.sharedMesh = mesh;

        var mr = child.AddComponent<MeshRenderer>();
        mr.sharedMaterial = defaultMat ?? new Material(Shader.Find("Universal Render Pipeline/Lit"));

        // Normalize to 40 mm box (in scene units)
        float targetBoxUnits = 40f * sceneUnitsPerMm;
        ScaleChildToBoundingBox(child, targetBoxUnits);

        // Rigidbody on parent
        var rb = carrier.AddComponent<Rigidbody>();
        rb.isKinematic = true;
        rb.useGravity = false;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;

        // Collider on child (after scaling)
        var mc = child.AddComponent<MeshCollider>();
        mc.convex = true;
        mc.isTrigger = true;
        mc.providesContacts = true;
        mc.sharedMesh = mesh;

        return carrier;
    }

    static void ScaleChildToBoundingBox(GameObject child, float targetBoxUnits)
    {
        // 1) bounds before scaling
        var rends = child.GetComponentsInChildren<Renderer>(true);
        if (rends.Length == 0) return;

        var b = rends[0].bounds;
        for (int i = 1; i < rends.Length; i++) b.Encapsulate(rends[i].bounds);

        float maxDim = Mathf.Max(b.size.x, Mathf.Max(b.size.y, b.size.z));
        if (maxDim <= 1e-8f) return;

        float s = targetBoxUnits / maxDim;

        // 2) apply scale
        child.transform.localScale *= s;

        // 3) recompute bounds AFTER scaling and recenter
        var rends2 = child.GetComponentsInChildren<Renderer>(true);
        var b2 = rends2[0].bounds;
        for (int i = 1; i < rends2.Length; i++) b2.Encapsulate(rends2[i].bounds);

        // translate so the world-space center of the mesh = carrier origin
        var centerLS = child.transform.InverseTransformPoint(b2.center);
        child.transform.localPosition -= centerLS;
    }
}
