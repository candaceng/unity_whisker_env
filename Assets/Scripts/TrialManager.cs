using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using UnityEngine;

public class TrialManager : MonoBehaviour
{
    private const int OutputFormatVersion = 4;

    [Header("Run")]
    [Min(1)] public int totalTrials = 100000;
    [Min(1)] public int trialsPerBatch = 1000;
    [Min(1)] public int progressLogInterval = 100;
    public int baseRandomSeed = 20260730;

    [Tooltip("Project-relative output folder. It is intentionally outside Assets.")]
    public string outputDirectory = "Data/RecordedTrials";

    [Header("References")]
    public WhiskerManager whiskerManager;
    public VideoWhiskerRecorder recorder;
    public Camera rightEyeCamera;
    public Material proceduralSkybox;

    [Header("Mouse right-eye pose (head-local coordinates)")]
    public Vector3 rightEyeLocalPosition =
        new Vector3(7.2068f, 5.5667f, -14.3873f);
    [Range(-180f, 180f)] public float rightEyeAzimuthDegrees = 42.5f;
    [Range(-89f, 89f)] public float rightEyeElevationDegrees = 22f;
    [Range(1f, 179f)] public float rightEyeFieldOfViewDegrees = 135f;

    [Header("Right-side random obstacle field (head-local coordinates)")]
    [Min(1)] public int minimumObstacles = 1;
    [Min(1)] public int maximumObstacles = 5;
    public Vector3 obstaclePositionMin = new Vector3(-20f, -0.5f, -10f);
    public Vector3 obstaclePositionMax = new Vector3(20f, 0.5f, 10f);
    public Vector2 obstacleScaleRange = new Vector2(1.5f, 10f);
    [Min(0f)] public float rightSideClearance = 0f;
    [Min(0f)] public float obstaclePadding = 0.25f;
    [Min(0f)] public float headExclusionRadius = 5f;
    [Min(1)] public int spawnAttemptsPerObstacle = 100;

    private readonly List<GameObject> spawnedObstacles = new List<GameObject>();
    private readonly Color[] groundColors =
    {
        new Color(0.5f, 0.4f, 0.3f),
        new Color(0.4f, 0.3f, 0.2f),
        new Color(0.18f, 0.42f, 0.11f),
        new Color(0.62f, 0.64f, 0.42f),
        new Color(0f, 0.6f, 0.09f)
    };

    private string outputRoot;
    private string batchFolder;
    private int currentBatch;
    private int trialCounter;
    private Material originalSkybox;
    private Material runtimeSkybox;

    [Serializable]
    private class DatasetFormat
    {
        public int formatVersion;
        public string laterality;
        public int whiskerCount;
        public int sourceFrameCount;
        public string tactileFile;
        public string tactileOrdering;
        public string[] tactileColumns;
        public string imageFile;
        public int imageWidth;
        public int imageHeight;
        public string imageProjection;
        public float imageFieldOfViewDegrees;
        public Vector3 rightEyeLocalPosition;
        public float rightEyeAzimuthDegrees;
        public float rightEyeElevationDegrees;
        public string[] whiskerNames;
        public int trialsPerBatch;
        public int baseRandomSeed;
    }

    IEnumerator Start()
    {
        Application.runInBackground = true;

        if (!ValidateConfiguration())
            yield break;

        while (!whiskerManager.IsReady && !whiskerManager.InitializationFailed)
            yield return null;

        if (whiskerManager.InitializationFailed)
        {
            Debug.LogError("[TrialManager] Right-whisker initialization failed.");
            yield break;
        }

        whiskerManager.animate = false;
        InitializeRuntimeSkybox();
        outputRoot = ResolveOutputRoot();
        Directory.CreateDirectory(outputRoot);

        currentBatch = FindFirstIncompleteBatch();
        trialCounter = Mathf.Min(currentBatch * trialsPerBatch, totalTrials);

        string completionMarker = Path.Combine(outputRoot, "dataset_complete.txt");
        if (trialCounter < totalTrials && File.Exists(completionMarker))
            File.Delete(completionMarker);

        Transform head = GameObject.Find("HeadOrigin")?.transform;
        Vector3 headStartPosition = head != null ? head.position : Vector3.zero;
        Quaternion headStartRotation = head != null ? head.rotation : Quaternion.identity;

        Debug.Log(
            $"[TrialManager] Right-only run: {trialCounter}/{totalTrials} trials already " +
            $"archived; output={outputRoot}"
        );

        while (trialCounter < totalTrials)
        {
            currentBatch = trialCounter / trialsPerBatch;
            PrepareBatchFolder();

            int trialNumber = trialCounter + 1;
            string finalTrialFolder = GetTrialFolder(trialNumber, partial: false);

            if (!IsTrialComplete(finalTrialFolder))
            {
                if (Directory.Exists(finalTrialFolder))
                    DeleteGeneratedDirectory(finalTrialFolder);

                string partialTrialFolder = GetTrialFolder(trialNumber, partial: true);
                if (Directory.Exists(partialTrialFolder))
                    DeleteGeneratedDirectory(partialTrialFolder);

                bool hadPreviousObstacles = spawnedObstacles.Count > 0;
                ClearSpawnedObstacles();
                if (hadPreviousObstacles)
                    yield return null;

                UnityEngine.Random.InitState(unchecked(baseRandomSeed + trialNumber));

                recorder.SetOutputPath(partialTrialFolder);
                recorder.frameCap = WhiskerManager.SourceFrameCount;
                if (!recorder.StartRecording(manualCapture: true))
                {
                    Debug.LogError($"[TrialManager] Failed to start trial {trialNumber}.");
                    yield break;
                }

                ConfigureTrialEnvironment(
                    head,
                    headStartPosition,
                    headStartRotation
                );

                if (!SpawnRightSideObstacles(head))
                {
                    recorder.StopRecording();
                    Debug.LogError(
                        $"[TrialManager] Could not construct trial {trialNumber}."
                    );
                    yield break;
                }

                bool capturedAllFrames = true;
                for (int sourceFrame = 0;
                     sourceFrame < WhiskerManager.SourceFrameCount;
                     sourceFrame++)
                {
                    recorder.ResetContacts();
                    whiskerManager.SetAllWhiskersToFrame(sourceFrame);
                    Physics.SyncTransforms();
                    whiskerManager.EvaluateContactsAgainstStimuli();

                    if (!recorder.CaptureFrame())
                    {
                        capturedAllFrames = false;
                        Debug.LogError(
                            $"[TrialManager] Trial {trialNumber} failed at source pose " +
                            $"{sourceFrame}."
                        );
                        break;
                    }
                }

                recorder.StopRecording();

                if (!capturedAllFrames || !IsTrialComplete(partialTrialFolder))
                {
                    Debug.LogError(
                        $"[TrialManager] Trial {trialNumber} is incomplete; its partial " +
                        "folder was retained for diagnosis."
                    );
                    yield break;
                }

                Directory.Move(partialTrialFolder, finalTrialFolder);
            }

            trialCounter = trialNumber;

            if (trialCounter % progressLogInterval == 0 ||
                trialCounter == totalTrials)
            {
                Debug.Log(
                    $"[TrialManager] Completed {trialCounter}/{totalTrials} trials."
                );
            }

            bool batchIsFull = trialCounter % trialsPerBatch == 0;
            bool runIsComplete = trialCounter == totalTrials;
            if (batchIsFull || runIsComplete)
            {
                if (!ArchiveCurrentBatch())
                    yield break;

                yield return Resources.UnloadUnusedAssets();
                GC.Collect();
            }

            // Keep the editor/player responsive without tying whisker angles to
            // rendered-frame timing.
            yield return null;
        }

        ClearSpawnedObstacles();
        ReleaseRuntimeSkybox();
        File.WriteAllText(
            completionMarker,
            $"format_version={OutputFormatVersion}\n" +
            $"completed_trials={trialCounter}\n" +
            $"whiskers={WhiskerManager.RightWhiskerCount}\n" +
            $"laterality=right\n"
        );
        Debug.Log("[TrialManager] RIGHT_ONLY_DATASET_COMPLETE");
    }

    private bool ValidateConfiguration()
    {
        if (whiskerManager == null || recorder == null || rightEyeCamera == null)
        {
            Debug.LogError(
                "[TrialManager] WhiskerManager, VideoWhiskerRecorder, and " +
                "RightEyeCamera references are required."
            );
            return false;
        }

        if (totalTrials < 1 || trialsPerBatch < 1)
        {
            Debug.LogError("[TrialManager] Trial and batch counts must be positive.");
            return false;
        }

        if (rightEyeFieldOfViewDegrees <= 0f ||
            rightEyeFieldOfViewDegrees >= 180f)
        {
            Debug.LogError("[TrialManager] Right-eye field of view must be between 0 and 180 degrees.");
            return false;
        }

        if (obstaclePositionMax.x <= rightSideClearance)
        {
            Debug.LogError(
                "[TrialManager] The obstacle X range does not include the right side."
            );
            return false;
        }

        if (minimumObstacles < 1 || maximumObstacles > 5 || maximumObstacles < minimumObstacles || spawnAttemptsPerObstacle < 1)
        {
            Debug.LogError("[TrialManager] Invalid count range (expected 1–5) or placement attempt limit.");
            return false;
        }

        if (obstacleScaleRange.x <= 0f ||
            obstacleScaleRange.y < obstacleScaleRange.x)
        {
            Debug.LogError("[TrialManager] Obstacle scale range is invalid.");
            return false;
        }

        return true;
    }

    private string ResolveOutputRoot()
    {
        if (Path.IsPathRooted(outputDirectory))
            return Path.GetFullPath(outputDirectory);

        string projectRoot = Path.GetFullPath(
            Path.Combine(Application.dataPath, "..")
        );
        return Path.GetFullPath(Path.Combine(projectRoot, outputDirectory));
    }

    private int FindFirstIncompleteBatch()
    {
        int batchIndex = 0;
        while (true)
        {
            string zipPath = GetBatchZipPath(batchIndex);
            if (!File.Exists(zipPath))
                return batchIndex;

            if (new FileInfo(zipPath).Length == 0)
            {
                File.Delete(zipPath);
                return batchIndex;
            }

            string completedBatchFolder = GetBatchFolder(batchIndex);
            if (Directory.Exists(completedBatchFolder))
                DeleteGeneratedDirectory(completedBatchFolder);

            batchIndex++;
        }
    }

    private void PrepareBatchFolder()
    {
        batchFolder = GetBatchFolder(currentBatch);
        Directory.CreateDirectory(batchFolder);

        var format = new DatasetFormat
        {
            formatVersion = OutputFormatVersion,
            laterality = "right",
            whiskerCount = WhiskerManager.RightWhiskerCount,
            sourceFrameCount = WhiskerManager.SourceFrameCount,
            tactileFile = VideoWhiskerRecorder.TactileFileName,
            tactileOrdering = "frame-major, then whiskerNames order",
            tactileColumns = new[] { "s", "theta_deg" },
            imageFile = VideoWhiskerRecorder.RightImageFileName,
            imageWidth = recorder.outW,
            imageHeight = recorder.outH,
            imageProjection = "standard perspective projection",
            imageFieldOfViewDegrees = rightEyeFieldOfViewDegrees,
            rightEyeLocalPosition = rightEyeLocalPosition,
            rightEyeAzimuthDegrees = rightEyeAzimuthDegrees,
            rightEyeElevationDegrees = rightEyeElevationDegrees,
            whiskerNames = whiskerManager.whiskerNames.ToArray(),
            trialsPerBatch = trialsPerBatch,
            baseRandomSeed = baseRandomSeed
        };

        File.WriteAllText(
            Path.Combine(batchFolder, "dataset_format.json"),
            JsonUtility.ToJson(format, prettyPrint: true)
        );
    }

    private string GetBatchFolder(int batchIndex)
    {
        return Path.Combine(outputRoot, $"batch_{batchIndex}");
    }

    private string GetBatchZipPath(int batchIndex)
    {
        return Path.Combine(outputRoot, $"batch_{batchIndex}.zip");
    }

    private string GetTrialFolder(int trialNumber, bool partial)
    {
        string suffix = partial ? ".partial" : string.Empty;
        return Path.Combine(batchFolder, $"trial_{trialNumber}{suffix}");
    }

    private bool IsTrialComplete(string trialFolder)
    {
        if (!Directory.Exists(trialFolder))
            return false;

        string imagePath = Path.Combine(
            trialFolder,
            VideoWhiskerRecorder.RightImageFileName
        );
        string tactilePath = Path.Combine(
            trialFolder,
            VideoWhiskerRecorder.TactileFileName
        );

        if (!File.Exists(imagePath) ||
            new FileInfo(imagePath).Length <= 8 ||
            !File.Exists(tactilePath))
        {
            return false;
        }

        int expectedRows =
            WhiskerManager.SourceFrameCount * WhiskerManager.RightWhiskerCount;
        int rows = 0;
        using (var reader = new StreamReader(tactilePath))
        {
            while (reader.ReadLine() != null)
            {
                rows++;
                if (rows > expectedRows)
                    return false;
            }
        }

        return rows == expectedRows;
    }

    private bool ArchiveCurrentBatch()
    {
        string zipPath = GetBatchZipPath(currentBatch);
        string partialZipPath = zipPath + ".partial";

        try
        {
            if (File.Exists(zipPath))
            {
                Debug.LogError(
                    $"[TrialManager] Refusing to overwrite existing archive {zipPath}."
                );
                return false;
            }

            if (File.Exists(partialZipPath))
                File.Delete(partialZipPath);

            ZipFile.CreateFromDirectory(
                batchFolder,
                partialZipPath,
                System.IO.Compression.CompressionLevel.Optimal,
                includeBaseDirectory: false
            );

            if (!File.Exists(partialZipPath) ||
                new FileInfo(partialZipPath).Length == 0)
            {
                Debug.LogError(
                    $"[TrialManager] Archive verification failed for batch {currentBatch}."
                );
                return false;
            }

            int expectedTrials = Mathf.Min(
                trialsPerBatch,
                totalTrials - currentBatch * trialsPerBatch
            );
            if (!VerifyArchive(partialZipPath, expectedTrials))
            {
                Debug.LogError(
                    $"[TrialManager] Archive content verification failed for batch " +
                    $"{currentBatch}."
                );
                return false;
            }

            File.Move(partialZipPath, zipPath);
            DeleteGeneratedDirectory(batchFolder);
            batchFolder = null;
            currentBatch++;
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError(
                $"[TrialManager] Failed to archive batch {currentBatch}: {exception}"
            );
            return false;
        }
    }

    private bool VerifyArchive(string archivePath, int expectedTrials)
    {
        int imageEntries = 0;
        int tactileEntries = 0;
        bool hasFormat = false;

        using (ZipArchive archive = ZipFile.OpenRead(archivePath))
        {
            foreach (ZipArchiveEntry entry in archive.Entries)
            {
                string normalizedName = entry.FullName.Replace('\\', '/');
                if (normalizedName == "dataset_format.json")
                    hasFormat = entry.Length > 0;
                else if (normalizedName.EndsWith(
                             "/" + VideoWhiskerRecorder.RightImageFileName,
                             StringComparison.Ordinal))
                    imageEntries++;
                else if (normalizedName.EndsWith(
                             "/" + VideoWhiskerRecorder.TactileFileName,
                             StringComparison.Ordinal))
                    tactileEntries++;
            }
        }

        return hasFormat &&
               imageEntries == expectedTrials &&
               tactileEntries == expectedTrials;
    }

    private void DeleteGeneratedDirectory(string path)
    {
        string fullPath = Path.GetFullPath(path);
        string rootWithSeparator =
            outputRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
            Path.DirectorySeparatorChar;

        if (!fullPath.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                $"Refusing to delete path outside generated output: {fullPath}"
            );

        Directory.Delete(fullPath, recursive: true);
    }

    private void ConfigureTrialEnvironment(
        Transform head,
        Vector3 headStartPosition,
        Quaternion headStartRotation)
    {
        if (head != null)
        {
            head.SetPositionAndRotation(headStartPosition, headStartRotation);

            HeadMovement movement = head.GetComponent<HeadMovement>();
            if (movement != null)
                movement.enabled = false;

            rightEyeCamera.transform.SetParent(head, worldPositionStays: false);
            rightEyeCamera.transform.localPosition = rightEyeLocalPosition;

            float azimuth = rightEyeAzimuthDegrees * Mathf.Deg2Rad;
            float elevation = rightEyeElevationDegrees * Mathf.Deg2Rad;
            Vector3 localOpticalAxis = new Vector3(
                Mathf.Sin(azimuth) * Mathf.Cos(elevation),
                Mathf.Sin(elevation),
                Mathf.Cos(azimuth) * Mathf.Cos(elevation)
            );
            rightEyeCamera.transform.localRotation = Quaternion.LookRotation(
                localOpticalAxis,
                Vector3.up
            );
            rightEyeCamera.fieldOfView = rightEyeFieldOfViewDegrees;
            rightEyeCamera.nearClipPlane = 0.01f;
            rightEyeCamera.farClipPlane = 100f;
        }

        if (runtimeSkybox == null)
            return;

        RenderSettings.skybox = runtimeSkybox;
        float timeOfDay = UnityEngine.Random.Range(0f, 1f);
        Color morning = new Color(0.6f, 0.5f, 0.7f);
        Color sunset = new Color(1f, 0.5f, 0.3f);

        runtimeSkybox.SetColor(
            "_SkyTint",
            Color.Lerp(morning, sunset, timeOfDay)
        );
        runtimeSkybox.SetFloat(
            "_AtmosphereThickness",
            Mathf.Lerp(0.4f, 1.5f, timeOfDay)
        );
        runtimeSkybox.SetFloat(
            "_Exposure",
            Mathf.Lerp(0.5f, 1.5f, timeOfDay)
        );
        runtimeSkybox.SetColor(
            "_GroundColor",
            groundColors[UnityEngine.Random.Range(0, groundColors.Length)]
        );
    }

    /// <summary>
    /// Applies the current scene's right-eye pose and procedural sky/ground
    /// styling without spawning the random training obstacle field.
    /// </summary>
    public void PrepareCurrentVisualEnvironment(
        Transform head,
        Vector3 headStartPosition,
        Quaternion headStartRotation)
    {
        InitializeRuntimeSkybox();
        ConfigureTrialEnvironment(head, headStartPosition, headStartRotation);
    }

    private bool SpawnRightSideObstacles(Transform head)
    {
        // Retain count/target through retries, avoiding a bias towards easy scenes.
        int count = UnityEngine.Random.Range(minimumObstacles, maximumObstacles + 1);
        int target = UnityEngine.Random.Range(0, WhiskerManager.RightWhiskerCount);
        for (int attempt = 0; attempt < spawnAttemptsPerObstacle; attempt++)
        {
            ClearSpawnedObstacles();
            if (TrySpawnRightSideObstacles(head, count, target))
                return true;
        }
        ClearSpawnedObstacles();
        return false;
    }

    private bool TrySpawnRightSideObstacles(Transform head, int count, int target)
    {
        var occupiedBounds = new List<Bounds>(count);
        if (!SpawnContactObstacle(head, target, occupiedBounds))
            return false;
        Vector3 headPosition = head != null ? head.position : Vector3.zero;

        for (int i = 1; i < count; i++)
        {
            PrimitiveType shape =
                UnityEngine.Random.value < 0.5f
                    ? PrimitiveType.Cube
                    : PrimitiveType.Sphere;
            float scale = UnityEngine.Random.Range(
                obstacleScaleRange.x,
                obstacleScaleRange.y
            );

            // scale is the primitive diameter. Keeping the centre at least one
            // radius past the midline ensures the whole shape is on the right.
            float minimumRightX = Mathf.Max(
                obstaclePositionMin.x,
                rightSideClearance + 0.5f * scale
            );
            if (minimumRightX > obstaclePositionMax.x)
            {
                Debug.LogError(
                    $"[TrialManager] Shape scale {scale:F2} cannot fit in the " +
                    "configured right-side X range."
                );
                return false;
            }

            bool foundPlacement = false;
            Vector3 worldPosition = Vector3.zero;
            Bounds candidateBounds = default;

            for (int attempt = 0; attempt < spawnAttemptsPerObstacle; attempt++)
            {
                Vector3 localPosition = new Vector3(
                    UnityEngine.Random.Range(minimumRightX, obstaclePositionMax.x),
                    UnityEngine.Random.Range(
                        obstaclePositionMin.y,
                        obstaclePositionMax.y
                    ),
                    UnityEngine.Random.Range(
                        obstaclePositionMin.z,
                        obstaclePositionMax.z
                    )
                );

                worldPosition =
                    head != null
                        ? head.TransformPoint(localPosition)
                        : headPosition + localPosition;
                candidateBounds = new Bounds(worldPosition, Vector3.one * scale);

                if (candidateBounds.SqrDistance(headPosition) <
                    headExclusionRadius * headExclusionRadius)
                {
                    continue;
                }

                bool overlapsExisting = false;
                foreach (Bounds occupied in occupiedBounds)
                {
                    Bounds padded = occupied;
                    padded.Expand(obstaclePadding * 2f);
                    if (padded.Intersects(candidateBounds))
                    {
                        overlapsExisting = true;
                        break;
                    }
                }

                if (overlapsExisting)
                    continue;

                foundPlacement = true;
                break;
            }

            if (!foundPlacement)
            {
                Debug.LogWarning(
                    $"[TrialManager] Could not place right-side obstacle {i} after " +
                    $"{spawnAttemptsPerObstacle} attempts; retrying the scene."
                );
                return false;
            }

            GameObject obstacle = GameObject.CreatePrimitive(shape);
            obstacle.name = $"Obstacle_{i}";
            obstacle.transform.position = worldPosition;
            obstacle.transform.localScale = Vector3.one * scale;
            obstacle.tag = "Stimulus";

            int stimulusLayer = LayerMask.NameToLayer("Stimulus");
            if (stimulusLayer >= 0)
                obstacle.layer = stimulusLayer;

            Collider collider = obstacle.GetComponent<Collider>();
            collider.isTrigger = false;

            spawnedObstacles.Add(obstacle);
            occupiedBounds.Add(candidateBounds);
        }

        return spawnedObstacles.Count == count;
    }

    private bool SpawnContactObstacle(Transform head, int target, List<Bounds> occupied)
    {
        if (!whiskerManager.TryGetWhisker(whiskerManager.whiskerNames[target], out Whisker whisker))
            return false;
        for (int attempt = 0; attempt < spawnAttemptsPerObstacle * 100; attempt++)
        {
            int frame = UnityEngine.Random.Range(0, WhiskerManager.SourceFrameCount);
            Vector3[] curve = whiskerManager.GetWhiskerPositions(frame, target);
            // Match Whisker.SampleAlong: continuous interpolation across every
            // source interval, with no discrete position bins or spatial gaps.
            float u = UnityEngine.Random.value * (curve.Length - 1);
            int segment = Mathf.Min(Mathf.FloorToInt(u), curve.Length - 2);
            Vector3 point = Vector3.Lerp(curve[segment], curve[segment + 1], u - segment);
            float scale = UnityEngine.Random.Range(obstacleScaleRange.x, obstacleScaleRange.y);
            // Both primitives contain this ball; the sampled point stays inside.
            Vector3 local = point + UnityEngine.Random.insideUnitSphere * (scale * 0.475f);
            if (local.x - scale * 0.5f < rightSideClearance)
                continue;
            Vector3 world = head != null ? head.TransformPoint(local) : local;
            Bounds bounds = new Bounds(world, Vector3.one * scale);
            Vector3 headPosition = head != null ? head.position : Vector3.zero;
            if (bounds.SqrDistance(headPosition) < headExclusionRadius * headExclusionRadius)
                continue;

            GameObject obstacle = GameObject.CreatePrimitive(
                UnityEngine.Random.value < 0.5f ? PrimitiveType.Cube : PrimitiveType.Sphere);
            obstacle.name = "Obstacle_0";
            obstacle.transform.position = world;
            obstacle.transform.localScale = Vector3.one * scale;
            obstacle.tag = "Stimulus";
            int layer = LayerMask.NameToLayer("Stimulus");
            if (layer >= 0)
                obstacle.layer = layer;
            obstacle.GetComponent<Collider>().isTrigger = false;
            spawnedObstacles.Add(obstacle);

            recorder.ResetContacts();
            whiskerManager.SetAllWhiskersToFrame(frame);
            Physics.SyncTransforms();
            whiskerManager.EvaluateContactsAgainstStimuli();
            bool valid = whisker.HasContact();
            recorder.ResetContacts();
            if (valid)
            {
                occupied.Add(bounds);
                return true;
            }
            ClearSpawnedObstacles();
        }
        Debug.LogError($"[TrialManager] Contact placement failed for {whiskerManager.whiskerNames[target]}.");
        return false;
    }

    private void ClearSpawnedObstacles()
    {
        foreach (GameObject obstacle in spawnedObstacles)
        {
            if (obstacle != null)
            {
                // Destroy is deferred: remove rejected objects from queries now.
                obstacle.SetActive(false);
                Destroy(obstacle);
            }
        }
        spawnedObstacles.Clear();
    }

    private void InitializeRuntimeSkybox()
    {
        if (proceduralSkybox == null || runtimeSkybox != null)
            return;

        originalSkybox = RenderSettings.skybox;
        runtimeSkybox = new Material(proceduralSkybox)
        {
            name = proceduralSkybox.name + " (Runtime)"
        };
        RenderSettings.skybox = runtimeSkybox;
    }

    private void ReleaseRuntimeSkybox()
    {
        if (runtimeSkybox == null)
            return;

        if (RenderSettings.skybox == runtimeSkybox)
            RenderSettings.skybox = originalSkybox;

        Destroy(runtimeSkybox);
        runtimeSkybox = null;
    }

    void OnDestroy()
    {
        ReleaseRuntimeSkybox();
    }
}
