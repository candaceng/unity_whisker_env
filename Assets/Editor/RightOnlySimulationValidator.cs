using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RightOnlySimulationValidator
{
    private const string ScenePath = "Assets/Scenes/ShapeNetScene.unity";

    [MenuItem("Whisker Simulation/Apply and Validate Right-Only Setup")]
    public static void ApplyAndValidate()
    {
        try
        {
            Scene scene = EditorSceneManager.OpenScene(
                ScenePath,
                OpenSceneMode.Single
            );

            Camera rightEye = FindInScene<Camera>(scene)
                .FirstOrDefault(camera => camera.name == "Right Eye Camera");
            if (rightEye == null)
                throw new InvalidOperationException("Right Eye Camera is missing.");

            foreach (Camera camera in FindInScene<Camera>(scene)
                         .Where(camera => camera.name == "Left Eye Camera")
                         .ToArray())
            {
                UnityEngine.Object.DestroyImmediate(camera.gameObject);
            }

            WhiskerManager whiskerManager =
                FindSingleInScene<WhiskerManager>(scene);
            VideoWhiskerRecorder recorder =
                FindSingleInScene<VideoWhiskerRecorder>(scene);
            TrialManager trialManager = FindSingleInScene<TrialManager>(scene);
            SweepBatchRunner sweepRunner =
                FindSingleInScene<SweepBatchRunner>(scene);

            whiskerManager.animate = false;

            recorder.enabled = true;
            recorder.rightEyeCamera = rightEye;
            recorder.whiskerManager = whiskerManager;
            recorder.frameCap = WhiskerManager.SourceFrameCount;

            trialManager.enabled = true;
            trialManager.totalTrials = 100000;
            trialManager.trialsPerBatch = 1000;
            trialManager.progressLogInterval = 100;
            trialManager.outputDirectory = "Data/RecordedTrials";
            trialManager.whiskerManager = whiskerManager;
            trialManager.recorder = recorder;
            trialManager.rightEyeCamera = rightEye;

            // Only one scene-level dataset runner may own the whiskers and
            // stimuli at a time.
            sweepRunner.enabled = false;

            EditorUtility.SetDirty(whiskerManager);
            EditorUtility.SetDirty(recorder);
            EditorUtility.SetDirty(trialManager);
            EditorUtility.SetDirty(sweepRunner);
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);

            ValidateSceneAndResources(scene);
            Debug.Log("[RightOnlyValidator] RIGHT_ONLY_SIMULATION_VALID");

            if (Application.isBatchMode)
                EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogError($"[RightOnlyValidator] Validation failed: {exception}");
            if (Application.isBatchMode)
                EditorApplication.Exit(1);
            else
                throw;
        }
    }

    [MenuItem("Whisker Simulation/Validate Right-Only Setup")]
    public static void ValidateOnly()
    {
        try
        {
            Scene scene = EditorSceneManager.OpenScene(
                ScenePath,
                OpenSceneMode.Single
            );
            ValidateSceneAndResources(scene);
            Debug.Log("[RightOnlyValidator] RIGHT_ONLY_SIMULATION_VALID");

            if (Application.isBatchMode)
                EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogError($"[RightOnlyValidator] Validation failed: {exception}");
            if (Application.isBatchMode)
                EditorApplication.Exit(1);
            else
                throw;
        }
    }

    private static void ValidateSceneAndResources(Scene scene)
    {
        var failures = new List<string>();

        Camera[] cameras = FindInScene<Camera>(scene).ToArray();
        if (cameras.Any(camera => camera.name == "Left Eye Camera"))
            failures.Add("The active scene still contains Left Eye Camera.");

        Camera rightEye = cameras.FirstOrDefault(
            camera => camera.name == "Right Eye Camera"
        );
        if (rightEye == null)
            failures.Add("The active scene does not contain Right Eye Camera.");

        WhiskerManager manager = FindSingleInScene<WhiskerManager>(scene);
        VideoWhiskerRecorder recorder =
            FindSingleInScene<VideoWhiskerRecorder>(scene);
        TrialManager trials = FindSingleInScene<TrialManager>(scene);
        SweepBatchRunner sweep = FindSingleInScene<SweepBatchRunner>(scene);

        if (manager.animate)
            failures.Add("WhiskerManager.animate must be disabled for exact pose capture.");
        if (!recorder.enabled)
            failures.Add("VideoWhiskerRecorder is disabled.");
        if (recorder.rightEyeCamera != rightEye)
            failures.Add("VideoWhiskerRecorder is not bound to Right Eye Camera.");
        if (!trials.enabled)
            failures.Add("TrialManager is disabled.");
        if (trials.totalTrials != 100000)
            failures.Add($"TrialManager totalTrials is {trials.totalTrials}, not 100000.");
        if (trials.trialsPerBatch != 1000)
            failures.Add($"TrialManager trialsPerBatch is {trials.trialsPerBatch}, not 1000.");
        if (trials.rightEyeCamera != rightEye)
            failures.Add("TrialManager is not bound to Right Eye Camera.");
        if (sweep.enabled)
            failures.Add("SweepBatchRunner must be disabled during random training.");

        ValidateWhiskerNames(failures);
        ValidateRightGeometry(failures);

        if (failures.Count > 0)
        {
            throw new InvalidOperationException(
                "Right-only validation errors:\n- " +
                string.Join("\n- ", failures)
            );
        }
    }

    private static void ValidateWhiskerNames(List<string> failures)
    {
        TextAsset namesAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(
            "Assets/Resources/param_name.csv"
        );
        if (namesAsset == null)
        {
            failures.Add("param_name.csv is missing.");
            return;
        }

        string[] rightNames = namesAsset.text
            .Split('\n')
            .Select(name => name.Trim())
            .Where(name => name.StartsWith("R", StringComparison.Ordinal))
            .ToArray();

        if (rightNames.Length != WhiskerManager.RightWhiskerCount)
        {
            failures.Add(
                $"param_name.csv has {rightNames.Length} right names, expected " +
                $"{WhiskerManager.RightWhiskerCount}."
            );
        }

        if (rightNames.Any(name => name.StartsWith("L", StringComparison.Ordinal)))
            failures.Add("The selected right-whisker name list contains a left name.");
    }

    private static void ValidateRightGeometry(List<string> failures)
    {
        int expectedRows =
            WhiskerManager.RightWhiskerCount * WhiskerManager.PointsPerWhisker;

        for (int frame = 0; frame < WhiskerManager.SourceFrameCount; frame++)
        {
            string assetPath =
                $"Assets/Resources/whisking_data/right_whiskers_frame_{frame}.csv";
            TextAsset frameAsset =
                AssetDatabase.LoadAssetAtPath<TextAsset>(assetPath);

            if (frameAsset == null)
            {
                failures.Add($"Missing right geometry: {assetPath}");
                continue;
            }

            int rows = frameAsset.text
                .Split('\n')
                .Count(line =>
                    !string.IsNullOrWhiteSpace(line) &&
                    !line.StartsWith("Frame", StringComparison.Ordinal)
                );

            if (rows != expectedRows)
            {
                failures.Add(
                    $"{assetPath} has {rows} data rows, expected {expectedRows}."
                );
            }
        }
    }

    private static T FindSingleInScene<T>(Scene scene) where T : Component
    {
        T[] matches = FindInScene<T>(scene).ToArray();
        if (matches.Length != 1)
        {
            throw new InvalidOperationException(
                $"Expected exactly one {typeof(T).Name} in {scene.path}; " +
                $"found {matches.Length}."
            );
        }

        return matches[0];
    }

    private static IEnumerable<T> FindInScene<T>(Scene scene) where T : Component
    {
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            foreach (T component in root.GetComponentsInChildren<T>(true))
                yield return component;
        }
    }
}
