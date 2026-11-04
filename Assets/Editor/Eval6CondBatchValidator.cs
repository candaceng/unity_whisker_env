using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class Eval6CondBatchValidator
{
    private const string ScenePath = "Assets/Scenes/ShapeNetScene.unity";
    private const string RelativeOutput = "Data/eval6cond_right_v3";
    private const string ActiveKey = "RightOnlyEval6.Active";
    private const string SawPlayModeKey = "RightOnlyEval6.SawPlayMode";
    private const string StartTicksKey = "RightOnlyEval6.StartTicks";
    private const string FailureKey = "RightOnlyEval6.Failure";

    private static readonly string[] Conditions =
    {
        "concave_near", "concave_med", "concave_far",
        "convex_near", "convex_med", "convex_far"
    };

    private static string OutputRoot =>
        Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", RelativeOutput)
        );

    [InitializeOnLoadMethod]
    private static void ResumeAfterDomainReload()
    {
        Application.logMessageReceived -= ObserveLog;
        Application.logMessageReceived += ObserveLog;
        if (SessionState.GetBool(ActiveKey, false))
        {
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
        }
    }

    public static void Run()
    {
        try
        {
            ResetSessionState();
            Scene scene = EditorSceneManager.OpenScene(
                ScenePath,
                OpenSceneMode.Single
            );

            foreach (TrialManager trialManager in scene
                         .GetRootGameObjects()
                         .SelectMany(root => root.GetComponentsInChildren<TrialManager>(true)))
            {
                trialManager.enabled = false;
            }

            SweepBatchRunner runner = scene
                .GetRootGameObjects()
                .SelectMany(root => root.GetComponentsInChildren<SweepBatchRunner>(true))
                .Single();
            runner.gameObject.SetActive(true);
            runner.enabled = true;
            runner.mode = SweepBatchRunner.GenMode.EVAL_6COND;

            if (runner.evalProfile == null ||
                runner.eval6CondPlan == null ||
                runner.whiskerManager == null ||
                runner.logger == null ||
                runner.arcSpawner == null)
            {
                throw new InvalidOperationException(
                    "Eval6Cond scene references are incomplete."
                );
            }

            runner.evalProfile.outputDir = RelativeOutput;
            runner.evalProfile.repeatsPerCond = 1;
            runner.evalProfile.maxFrames = WhiskerManager.SourceFrameCount;
            runner.evalProfile.labels = (string[])Conditions.Clone();

            Directory.CreateDirectory(OutputRoot);
            DeleteIfPresent(Path.Combine(OutputRoot, "dataset_complete.txt"));
            DeleteIfPresent(Path.Combine(OutputRoot, "dataset_format.json"));

            SessionState.SetBool(ActiveKey, true);
            SessionState.SetBool(SawPlayModeKey, false);
            SessionState.SetString(FailureKey, string.Empty);
            SessionState.SetString(
                StartTicksKey,
                DateTime.UtcNow.Ticks.ToString(CultureInfo.InvariantCulture)
            );
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
            EditorSceneManager.MarkSceneDirty(scene);

            Debug.Log(
                $"[RightOnlyEval6] Starting six-condition generation: {OutputRoot}"
            );
            EditorApplication.EnterPlaymode();
        }
        catch (Exception exception)
        {
            Fail(exception);
        }
    }

    private static void Poll()
    {
        if (!SessionState.GetBool(ActiveKey, false))
        {
            EditorApplication.update -= Poll;
            return;
        }

        try
        {
            string failure = SessionState.GetString(FailureKey, string.Empty);
            if (!string.IsNullOrEmpty(failure))
                throw new InvalidOperationException(failure);

            if (HasTimedOut())
                throw new TimeoutException("Right-only Eval6Cond exceeded ten minutes.");

            if (EditorApplication.isPlaying)
                SessionState.SetBool(SawPlayModeKey, true);

            if (HasCompletionMarker())
            {
                if (EditorApplication.isPlaying)
                {
                    EditorApplication.ExitPlaymode();
                    return;
                }

                VerifyOutputs();
                ResetSessionState();
                Debug.Log("[RightOnlyEval6] RIGHT_ONLY_EVAL6_VALID");
                EditorApplication.Exit(0);
                return;
            }

            if (SessionState.GetBool(SawPlayModeKey, false) &&
                !EditorApplication.isPlayingOrWillChangePlaymode)
            {
                throw new InvalidOperationException(
                    "Play mode stopped before Eval6Cond completed."
                );
            }
        }
        catch (Exception exception)
        {
            Fail(exception);
        }
    }

    private static void ObserveLog(
        string condition,
        string stackTrace,
        LogType type)
    {
        if (!SessionState.GetBool(ActiveKey, false) ||
            (type != LogType.Error &&
             type != LogType.Exception &&
             type != LogType.Assert))
        {
            return;
        }

        string[] relevantPrefixes =
        {
            "[EVAL_6COND]",
            "[PassiveSweepLogger]",
            "[WhiskerManager]",
            "[Runner]"
        };
        if (relevantPrefixes.Any(prefix => condition.Contains(prefix)))
            SessionState.SetString(FailureKey, condition);
    }

    private static void VerifyOutputs()
    {
        string manifestPath = Path.Combine(OutputRoot, "dataset_format.json");
        if (!File.Exists(manifestPath) || new FileInfo(manifestPath).Length == 0)
            throw new InvalidDataException("Eval6Cond manifest is missing.");

        string[] csvFiles = Directory.GetFiles(
            OutputRoot,
            "*.csv",
            SearchOption.TopDirectoryOnly
        );
        var expectedNames = new HashSet<string>(
            Conditions.Select(condition => $"{condition}_rep00.csv"),
            StringComparer.Ordinal
        );
        var actualNames = new HashSet<string>(
            csvFiles.Select(Path.GetFileName),
            StringComparer.Ordinal
        );
        if (!actualNames.SetEquals(expectedNames))
        {
            throw new InvalidDataException(
                $"Eval6Cond CSV set is incorrect: {string.Join(", ", actualNames)}"
            );
        }

        int expectedRows =
            WhiskerManager.RightWhiskerCount * WhiskerManager.SourceFrameCount;
        foreach (string csvPath in csvFiles)
        {
            int rows = 0;
            using (var reader = new StreamReader(csvPath))
            {
                while (reader.ReadLine() != null)
                    rows++;
            }
            if (rows != expectedRows)
            {
                throw new InvalidDataException(
                    $"{csvPath} has {rows} rows; expected {expectedRows}."
                );
            }
        }
    }

    private static bool HasCompletionMarker()
    {
        string markerPath = Path.Combine(OutputRoot, "dataset_complete.txt");
        if (!File.Exists(markerPath))
            return false;
        string contents = File.ReadAllText(markerPath);
        return contents.Contains("format_version=3") &&
               contents.Contains("completed_trials=6") &&
               contents.Contains("whiskers=30") &&
               contents.Contains("laterality=right");
    }

    private static bool HasTimedOut()
    {
        string raw = SessionState.GetString(StartTicksKey, string.Empty);
        if (!long.TryParse(
                raw,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out long ticks))
        {
            return true;
        }
        return DateTime.UtcNow - new DateTime(ticks, DateTimeKind.Utc) >
               TimeSpan.FromMinutes(10);
    }

    private static void DeleteIfPresent(string path)
    {
        if (File.Exists(path))
            File.Delete(path);
    }

    private static void ResetSessionState()
    {
        SessionState.SetBool(ActiveKey, false);
        SessionState.SetBool(SawPlayModeKey, false);
        SessionState.SetString(StartTicksKey, string.Empty);
        SessionState.SetString(FailureKey, string.Empty);
        EditorApplication.update -= Poll;
    }

    private static void Fail(Exception exception)
    {
        Debug.LogError($"[RightOnlyEval6] Failed: {exception}");
        ResetSessionState();
        if (EditorApplication.isPlayingOrWillChangePlaymode)
            EditorApplication.ExitPlaymode();
        EditorApplication.Exit(1);
    }
}
