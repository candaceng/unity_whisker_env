using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RightOnlyProductionLauncher
{
    private const string ScenePath = "Assets/Scenes/ShapeNetScene.unity";
    private const int TotalTrials = 100000;
    private const int TrialsPerBatch = 1000;
    private const string OutputDirectory = "Data/RecordedTrials";

    private const string ActiveKey = "RightOnlyProduction.Active";
    private const string SawPlayModeKey = "RightOnlyProduction.SawPlayMode";
    private const string StartTicksKey = "RightOnlyProduction.StartTicks";
    private const string FailureKey = "RightOnlyProduction.Failure";

    private static string ProjectRoot =>
        Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    private static string OutputRoot =>
        Path.GetFullPath(Path.Combine(ProjectRoot, OutputDirectory));

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

    public static void Validate()
    {
        try
        {
            TrialManager manager = OpenAndConfigureScene();
            int firstIncompleteBatch = ValidateExistingArchives();
            int nextTrial = firstIncompleteBatch * manager.trialsPerBatch + 1;
            Debug.Log(
                $"[RightOnlyProduction] VALID: {firstIncompleteBatch} complete " +
                $"batches; next trial={nextTrial}."
            );
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                $"[RightOnlyProduction] Validation failed: {exception}"
            );
            EditorApplication.Exit(1);
        }
    }

    public static void Run()
    {
        try
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                throw new InvalidOperationException(
                    "Cannot start production while the editor is in play mode."
                );
            }

            ResetSessionState();
            TrialManager manager = OpenAndConfigureScene();
            int firstIncompleteBatch = ValidateExistingArchives();
            int completedTrials = firstIncompleteBatch * manager.trialsPerBatch;

            if (completedTrials >= manager.totalTrials)
            {
                VerifyCompleteRun();
                Debug.Log(
                    "[RightOnlyProduction] Dataset is already complete."
                );
                EditorApplication.Exit(0);
                return;
            }

            string completionMarker = GetCompletionMarkerPath();
            if (File.Exists(completionMarker))
                File.Delete(completionMarker);

            SessionState.SetBool(ActiveKey, true);
            SessionState.SetBool(SawPlayModeKey, false);
            SessionState.SetString(FailureKey, string.Empty);
            SessionState.SetString(
                StartTicksKey,
                DateTime.UtcNow.Ticks.ToString(CultureInfo.InvariantCulture)
            );

            Application.logMessageReceived -= ObserveLog;
            Application.logMessageReceived += ObserveLog;
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;

            Debug.Log(
                $"[RightOnlyProduction] Starting unattended generation at trial " +
                $"{completedTrials + 1}; target={manager.totalTrials}; " +
                $"batchSize={manager.trialsPerBatch}; output={OutputRoot}"
            );
            EditorApplication.EnterPlaymode();
        }
        catch (Exception exception)
        {
            Fail(exception);
        }
    }

    private static TrialManager OpenAndConfigureScene()
    {
        Scene scene = EditorSceneManager.OpenScene(
            ScenePath,
            OpenSceneMode.Single
        );
        TrialManager manager = scene
            .GetRootGameObjects()
            .SelectMany(root =>
                root.GetComponentsInChildren<TrialManager>(includeInactive: true))
            .Single();

        manager.totalTrials = TotalTrials;
        manager.trialsPerBatch = TrialsPerBatch;
        manager.progressLogInterval = 100;
        manager.outputDirectory = OutputDirectory;
        manager.enabled = true;
        EditorSceneManager.MarkSceneDirty(scene);
        return manager;
    }

    private static int ValidateExistingArchives()
    {
        Directory.CreateDirectory(OutputRoot);
        var archiveIndices = new SortedSet<int>();

        foreach (string archivePath in Directory.GetFiles(
                     OutputRoot,
                     "batch_*.zip",
                     SearchOption.TopDirectoryOnly))
        {
            string stem = Path.GetFileNameWithoutExtension(archivePath);
            string rawIndex = stem.Substring("batch_".Length);
            if (!int.TryParse(rawIndex, out int batchIndex) || batchIndex < 0)
            {
                throw new InvalidDataException(
                    $"Invalid batch archive name: {archivePath}"
                );
            }
            archiveIndices.Add(batchIndex);
        }

        int firstIncompleteBatch = 0;
        while (archiveIndices.Contains(firstIncompleteBatch))
            firstIncompleteBatch++;

        if (archiveIndices.Any(index => index > firstIncompleteBatch))
        {
            throw new InvalidDataException(
                $"Archive sequence has a gap at batch_{firstIncompleteBatch}.zip."
            );
        }

        if (firstIncompleteBatch > TotalTrials / TrialsPerBatch)
        {
            throw new InvalidDataException(
                $"Found more than {TotalTrials / TrialsPerBatch} batch archives."
            );
        }

        for (int batchIndex = 0;
             batchIndex < firstIncompleteBatch;
             batchIndex++)
        {
            ValidateArchive(
                Path.Combine(OutputRoot, $"batch_{batchIndex}.zip")
            );
        }

        return firstIncompleteBatch;
    }

    private static void ValidateArchive(string archivePath)
    {
        if (new FileInfo(archivePath).Length == 0)
            throw new InvalidDataException($"Empty archive: {archivePath}");

        int images = 0;
        int tactileFiles = 0;
        bool hasManifest = false;

        using (ZipArchive archive = ZipFile.OpenRead(archivePath))
        {
            foreach (ZipArchiveEntry entry in archive.Entries)
            {
                string name = entry.FullName.Replace('\\', '/');
                if (name == "dataset_format.json")
                    hasManifest = entry.Length > 0;
                else if (name.EndsWith(
                             "/" + VideoWhiskerRecorder.RightImageFileName,
                             StringComparison.Ordinal))
                    images++;
                else if (name.EndsWith(
                             "/" + VideoWhiskerRecorder.TactileFileName,
                             StringComparison.Ordinal))
                    tactileFiles++;

                if (name.IndexOf(
                        "left",
                        StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    throw new InvalidDataException(
                        $"Left-side entry found in {archivePath}: {name}"
                    );
                }
            }
        }

        if (!hasManifest ||
            images != TrialsPerBatch ||
            tactileFiles != TrialsPerBatch)
        {
            throw new InvalidDataException(
                $"Invalid archive {archivePath}: manifest={hasManifest}, " +
                $"images={images}, tactile={tactileFiles}."
            );
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
            {
                throw new TimeoutException(
                    "Right-only production run exceeded eight hours."
                );
            }

            if (EditorApplication.isPlaying)
                SessionState.SetBool(SawPlayModeKey, true);

            if (HasExpectedCompletionMarker())
            {
                if (EditorApplication.isPlaying)
                {
                    EditorApplication.ExitPlaymode();
                    return;
                }

                VerifyCompleteRun();
                CompleteSuccessfully();
                return;
            }

            bool sawPlayMode = SessionState.GetBool(SawPlayModeKey, false);
            if (sawPlayMode &&
                !EditorApplication.isPlayingOrWillChangePlaymode)
            {
                throw new InvalidOperationException(
                    "Play mode stopped before the dataset completed."
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

        string[] productionPrefixes =
        {
            "[TrialManager]",
            "[VideoWhiskerRecorder]",
            "[WhiskerManager]"
        };
        if (productionPrefixes.Any(prefix => condition.Contains(prefix)))
            SessionState.SetString(FailureKey, condition);
    }

    private static bool HasExpectedCompletionMarker()
    {
        string markerPath = GetCompletionMarkerPath();
        if (!File.Exists(markerPath))
            return false;

        string contents = File.ReadAllText(markerPath);
        return contents.Contains($"completed_trials={TotalTrials}") &&
               contents.Contains("whiskers=30") &&
               contents.Contains("laterality=right");
    }

    private static string GetCompletionMarkerPath()
    {
        return Path.Combine(OutputRoot, "dataset_complete.txt");
    }

    private static bool HasTimedOut()
    {
        string rawTicks = SessionState.GetString(StartTicksKey, string.Empty);
        if (!long.TryParse(
                rawTicks,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out long ticks))
        {
            return true;
        }

        return DateTime.UtcNow - new DateTime(ticks, DateTimeKind.Utc) >
               TimeSpan.FromHours(8);
    }

    private static void VerifyCompleteRun()
    {
        int firstIncompleteBatch = ValidateExistingArchives();
        int expectedBatches = TotalTrials / TrialsPerBatch;
        if (firstIncompleteBatch != expectedBatches)
        {
            throw new InvalidDataException(
                $"Expected {expectedBatches} archives, found " +
                $"{firstIncompleteBatch}."
            );
        }
        if (!HasExpectedCompletionMarker())
            throw new InvalidDataException("Completion marker is invalid.");
    }

    private static void CompleteSuccessfully()
    {
        ResetSessionState();
        Debug.Log(
            "[RightOnlyProduction] RIGHT_ONLY_PRODUCTION_COMPLETE"
        );
        EditorApplication.Exit(0);
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
        Debug.LogError(
            $"[RightOnlyProduction] Production run failed: {exception}"
        );
        ResetSessionState();

        if (EditorApplication.isPlayingOrWillChangePlaymode)
            EditorApplication.ExitPlaymode();

        EditorApplication.Exit(1);
    }
}
