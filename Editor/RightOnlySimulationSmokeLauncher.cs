using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class RightOnlySimulationSmokeLauncher
{
    private const string ScenePath = "Assets/Scenes/ShapeNetScene.unity";
    private const string ActiveKey = "RightOnlySmokeV2.Active";
    private const string StartTicksKey = "RightOnlySmokeV2.StartTicks";
    private static string SmokeRoot =>
        Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", "tmp", "right_only_runtime_smoke")
        );

    [InitializeOnLoadMethod]
    private static void ResumeAfterDomainReload()
    {
        if (SessionState.GetBool(ActiveKey, false))
            EditorApplication.update += Poll;
    }

    public static void Run()
    {
        try
        {
            SessionState.SetBool(ActiveKey, false);
            DeleteSmokeRootIfPresent();

            Scene scene = EditorSceneManager.OpenScene(
                ScenePath,
                OpenSceneMode.Single
            );
            TrialManager trialManager = scene
                .GetRootGameObjects()
                .SelectMany(root =>
                    root.GetComponentsInChildren<TrialManager>(includeInactive: true))
                .Single();

            // These values live only in the unsaved editor scene used by this
            // batch process. The checked-in 100,000-trial settings stay intact.
            trialManager.totalTrials = 1;
            trialManager.trialsPerBatch = 1;
            trialManager.progressLogInterval = 1;
            trialManager.numObstacles = 1;
            trialManager.outputDirectory = SmokeRoot;
            trialManager.enabled = true;
            EditorSceneManager.MarkSceneDirty(scene);

            SessionState.SetBool(ActiveKey, true);
            SessionState.SetString(
                StartTicksKey,
                DateTime.UtcNow.Ticks.ToString()
            );
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
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
            if (HasTimedOut())
                throw new TimeoutException("Right-only runtime smoke test timed out.");

            string completionMarker = Path.Combine(
                SmokeRoot,
                "dataset_complete.txt"
            );
            if (!File.Exists(completionMarker))
                return;

            if (EditorApplication.isPlaying)
            {
                EditorApplication.ExitPlaymode();
                return;
            }

            VerifySmokeArchive();
            DeleteSmokeRootIfPresent();
            SessionState.SetBool(ActiveKey, false);
            EditorApplication.update -= Poll;
            Debug.Log("[RightOnlySmoke] RIGHT_ONLY_RUNTIME_SMOKE_VALID");
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Fail(exception);
        }
    }

    private static bool HasTimedOut()
    {
        string rawTicks = SessionState.GetString(StartTicksKey, string.Empty);
        if (!long.TryParse(rawTicks, out long ticks))
            return true;

        return DateTime.UtcNow - new DateTime(ticks, DateTimeKind.Utc) >
               TimeSpan.FromMinutes(3);
    }

    private static void VerifySmokeArchive()
    {
        string archivePath = Path.Combine(SmokeRoot, "batch_0.zip");
        if (!File.Exists(archivePath))
            throw new FileNotFoundException("Smoke archive was not created.", archivePath);

        using (ZipArchive archive = ZipFile.OpenRead(archivePath))
        {
            ZipArchiveEntry image = archive.GetEntry(
                "trial_1/" + VideoWhiskerRecorder.RightImageFileName
            );
            ZipArchiveEntry tactile = archive.GetEntry(
                "trial_1/" + VideoWhiskerRecorder.TactileFileName
            );
            ZipArchiveEntry format = archive.GetEntry("dataset_format.json");

            if (image == null || image.Length <= 8)
                throw new InvalidDataException("Right-eye PNG is missing or empty.");
            if (format == null || format.Length == 0)
                throw new InvalidDataException("Dataset format manifest is missing.");
            if (archive.Entries.Any(entry =>
                    entry.FullName.IndexOf(
                        "left",
                        StringComparison.OrdinalIgnoreCase
                    ) >= 0))
            {
                throw new InvalidDataException("Smoke archive contains a left-side file.");
            }
            if (tactile == null)
                throw new InvalidDataException("Tactile CSV is missing.");

            int rows = 0;
            using (var reader = new StreamReader(tactile.Open()))
            {
                while (reader.ReadLine() != null)
                    rows++;
            }

            int expectedRows =
                WhiskerManager.SourceFrameCount *
                WhiskerManager.RightWhiskerCount;
            if (rows != expectedRows)
            {
                throw new InvalidDataException(
                    $"Tactile CSV has {rows} rows; expected {expectedRows}."
                );
            }
        }
    }

    private static void DeleteSmokeRootIfPresent()
    {
        if (!Directory.Exists(SmokeRoot))
            return;

        string allowedParent = Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", "tmp")
        ).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
            Path.DirectorySeparatorChar;

        if (!SmokeRoot.StartsWith(
                allowedParent,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Refusing to delete unexpected smoke path: {SmokeRoot}"
            );
        }

        Directory.Delete(SmokeRoot, recursive: true);
    }

    private static void Fail(Exception exception)
    {
        Debug.LogError($"[RightOnlySmoke] Runtime smoke test failed: {exception}");
        SessionState.SetBool(ActiveKey, false);
        EditorApplication.update -= Poll;

        if (EditorApplication.isPlaying)
            EditorApplication.ExitPlaymode();

        EditorApplication.Exit(1);
    }
}
