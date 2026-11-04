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

            // Smoke parameters apply to the unsaved scene for this batch process.
            trialManager.totalTrials = 30;
            trialManager.trialsPerBatch = 30;
            trialManager.progressLogInterval = 1;
            trialManager.minimumObstacles = 1;
            trialManager.maximumObstacles = 5;
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
            for (int trial = 1; trial <= 30; trial++)
            {
                ZipArchiveEntry image = archive.GetEntry(
                    $"trial_{trial}/" + VideoWhiskerRecorder.RightImageFileName
                );
                ZipArchiveEntry tactile = archive.GetEntry(
                    $"trial_{trial}/" + VideoWhiskerRecorder.TactileFileName
                );
                ZipArchiveEntry format = archive.GetEntry("dataset_format.json");

                if (image == null || image.Length <= 8)
                    throw new InvalidDataException("Right-eye PNG is missing or empty.");
                using (var header = new BinaryReader(image.Open()))
                {
                    byte[] png = header.ReadBytes(24);
                    if (png.Length != 24 || png[0] != 137 || png[1] != 80 ||
                        png[16] != 0 || png[17] != 0 || png[18] != 0 || png[19] != 64 ||
                        png[20] != 0 || png[21] != 0 || png[22] != 0 || png[23] != 64)
                        throw new InvalidDataException("Paired image is not a 64 x 64 PNG.");
                }
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
                bool hasContact = false;
                using (var reader = new StreamReader(tactile.Open()))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        rows++;
                        if (float.Parse(line.Split(',')[0], System.Globalization.CultureInfo.InvariantCulture) > 0f)
                            hasContact = true;
                    }
                }

                int expectedRows =
                    WhiskerManager.SourceFrameCount *
                    WhiskerManager.RightWhiskerCount;
                if (!hasContact)
                    throw new InvalidDataException("Contact-centred trial has no recorded physical contact.");
                if (rows != expectedRows)
                {
                    throw new InvalidDataException(
                        $"Tactile CSV has {rows} rows; expected {expectedRows}."
                    );
                }
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
