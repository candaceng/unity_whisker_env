using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class PassiveSweepLogger : MonoBehaviour
{
    private string path;
    private StreamWriter sw;
    public WhiskerManager whiskerManager;

    private bool recording = false;
    private bool manuallyDriven = false;
    private bool initializedFromManager = false;
    public bool IsRecording => recording;
    public int CapturedFrames => trialFrame;

    public int maxFrames = 51;
    private int trialFrame = 0;

    public List<string> whiskerNames = new List<string>();
    public Dictionary<string, Whisker> whiskerMap = new Dictionary<string, Whisker>();

    public void SetOutput(string filepath)
    {
        path = filepath;
    }

    private void InitWhiskers()
    {
        if (whiskerManager != null &&
            whiskerManager.IsReady &&
            !initializedFromManager)
        {
            whiskerNames = new List<string>(whiskerManager.whiskerNames);
            initializedFromManager = true;
            Debug.Log($"[PassiveSweepLogger] Loaded {whiskerNames.Count} whisker names from WhiskerManager.");
        }

        if (whiskerMap == null) whiskerMap = new Dictionary<string, Whisker>();
        if (whiskerMap.Count == 0)
        {
            whiskerMap.Clear();
            foreach (var w in FindObjectsByType<Whisker>(FindObjectsSortMode.None))
                whiskerMap[w.name] = w;
            Debug.Log($"[PassiveSweepLogger] Built whiskerMap with {whiskerMap.Count} Whisker components.");
        }

        if (whiskerNames == null || whiskerNames.Count == 0)
        {
            whiskerNames = new List<string>(whiskerMap.Keys);
            whiskerNames.Sort();
            Debug.Log($"[PassiveSweepLogger] Fallback whiskerNames from scene: {whiskerNames.Count}");
        }
    }

    public void BeginTrial(bool manualCapture = false)
    {
        InitWhiskers();
        Debug.Log($"[PassiveSweepLogger] BeginTrial {path} names={whiskerNames.Count} map={whiskerMap.Count}");

        if (whiskerNames.Count != WhiskerManager.RightWhiskerCount)
        {
            Debug.LogError(
                $"[PassiveSweepLogger] Expected {WhiskerManager.RightWhiskerCount} " +
                $"right whiskers, found {whiskerNames.Count}."
            );
            return;
        }

        if (string.IsNullOrEmpty(path))
        {
            Debug.LogError("[PassiveSweepLogger] Path not set. Call SetOutput(...) before BeginTrial().");
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        sw = new StreamWriter(path, false);
        trialFrame = 0;
        manuallyDriven = manualCapture;
        ResetContacts();
        recording = true;
    }

    void LateUpdate()
    {
        if (!recording || manuallyDriven) return;
        CaptureFrame();
    }

    public bool CaptureFrame()
    {
        if (!recording || sw == null) return false;

        if (trialFrame >= maxFrames)
        {
            EndTrial();
            return false;
        }

        foreach (string whiskerName in whiskerNames)
        {
            float s = 0f;
            float theta = 0f;

            if (whiskerMap.TryGetValue(whiskerName, out Whisker w) && w != null)
            {
                s = w.HasContact() ? w.SContact : 0f;
                theta = w.thetaWDeg;
            }

            // Preserve exactly 30 ordered right-whisker rows per pose, even if a scene
            // reference is unexpectedly missing.
            sw.WriteLine(string.Join(",",
                s.ToString("F4", CultureInfo.InvariantCulture),
                theta.ToString("F4", CultureInfo.InvariantCulture)
            ));
        }

        ResetContacts();
        trialFrame++;
        return true;
    }

    public void ResetContacts()
    {
        InitWhiskers();
        foreach (var kv in whiskerMap)
        {
            if (kv.Value != null)
                kv.Value.ResetContactInfo();
        }
    }

    public void EndTrial()
    {
        recording = false;

        sw?.Flush();
        sw?.Dispose();
        sw?.Close();
        sw = null;
        manuallyDriven = false;

        Debug.Log($"[PassiveSweepLogger] Saved {path}");
    }
}
