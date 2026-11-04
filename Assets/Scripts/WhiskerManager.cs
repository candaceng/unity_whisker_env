using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

public class WhiskerManager : MonoBehaviour
{
    public const int RightWhiskerCount = 30;
    public const int SourceFrameCount = 51;
    public const int PointsPerWhisker = 100;

    [Header("Animation")]
    public bool animate = true;
    public int restingFrame = 0;

    [Header("File Paths")]
    public string paramNameCSV = "param_name";
    public string whiskingFolder = "whisking_data";
    public GameObject ratHeadObject;

    [Header("Whisker Settings")]
    public Material whiskerMaterial;

    [Header("Animation Settings")]
    public float oscillationSpeed = 1f;
    public bool smoothInterpolation = true;

    private readonly Dictionary<string, Whisker> whiskers =
        new Dictionary<string, Whisker>(RightWhiskerCount);
    private readonly Dictionary<string, int> whiskerIndices =
        new Dictionary<string, int>(RightWhiskerCount);
    private readonly Dictionary<int, List<Vector3[]>> rightFrameData =
        new Dictionary<int, List<Vector3[]>>(RightWhiskerCount);

    private float animationTimer;

    public List<string> whiskerNames = new List<string>(RightWhiskerCount);
    public bool IsReady { get; private set; }
    public bool InitializationFailed { get; private set; }
    public int WhiskerCount => whiskers.Count;

    IEnumerator Start()
    {
        IsReady = false;
        InitializationFailed = false;

        whiskerNames = LoadRightWhiskerNames();
        if (whiskerNames.Count != RightWhiskerCount)
        {
            InitializationFailed = true;
            Debug.LogError(
                $"[WhiskerManager] Expected {RightWhiskerCount} right-whisker names " +
                $"in {paramNameCSV}.csv, but found {whiskerNames.Count}."
            );
            yield break;
        }

        if (!LoadAllRightFrames())
        {
            InitializationFailed = true;
            yield break;
        }

        RebuildWhiskers();
        if (whiskers.Count != RightWhiskerCount)
        {
            InitializationFailed = true;
            Debug.LogError(
                $"[WhiskerManager] Built {whiskers.Count}/{RightWhiskerCount} right whiskers."
            );
            yield break;
        }

        if (!animate)
            SetAllWhiskersToFrame(restingFrame);

        IsReady = true;
        Debug.Log(
            $"[WhiskerManager] Ready with {whiskers.Count} right whiskers and " +
            $"{SourceFrameCount} source poses."
        );

        int whiskerLayer = LayerMask.NameToLayer("Whisker");
        if (whiskerLayer != -1)
            Physics.IgnoreLayerCollision(whiskerLayer, whiskerLayer, true);
    }

    private bool LoadAllRightFrames()
    {
        rightFrameData.Clear();
        bool loadedEveryFile = true;

        for (int frameIndex = 0; frameIndex < SourceFrameCount; frameIndex++)
        {
            loadedEveryFile &= LoadRightWhiskingData(
                $"right_whiskers_frame_{frameIndex}",
                frameIndex
            );
        }

        if (rightFrameData.Count != RightWhiskerCount)
        {
            Debug.LogError(
                $"[WhiskerManager] Expected geometry for {RightWhiskerCount} right whiskers, " +
                $"but loaded {rightFrameData.Count}."
            );
            return false;
        }

        for (int whiskerIndex = 0; whiskerIndex < RightWhiskerCount; whiskerIndex++)
        {
            if (!rightFrameData.TryGetValue(whiskerIndex, out List<Vector3[]> frames) ||
                frames.Count != SourceFrameCount)
            {
                Debug.LogError(
                    $"[WhiskerManager] Right whisker index {whiskerIndex} does not have " +
                    $"{SourceFrameCount} source poses."
                );
                return false;
            }
        }

        return loadedEveryFile;
    }

    private bool LoadRightWhiskingData(string fileName, int expectedFrameIndex)
    {
        TextAsset csvData = Resources.Load<TextAsset>($"{whiskingFolder}/{fileName}");
        if (csvData == null)
        {
            Debug.LogError($"[WhiskerManager] Failed to load {fileName}.csv.");
            return false;
        }

        int parsedRows = 0;
        string[] lines = csvData.text.Split('\n');
        foreach (string rawLine in lines)
        {
            string line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("Frame", StringComparison.Ordinal))
                continue;

            string[] values = line.Split(',');
            if (values.Length != 6 ||
                !int.TryParse(values[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int frameIndex) ||
                !int.TryParse(values[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int whiskerIndex) ||
                !int.TryParse(values[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int pointIndex) ||
                !float.TryParse(values[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float x) ||
                !float.TryParse(values[4], NumberStyles.Float, CultureInfo.InvariantCulture, out float y) ||
                !float.TryParse(values[5], NumberStyles.Float, CultureInfo.InvariantCulture, out float z))
            {
                Debug.LogError($"[WhiskerManager] Malformed row in {fileName}.csv: {line}");
                return false;
            }

            if (frameIndex != expectedFrameIndex ||
                whiskerIndex < 0 || whiskerIndex >= RightWhiskerCount ||
                pointIndex < 0 || pointIndex >= PointsPerWhisker)
            {
                Debug.LogError(
                    $"[WhiskerManager] Out-of-contract index in {fileName}.csv: {line}"
                );
                return false;
            }

            if (!rightFrameData.TryGetValue(whiskerIndex, out List<Vector3[]> frames))
            {
                frames = new List<Vector3[]>(SourceFrameCount);
                for (int i = 0; i < SourceFrameCount; i++)
                    frames.Add(new Vector3[PointsPerWhisker]);
                rightFrameData.Add(whiskerIndex, frames);
            }

            // Source CSV coordinates are (x, y, z); Unity uses the existing
            // project convention (x, z, y) in the head-centred frame.
            frames[frameIndex][pointIndex] = new Vector3(x, z, y);
            parsedRows++;
        }

        int expectedRows = RightWhiskerCount * PointsPerWhisker;
        if (parsedRows != expectedRows)
        {
            Debug.LogError(
                $"[WhiskerManager] {fileName}.csv contains {parsedRows} valid rows; " +
                $"expected {expectedRows}."
            );
            return false;
        }

        return true;
    }

    public Vector3[] GetWhiskerPositions(int frameIndex, int whiskerIndex)
    {
        if (rightFrameData.TryGetValue(whiskerIndex, out List<Vector3[]> frames) &&
            frameIndex >= 0 && frameIndex < frames.Count)
        {
            return frames[frameIndex];
        }

        return new Vector3[PointsPerWhisker];
    }

    private List<string> LoadRightWhiskerNames()
    {
        var names = new List<string>(RightWhiskerCount);
        TextAsset csvData = Resources.Load<TextAsset>(paramNameCSV);
        if (csvData == null)
        {
            Debug.LogError($"[WhiskerManager] Failed to load {paramNameCSV}.csv.");
            return names;
        }

        foreach (string rawLine in csvData.text.Split('\n'))
        {
            string name = rawLine.Trim();
            if (name.StartsWith("R", StringComparison.Ordinal))
                names.Add(name);
        }

        return names;
    }

    public bool TryGetWhisker(string whiskerName, out Whisker whisker)
    {
        return whiskers.TryGetValue(whiskerName, out whisker);
    }

    private bool CreateWhisker(int whiskerIndex, List<Vector3[]> frames)
    {
        if (whiskerIndex < 0 || whiskerIndex >= whiskerNames.Count)
        {
            Debug.LogError($"[WhiskerManager] Invalid right-whisker index {whiskerIndex}.");
            return false;
        }

        if (ratHeadObject == null)
        {
            Debug.LogError("[WhiskerManager] ratHeadObject is not assigned.");
            return false;
        }

        string whiskerName = whiskerNames[whiskerIndex];
        var whiskerObject = new GameObject(whiskerName);
        whiskerObject.transform.SetParent(ratHeadObject.transform, false);
        whiskerObject.transform.localPosition = Vector3.zero;
        whiskerObject.transform.localRotation = Quaternion.identity;

        var rigidbody = whiskerObject.AddComponent<Rigidbody>();
        rigidbody.isKinematic = true;
        rigidbody.useGravity = false;
        rigidbody.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
        rigidbody.interpolation = RigidbodyInterpolation.Interpolate;

        Whisker whisker = whiskerObject.AddComponent<Whisker>();
        whisker.Initialize(frames, whiskerMaterial);

        whiskers.Add(whiskerName, whisker);
        whiskerIndices.Add(whiskerName, whiskerIndex);
        return true;
    }

    public void RebuildWhiskers()
    {
        foreach (Whisker whisker in whiskers.Values)
        {
            if (whisker != null)
                Destroy(whisker.gameObject);
        }

        whiskers.Clear();
        whiskerIndices.Clear();

        // Build in anatomical/name order instead of relying on Dictionary
        // enumeration order.
        for (int whiskerIndex = 0; whiskerIndex < RightWhiskerCount; whiskerIndex++)
        {
            if (!rightFrameData.TryGetValue(whiskerIndex, out List<Vector3[]> frames))
            {
                Debug.LogError(
                    $"[WhiskerManager] Missing right-whisker geometry at index {whiskerIndex}."
                );
                continue;
            }

            CreateWhisker(whiskerIndex, frames);
        }
    }

    public void SetAllWhiskersToFrame(int frameIndex)
    {
        frameIndex = Mathf.Clamp(frameIndex, 0, SourceFrameCount - 1);
        foreach (KeyValuePair<string, Whisker> pair in whiskers)
        {
            if (pair.Value != null)
                pair.Value.SetExactFrame(frameIndex, whiskerIndices[pair.Key]);
        }
    }

    public void EvaluateContactsAgainstStimuli()
    {
        int stimulusLayer = LayerMask.NameToLayer("Stimulus");
        if (stimulusLayer < 0)
        {
            Debug.LogError("[WhiskerManager] Stimulus layer is not defined.");
            return;
        }

        int stimulusLayerMask = 1 << stimulusLayer;
        foreach (Whisker whisker in whiskers.Values)
        {
            if (whisker != null)
                whisker.EvaluateContacts(stimulusLayerMask);
        }
    }

    void Update()
    {
        if (!animate || !IsReady)
            return;

        animationTimer += Time.deltaTime * oscillationSpeed;
        float progress = Mathf.PingPong(animationTimer, 1f);
        float framePosition = progress * (SourceFrameCount - 1);
        int frameIndex = Mathf.FloorToInt(framePosition);
        float blend = smoothInterpolation ? framePosition - frameIndex : 0f;

        foreach (KeyValuePair<string, Whisker> pair in whiskers)
        {
            if (pair.Value != null)
                pair.Value.UpdateFrame(frameIndex, blend, whiskerIndices[pair.Key]);
        }
    }
}
