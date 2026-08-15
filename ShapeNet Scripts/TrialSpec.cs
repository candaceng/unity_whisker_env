using UnityEngine;

[System.Serializable]
public struct TrialSpec
{
    public string label, synset, modelId;

    // Support BOTH code paths:
    public Mesh mesh;             // used by old/eval-mesh paths (OK if null)
    public GameObject prefab;     // used by new eval-prefab path (OK if null)

    public Vector3 position;
    public Quaternion rotation;
    public float baseAngleOffsetDeg;
    public int repeatIndex;
    public string outFile;

    // sweep params
    public float H_mm, y_mm, startZ_mm, endZ_mm, yawDeg;
}
