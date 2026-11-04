using UnityEngine;

[System.Serializable]
public struct TrialSpec
{
    public string label, synset, modelId;

    // Mesh and prefab trial representations.
    public Mesh mesh;             // Optional mesh for mesh-based trials.
    public GameObject prefab;     // Optional prefab for prefab-based trials.

    public Vector3 position;
    public Quaternion rotation;
    public float baseAngleOffsetDeg;
    public int repeatIndex;
    public string outFile;

    // Sweep parameters.
    public float H_mm, y_mm, startZ_mm, endZ_mm, yawDeg;
}
