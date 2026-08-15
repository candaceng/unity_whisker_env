using UnityEngine;

[CreateAssetMenu(menuName = "Plans/SupervisedProfile", fileName = "SupervisedProfile")]
public class SupervisedProfile : ScriptableObject
{
    [Header("Output")]
    public string outputDir = "Data/supervised_whisker_logs";

    [Header("Balanced conditions")]
    public int repeatsPerCond = 50;
    public string[] labels = new string[] { "concave", "convex" };
    public float[] distances = new float[] { 3f, 5f, 8f };

    [Header("Deterministic nuisance variation")]
    [Tooltip("The same sampled transform is used for concave and convex trials in each repeat/distance group.")]
    public int randomSeed = 20260720;
    [Min(0f)] public float distanceJitter = 0.25f;
    [Min(0f)] public float verticalJitter = 0.25f;
    [Min(0f)] public float yawJitterDeg = 2f;

    [Header("Exact pose recording")]
    public int maxFrames = 51;
    public float preHold = 0.1f;
    public float postHold = 0.1f;
}
