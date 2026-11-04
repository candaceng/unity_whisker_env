using UnityEngine;

[CreateAssetMenu(menuName = "Plans/Eval6CondProfile", fileName = "Eval6CondProfile")]
public class Eval6CondProfile : ScriptableObject
{
    [Header("Output")]
    public string outputDir = "Data/eval6cond_right_v3";

    [Header("Trials")]
    [Min(1)] public int repeatsPerCond = 1;

    [Header("Timing")]
    public float preHold = 0.2f;
    public float postHold = 0.2f;

    [Header("Logging")]
    [Min(1)] public int maxFrames = 51;

    [Header("Condition labels (must match ArcStimulusSpawner child names)")]
    public string[] labels = new string[]
    {
        "concave_near", "concave_med", "concave_far",
        "convex_near",  "convex_med",  "convex_far"
    };
}
