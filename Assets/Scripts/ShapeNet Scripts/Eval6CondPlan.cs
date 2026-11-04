using System.Collections.Generic;
using System.IO;
using UnityEngine;

[CreateAssetMenu(menuName = "Plans/Eval6CondPlan", fileName = "Eval6CondPlan")]
public class Eval6CondPlan : ScriptableObject, ITrialPlan
{
    public Eval6CondProfile profile;

    public IEnumerable<TrialSpec> GenerateTrials()
    {
        if (profile == null)
        {
            Debug.LogError("Eval6CondPlan: profile is null");
            yield break;
        }

        Directory.CreateDirectory(profile.outputDir);

        for (int rep = 0; rep < profile.repeatsPerCond; rep++)
        {
            foreach (var label in profile.labels)
            {
                string outFile = Path.Combine(profile.outputDir, $"{label}_rep{rep:D2}.csv");

                yield return new TrialSpec
                {
                    label = label,
                    repeatIndex = rep,
                    outFile = outFile,

                    // Not used for eval6cond with ArcStimulusSpawner:
                    prefab = null,
                    mesh = null,
                    position = Vector3.zero,
                    rotation = Quaternion.identity,
                    baseAngleOffsetDeg = 0f,

                    H_mm = 0f, y_mm = 0f, startZ_mm = 0f, endZ_mm = 0f, yawDeg = 0f,
                    synset = "", modelId = ""
                };
            }
        }
    }
}
