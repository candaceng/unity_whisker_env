using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

[CreateAssetMenu(menuName = "Plans/SupervisedPlan", fileName = "SupervisedPlan")]
public class SupervisedPlan : ScriptableObject, ITrialPlan
{
    public SupervisedProfile profile;

    public IEnumerable<TrialSpec> GenerateTrials()
    {
        if (profile == null)
        {
            Debug.LogError("SupervisedPlan: profile is null");
            yield break;
        }

        Directory.CreateDirectory(profile.outputDir);
        var trials = new List<TrialSpec>();
        var manifest = new List<string>
        {
            "file,label,shape,distance_condition,repeat_group,base_distance,distance_offset,actual_distance,y_offset,yaw_deg,seed"
        };

        for (int rep = 0; rep < profile.repeatsPerCond; rep++)
        {
            for (int distanceIndex = 0; distanceIndex < profile.distances.Length; distanceIndex++)
            {
                float baseDistance = profile.distances[distanceIndex];
                int trialSeed = StableSeed(profile.randomSeed, rep, distanceIndex);
                var random = new System.Random(trialSeed);

                float distanceOffset = SymmetricJitter(random, profile.distanceJitter);
                float yOffset = SymmetricJitter(random, profile.verticalJitter);
                float yawDeg = SymmetricJitter(random, profile.yawJitterDeg);
                float actualDistance = baseDistance + distanceOffset;

                string distTag = DistanceTag(distanceIndex, baseDistance);

                // Concave and convex receive the identical nuisance transform.
                // This prevents position, height, or yaw from leaking the class label.
                foreach (string shape in profile.labels)
                {
                    string label = $"{shape}_{distTag}";
                    string outFile = Path.Combine(profile.outputDir, $"{label}_rep{rep:D3}.csv");

                    trials.Add(new TrialSpec
                    {
                        label = label,
                        repeatIndex = rep,
                        outFile = outFile,

                        H_mm = 0f,
                        y_mm = yOffset,
                        startZ_mm = actualDistance,
                        endZ_mm = actualDistance,
                        yawDeg = yawDeg,

                        synset = "",
                        modelId = ""
                    });

                    manifest.Add(string.Join(",",
                        Path.GetFileName(outFile),
                        label,
                        shape,
                        distTag,
                        rep.ToString(CultureInfo.InvariantCulture),
                        F(baseDistance),
                        F(distanceOffset),
                        F(actualDistance),
                        F(yOffset),
                        F(yawDeg),
                        trialSeed.ToString(CultureInfo.InvariantCulture)
                    ));
                }
            }
        }

        File.WriteAllLines(Path.Combine(profile.outputDir, "trial_manifest.csv"), manifest);

        foreach (var trial in trials)
            yield return trial;
    }

    private static int StableSeed(int baseSeed, int repeatIndex, int distanceIndex)
    {
        unchecked
        {
            int seed = baseSeed;
            seed = seed * 397 ^ repeatIndex;
            seed = seed * 397 ^ distanceIndex;
            return seed & 0x7fffffff;
        }
    }

    private static float SymmetricJitter(System.Random random, float amplitude)
    {
        if (amplitude <= 0f) return 0f;
        return (float)((2.0 * random.NextDouble() - 1.0) * amplitude);
    }

    private static string DistanceTag(int index, float distance)
    {
        if (index == 0) return "near";
        if (index == 1) return "med";
        if (index == 2) return "far";
        return $"d{distance.ToString("0.###", CultureInfo.InvariantCulture)}";
    }

    private static string F(float value) => value.ToString("0.######", CultureInfo.InvariantCulture);
}
