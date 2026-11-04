using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class TrainAugPlan : ITrialPlan
{
    private readonly string[] objPaths;
    private readonly string outDir;
    private readonly float[] H_mm;
    private readonly float[] height_mm;
    private readonly float[] yaw_deg;
    private readonly float sweep_len_mm;
    private readonly float sceneUnitsPerMm;
    private readonly Vector3 headOffsetUnits;

    public TrainAugPlan(
        string[] objPaths, string outDir,
        float[] H_mm, float[] height_mm, float[] yaw_deg,
        float sweep_len_mm, float sceneUnitsPerMm,
        Vector3 headOffsetUnits)
    {
        this.objPaths = objPaths;
        this.outDir = outDir;
        this.H_mm = H_mm;
        this.height_mm = height_mm;
        this.yaw_deg = yaw_deg;
        this.sweep_len_mm = sweep_len_mm;
        this.sceneUnitsPerMm = sceneUnitsPerMm;
        this.headOffsetUnits = headOffsetUnits;
        Directory.CreateDirectory(outDir);
    }

    public IEnumerable<TrialSpec> GenerateTrials()
    {
        foreach (var objPath in objPaths)
        {
            var modelsDir = Path.GetDirectoryName(objPath);
            var modelId = Path.GetFileName(Path.GetDirectoryName(modelsDir));
            var synset  = Path.GetFileName(Path.GetDirectoryName(Path.GetDirectoryName(modelsDir)));

            foreach (var H in H_mm)
            foreach (var y in height_mm)
            foreach (var yaw in yaw_deg)
            {
                float half = sweep_len_mm * 0.5f;
                float startZ_mm = H + half;
                float endZ_mm   = H - half;

                string fname   = $"{synset}_{modelId}_H{H}_y{y}_yaw{yaw}.csv";
                string outFile = Path.Combine(outDir, fname);

                yield return new TrialSpec{
                    label = $"{synset}_{modelId}",
                    synset = synset,
                    modelId = modelId,
                    mesh = null, // loaded by runner from objPath
                    position = Vector3.zero,
                    rotation = Quaternion.Euler(0f, yaw, 0f),
                    baseAngleOffsetDeg = 0f,
                    repeatIndex = 0,
                    outFile = outFile,
                    H_mm = H,
                    y_mm = y,
                    startZ_mm = startZ_mm,
                    endZ_mm = endZ_mm,
                    yawDeg = yaw
                };
            }
        }
    }

}
