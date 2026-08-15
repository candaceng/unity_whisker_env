using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public static class RuntimeObjLoader
{
    struct Idx { public int v, vt, vn; } // 1-based in OBJ; 0=missing
    static readonly char[] splitSpace = new[]{' '};
    static readonly char[] splitSlash = new[]{'/'};

    public static Mesh LoadMesh(string objPath, bool recalcNormalsIfMissing = true, float uniformScale = 1f)
    {
        using var sr = new StreamReader(objPath);
        var verts = new List<Vector3>();
        var norms = new List<Vector3>();
        var uvs   = new List<Vector2>();
        var faces = new List<Idx[]>(); // each face is 3+ corners

        var ci = CultureInfo.InvariantCulture;

        string line;
        while ((line = sr.ReadLine()) != null)
        {
            if (line.Length == 0 || line[0] == '#') continue;
            var t = line.TrimStart();

            if (t.StartsWith("v "))
            {
                var p = t.Split(splitSpace, StringSplitOptions.RemoveEmptyEntries);
                verts.Add(new Vector3(
                    float.Parse(p[1], ci) * uniformScale,
                    float.Parse(p[2], ci) * uniformScale,
                    float.Parse(p[3], ci) * uniformScale
                ));
            }
            else if (t.StartsWith("vn "))
            {
                var p = t.Split(splitSpace, StringSplitOptions.RemoveEmptyEntries);
                norms.Add(new Vector3(
                    float.Parse(p[1], ci),
                    float.Parse(p[2], ci),
                    float.Parse(p[3], ci)
                ).normalized);
            }
            else if (t.StartsWith("vt "))
            {
                var p = t.Split(splitSpace, StringSplitOptions.RemoveEmptyEntries);
                uvs.Add(new Vector2(
                    float.Parse(p[1], ci),
                    float.Parse(p[2], ci)
                ));
            }
            else if (t.StartsWith("f "))
            {
                var p = t.Split(splitSpace, StringSplitOptions.RemoveEmptyEntries);
                var poly = new List<Idx>(p.Length - 1);
                for (int i = 1; i < p.Length; i++)
                {
                    var parts = p[i].Split(splitSlash); // v / vt / vn
                    var idx = new Idx{
                        v  = ParseIndex(parts, 0),
                        vt = ParseIndex(parts, 1),
                        vn = ParseIndex(parts, 2)
                    };
                    poly.Add(idx);
                }
                // triangulate fan
                for (int i = 1; i + 1 < poly.Count; i++)
                    faces.Add(new[] { poly[0], poly[i], poly[i+1] });
            }
        }

        // Build unified arrays (dedup v/vt/vn combos)
        var map = new Dictionary<(int,int,int), int>();
        var outV = new List<Vector3>();
        var outN = new List<Vector3>();
        var outT = new List<Vector2>();
        var tris = new List<int>();

        foreach (var tri in faces)
        {
            for (int k = 0; k < 3; k++)
            {
                var key = (tri[k].v, tri[k].vt, tri[k].vn);
                if (!map.TryGetValue(key, out int newIdx))
                {
                    newIdx = outV.Count;
                    map[key] = newIdx;
                    outV.Add(verts[tri[k].v - 1]);
                    outT.Add(tri[k].vt > 0 ? uvs[tri[k].vt - 1] : Vector2.zero);
                    outN.Add(tri[k].vn > 0 ? norms[tri[k].vn - 1] : Vector3.zero);
                }
                tris.Add(newIdx);
            }
        }

        var mesh = new Mesh{ name = Path.GetFileNameWithoutExtension(objPath) };
        mesh.SetVertices(outV);
        mesh.SetTriangles(tris, 0);
        if (outN.TrueForAll(n => n == Vector3.zero))
        {
            if (recalcNormalsIfMissing) mesh.RecalculateNormals();
        }
        else mesh.SetNormals(outN);
        if (outT.Exists(_ => true)) mesh.SetUVs(0, outT);

        mesh.RecalculateBounds();
        return mesh;
    }

    static int ParseIndex(string[] parts, int i)
    {
        if (i >= parts.Length || string.IsNullOrEmpty(parts[i])) return 0;
        return int.Parse(parts[i], CultureInfo.InvariantCulture);
    }
}
