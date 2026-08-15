using UnityEngine;

[ExecuteAlways]
public class ArcStimulusSpawner : MonoBehaviour
{
    public enum Curvature { Convex, Concave }

    [Header("Stimulus physics")]
    public string stimulusTag = "Stimulus";
    public string stimulusLayerName = "Stimulus";
    public bool colliderIsTrigger = false;

    [Tooltip("Number of static primitive boxes used to approximate the curved collision surface.")]
    [Range(4, 64)] public int compoundColliderSegments = 16;

    [Tooltip("Small tangent overlap between neighboring boxes to avoid collision seams.")]
    [Min(0f)] public float compoundColliderOverlap = 0.05f;

    [Header("Geometry (Unity units)")]
    [Tooltip("Circular fit to Chung et al.'s convex-shape.obj contact surface.")]
    [Min(0.001f)] public float radius = 63.04664f;
    [Tooltip("Chord length of Chung et al.'s contact surface.")]
    [Min(0.001f)] public float arcWidth = 50.84435f;
    [Tooltip("Vertical extent of Chung et al.'s stimulus after OBJ_SCALE=55.")]
    [Min(0.001f)] public float extrudeLength = 17.43234f;
    [Tooltip("Thickness of Chung et al.'s curved stimulus.")]
    [Min(0.0001f)] public float thickness = 1.72807f;
    public float x_offset = 0f;

    [Header("Resolution")]
    [Range(4, 256)] public int arcSegments = 64;
    [Range(1, 128)] public int zSegments = 8;


    [Header("Chung/Rodgers six-condition placement")]
    [Tooltip("Use the head-centred Unity conversion of Chung et al.'s rodgers_stimuli.sh setup.")]
    public bool useChungRodgersPlacement = true;

    // These values assume convex-shape.obj has no intrinsic rotation.
    public Vector3 chungSurfaceTangent = new Vector3(0.87543f, 0.23457f, -0.42262f);
    public Vector3 chungSurfaceNormalTowardHead = new Vector3(-0.40822f, -0.10938f, -0.90631f);
    public Vector3 chungVerticalAxis = new Vector3(-0.25882f, 0.96593f, 0f);
    public Vector3 chungConcaveNearPosition = new Vector3(22.32463f, 8.99355f, 8.44034f);
    public Vector3 chungConvexNearPosition = new Vector3(17.03400f, 7.57593f, 12.87371f);

    [Tooltip("Near-to-medium (and medium-to-far) translation used by Chung et al.")]
    public Vector3 chungDistanceStep = new Vector3(0.96593f, 0.25882f, 1f);

    [Header("Legacy/custom placement")]
    public Vector3 basePosition = Vector3.zero;
    public Vector3 approachAxis = Vector3.right;

    [Header("Legacy Eval6cond distance")]
    public float distNear = 2f;

    [Header("Rendering / Collision")]
    public Material material;

    [Header("Regenerate")]
    public bool autoRegenerateInEditor = false;

    public bool showAllInEditor = true;
    void OnEnable()
    {
        RegenerateAll();

        if (Application.isPlaying)
        {
            SetAllActive(false);
        }
        else
        {
            SetAllActive(showAllInEditor);
        }
    }

    public void SetAllActive(bool on)
    {
        var root = transform.Find("__GeneratedStimuli__");
        if (!root) return;
        for (int i = 0; i < root.childCount; i++)
            root.GetChild(i).gameObject.SetActive(on);
    }

    GameObject concaveNear, concaveMed, concaveFar;
    GameObject convexNear,  convexMed,  convexFar;

    [SerializeField] private Transform generatedRoot;

    private Transform GetOrCreateGeneratedRoot()
    {
        if (generatedRoot != null) return generatedRoot;

        var t = transform.Find("__GeneratedStimuli__");
        if (t == null)
        {
            var go = new GameObject("__GeneratedStimuli__");
            go.transform.SetParent(transform, false);
            t = go.transform;
        }
        generatedRoot = t;
        return generatedRoot;
    }

    private void ClearGenerated()
    {
    #if UNITY_EDITOR
        var root = GetOrCreateGeneratedRoot();
        for (int i = root.childCount - 1; i >= 0; i--)
            DestroyImmediate(root.GetChild(i).gameObject);
    #else
        var root = GetOrCreateGeneratedRoot();
        for (int i = root.childCount - 1; i >= 0; i--)
            Destroy(root.GetChild(i).gameObject);
    #endif
    }

    void OnValidate()
    {
        if (!autoRegenerateInEditor) return;
        RegenerateAll();
    }

    public void RegenerateAll()
    {
        ClearGenerated();

        DestroyChild(ref concaveNear); DestroyChild(ref concaveMed); DestroyChild(ref concaveFar);
        DestroyChild(ref convexNear);  DestroyChild(ref convexMed);  DestroyChild(ref convexFar);

        if (useChungRodgersPlacement)
        {
            // Chung et al. use the same mesh for both shapes, rotate it by
            // 180 degrees, and translate each successive distance by (+1,+1)
            // in WHISKiT's horizontal plane. These poses express that setup in
            // the same whisker-pad coordinate frame as the imported CSV poses.
            concaveNear = MakeStimulusAtPose(
                "concave_near", Curvature.Concave,
                chungConcaveNearPosition, ChungRodgersRotation());
            concaveMed = MakeStimulusAtPose(
                "concave_med", Curvature.Concave,
                chungConcaveNearPosition + chungDistanceStep, ChungRodgersRotation());
            concaveFar = MakeStimulusAtPose(
                "concave_far", Curvature.Concave,
                chungConcaveNearPosition + 2f * chungDistanceStep, ChungRodgersRotation());

            convexNear = MakeStimulusAtPose(
                "convex_near", Curvature.Convex,
                chungConvexNearPosition, ChungRodgersRotation());
            convexMed = MakeStimulusAtPose(
                "convex_med", Curvature.Convex,
                chungConvexNearPosition + chungDistanceStep, ChungRodgersRotation());
            convexFar = MakeStimulusAtPose(
                "convex_far", Curvature.Convex,
                chungConvexNearPosition + 2f * chungDistanceStep, ChungRodgersRotation());
        }
        else
        {
            concaveNear = MakeStimulus("concave_near", Curvature.Concave, distNear);
            concaveMed  = MakeStimulus("concave_med",  Curvature.Concave, distNear + 2.0f);
            concaveFar  = MakeStimulus("concave_far",  Curvature.Concave, distNear + 4.0f);

            convexNear = MakeStimulus("convex_near", Curvature.Convex, distNear);
            convexMed  = MakeStimulus("convex_med",  Curvature.Convex, distNear + 2.0f);
            convexFar  = MakeStimulus("convex_far",  Curvature.Convex, distNear + 4.0f);
        }
    }

    [ContextMenu("Apply Chung/Rodgers Defaults")]
    public void ApplyChungRodgersDefaults()
    {
        useChungRodgersPlacement = true;

        radius = 63.04664f;
        arcWidth = 50.84435f;
        extrudeLength = 17.43234f;
        thickness = 1.72807f;

        chungDistanceStep = new Vector3(0.96593f, 0.25882f, 1f);

        // These values assume convex-shape.obj has no intrinsic rotation.
        chungSurfaceTangent = new Vector3(0.87543f, 0.23457f, -0.42262f);
        chungSurfaceNormalTowardHead = new Vector3(-0.40822f, -0.10938f, -0.90631f);
        chungVerticalAxis = new Vector3(-0.25882f, 0.96593f, 0f);

        chungConcaveNearPosition = new Vector3(22.32463f, 8.99355f, 8.44034f);
        chungConvexNearPosition = new Vector3(17.03400f, 7.57593f, 12.87371f);

        RegenerateAll();
    }

    void DestroyChild(ref GameObject go)
    {
        if (!go) return;
        #if UNITY_EDITOR
                if (!Application.isPlaying) DestroyImmediate(go);
                else Destroy(go);
        #else
                Destroy(go);
        #endif
                go = null;
    }

    GameObject MakeStimulus(string name, Curvature curvature, float distance)
    {
        var go = MakeStimulusAtPose(
            name,
            curvature,
            new Vector3(x_offset, 0f, distance),
            Quaternion.Euler(-90f, 0f, 0f)
        );
        return go;
    }

    GameObject MakeStimulusAtPose(
        string name,
        Curvature curvature,
        Vector3 localPosition,
        Quaternion localRotation)
    {
        var go = new GameObject(name);
        go.transform.SetParent(GetOrCreateGeneratedRoot(), false);
        go.transform.localPosition = localPosition;
        go.transform.localRotation = localRotation;
        go.transform.localScale = Vector3.one;

        go.tag = stimulusTag;
        int stimLayer = LayerMask.NameToLayer(stimulusLayerName);
        go.layer = stimLayer;

        var mf = go.AddComponent<MeshFilter>();
        var mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = material;

        Mesh mesh = BuildArcExtrusionMesh(curvature);
        mf.sharedMesh = mesh;

        // Keep the procedural mesh for rendering, but use a compound chain of
        // primitive boxes for physics. This avoids a moving/concave MeshCollider
        // while keeping the number of moving whisker colliders unchanged.
        BuildCompoundColliders(go, curvature, stimLayer);

        return go;
    }

    Quaternion ChungRodgersRotation()
    {
        Vector3 forward = chungVerticalAxis.normalized;
        Vector3 up = chungSurfaceNormalTowardHead.normalized;
        Quaternion rotation = Quaternion.LookRotation(forward, up);

        // Keep the Inspector tangent as a readable conversion check. The
        // rotation itself is defined by the surface normal and vertical axis.
        if (Vector3.Dot(rotation * Vector3.right, chungSurfaceTangent.normalized) < 0.999f)
            Debug.LogWarning("[ArcStimulusSpawner] Chung wall axes are not mutually consistent.");

        return rotation;
    }

    private void BuildCompoundColliders(GameObject stimulus, Curvature curvature, int stimulusLayer)
    {
        int count = Mathf.Clamp(compoundColliderSegments, 4, 64);

        var root = new GameObject("__CompoundColliders__");
        root.transform.SetParent(stimulus.transform, false);
        root.tag = stimulusTag;
        root.layer = stimulusLayer;

        float halfW = arcWidth * 0.5f;
        float maxChord = 2f * radius * 0.999f;
        float chord = Mathf.Min(arcWidth, maxChord);
        float thetaHalf = Mathf.Asin((chord * 0.5f) / radius);
        float xEndpoint = radius * Mathf.Sin(thetaHalf);
        float xScale = (xEndpoint > 1e-6f) ? (halfW / xEndpoint) : 1f;
        float sign = (curvature == Curvature.Convex) ? 1f : -1f;

        Vector2 PointOnArc(float angle)
        {
            float x = radius * Mathf.Sin(angle) * xScale;
            float y = (radius * Mathf.Cos(angle) - radius) * sign;
            return new Vector2(x, y);
        }

        for (int i = 0; i < count; i++)
        {
            float t0 = (float)i / count;
            float t1 = (float)(i + 1) / count;
            float a0 = Mathf.Lerp(-thetaHalf, thetaHalf, t0);
            float a1 = Mathf.Lerp(-thetaHalf, thetaHalf, t1);

            Vector2 p0 = PointOnArc(a0);
            Vector2 p1 = PointOnArc(a1);
            Vector2 tangent = p1 - p0;
            float tangentLength = Mathf.Max(0.001f, tangent.magnitude);
            tangent /= tangentLength;

            // Local +Y of each box points toward the head at the center of the
            // surface. Offset the box behind the rendered face so its front face
            // coincides with the requested arc rather than shifting the distance.
            Vector2 normalTowardHead = new Vector2(-tangent.y, tangent.x);
            Vector2 surfaceMidpoint = 0.5f * (p0 + p1);
            Vector2 boxCenter = surfaceMidpoint - normalTowardHead * (thickness * 0.5f);

            var segmentObject = new GameObject($"SurfaceBox_{i:D2}");
            segmentObject.transform.SetParent(root.transform, false);
            segmentObject.transform.localPosition = new Vector3(boxCenter.x, boxCenter.y, 0f);
            segmentObject.transform.localRotation = Quaternion.Euler(
                0f,
                0f,
                Mathf.Atan2(tangent.y, tangent.x) * Mathf.Rad2Deg
            );
            segmentObject.tag = stimulusTag;
            segmentObject.layer = stimulusLayer;

            var box = segmentObject.AddComponent<BoxCollider>();
            box.isTrigger = colliderIsTrigger;
            box.providesContacts = true;
            box.size = new Vector3(
                tangentLength + compoundColliderOverlap,
                Mathf.Max(0.001f, thickness),
                extrudeLength + 2f * compoundColliderOverlap
            );
        }
    }

    public GameObject SpawnTrialStimulus(string name, Curvature curvature, float distance, float yOffset = 0f, float yawDeg = 0f)
    {
        var go = MakeStimulus(name, curvature, distance);

        // local placement relative to spawner
        go.transform.localPosition = new Vector3(x_offset, yOffset, distance);
        go.transform.localRotation =
            Quaternion.AngleAxis(yawDeg, Vector3.up)
            * Quaternion.Euler(-90f, 0f, 0f);

        return go;
    }

    public void DestroyStimulus(GameObject go)
    {
        if (!go) return;
    #if UNITY_EDITOR
        if (!Application.isPlaying) DestroyImmediate(go);
        else Destroy(go);
    #else
        Destroy(go);
    #endif
    }

    Mesh BuildArcExtrusionMesh(Curvature curvature)
    {
        float halfW = arcWidth * 0.5f;
        float halfL = extrudeLength * 0.5f;

        float maxChord = 2f * radius * 0.999f;
        float chord = Mathf.Min(arcWidth, maxChord);

        float theta = 2f * Mathf.Asin((chord * 0.5f) / radius); // chord = 2R sin(theta/2) => theta = 2 asin(chord/(2R))
        float thetaHalf = theta * 0.5f;

        // parameterize points by angle a in [-thetaHalf, +thetaHalf]
        // and map to x = R sin(a), y = R cos(a) - R (so y=0 at center for a=0)
        // then flip sign on y for concave vs convex.
        int nx = arcSegments + 1;
        int nz = zSegments + 1;

        int vertsPerLayer = nx * nz;
        int totalVerts = vertsPerLayer * 2; // front + back
        var verts = new Vector3[totalVerts];
        var normals = new Vector3[totalVerts];
        var uvs = new Vector2[totalVerts];

        float sign = (curvature == Curvature.Convex) ? 1f : -1f;

        // Build front surface (layer 0) and back surface (layer 1)
        for (int iz = 0; iz < nz; iz++)
        {
            float tz = (float)iz / (nz - 1);
            float z = Mathf.Lerp(-halfL, halfL, tz);

            for (int ix = 0; ix < nx; ix++)
            {
                float tx = (float)ix / (nx - 1);
                float a = Mathf.Lerp(-thetaHalf, thetaHalf, tx);

                float x = radius * Mathf.Sin(a);
                float y = (radius * Mathf.Cos(a) - radius); // <= 0
                y *= sign;

                // Center x to match requested arcWidth (since chord may be clamped)
                // Also scale x so endpoints match chord/2
                float xEndpoint = radius * Mathf.Sin(thetaHalf);
                float xScale = (xEndpoint > 1e-6f) ? (halfW / xEndpoint) : 1f;
                x *= xScale;

                int i0 = iz * nx + ix;              // front
                int i1 = vertsPerLayer + i0;        // back

                Vector3 p = new Vector3(x, y, z);
                verts[i0] = p;
                // The compound colliders place the solid behind the contact
                // surface along local -Y. Match that convention for rendering;
                // local Z is the vertical extrusion axis after placement.
                verts[i1] = p + new Vector3(0, -thickness, 0);

                // Approx normals: derivative-based normal on the arc surface; good enough
                // For this parameterization, the normal points approximately out of the surface.
                Vector3 n = new Vector3(0, sign, 0);
                normals[i0] = n;
                normals[i1] = -n; // opposite for back

                uvs[i0] = new Vector2(tx, tz);
                uvs[i1] = new Vector2(tx, tz);
            }
        }

        // Triangles: front + back + side walls (4 sides: left/right in X, and two Z caps)
        // Front
        var tris = new System.Collections.Generic.List<int>();

        void AddQuad(int a, int b, int c, int d, bool flip)
        {
            // (a,b,c,d) in CCW order for one side
            if (!flip)
            {
                tris.Add(a); tris.Add(b); tris.Add(c);
                tris.Add(a); tris.Add(c); tris.Add(d);
            }
            else
            {
                tris.Add(a); tris.Add(c); tris.Add(b);
                tris.Add(a); tris.Add(d); tris.Add(c);
            }
        }

        // helper to index front/back
        int Front(int ix, int iz) => iz * nx + ix;
        int Back(int ix, int iz) => vertsPerLayer + iz * nx + ix;

        // Front surface
        for (int iz = 0; iz < nz - 1; iz++)
        for (int ix = 0; ix < nx - 1; ix++)
        {
            int a = Front(ix, iz);
            int b = Front(ix + 1, iz);
            int c = Front(ix + 1, iz + 1);
            int d = Front(ix, iz + 1);
            AddQuad(a, b, c, d, flip:false);
        }

        // Back surface (flip winding)
        for (int iz = 0; iz < nz - 1; iz++)
        for (int ix = 0; ix < nx - 1; ix++)
        {
            int a = Back(ix, iz);
            int b = Back(ix + 1, iz);
            int c = Back(ix + 1, iz + 1);
            int d = Back(ix, iz + 1);
            AddQuad(a, b, c, d, flip:true);
        }

        // Side walls along X edges (ix=0 and ix=nx-1)
        for (int iz = 0; iz < nz - 1; iz++)
        {
            // left edge
            AddQuad(Front(0, iz), Front(0, iz + 1), Back(0, iz + 1), Back(0, iz), flip:false);
            // right edge
            AddQuad(Front(nx - 1, iz), Back(nx - 1, iz), Back(nx - 1, iz + 1), Front(nx - 1, iz + 1), flip:false);
        }

        // Caps at z ends (iz=0 and iz=nz-1)
        for (int ix = 0; ix < nx - 1; ix++)
        {
            // near cap (iz=0)
            AddQuad(Front(ix, 0), Back(ix, 0), Back(ix + 1, 0), Front(ix + 1, 0), flip:false);
            // far cap (iz=nz-1)
            AddQuad(Front(ix, nz - 1), Front(ix + 1, nz - 1), Back(ix + 1, nz - 1), Back(ix, nz - 1), flip:false);
        }

        var mesh = new Mesh();
        mesh.name = $"ArcExtrusion_{curvature}";
        mesh.indexFormat = (totalVerts > 65000) ? UnityEngine.Rendering.IndexFormat.UInt32 : UnityEngine.Rendering.IndexFormat.UInt16;

        mesh.vertices = verts;
        mesh.normals = normals;
        mesh.uv = uvs;
        mesh.triangles = tris.ToArray();
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();

        return mesh;
    }
}
