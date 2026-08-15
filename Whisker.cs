using UnityEngine;
using System.Collections.Generic;
using System;

public class Whisker : MonoBehaviour {
    [Header("References")]
    [SerializeField] private Transform headOrigin;
    [SerializeField] private LineRenderer lineRenderer;

    private Vector3[] currentShapeHead;
    private Vector3[] currentShapeWorld;
    private int currentFrameIndex;

    private MeshCollider meshCollider;
    private List<Vector3[]> framePoints = new List<Vector3[]>();
    private int currentFrame;
    private float blendFactor;

    private bool hasContact = false;
    private Vector3 lastContactPoint = Vector3.zero;
    private Vector3 lastContactNormal = Vector3.zero;
    private string lastContactObjectName = "";
    private Mesh bakedMesh;

    [Header("Collision segments along whisker")]
    public int collisionSegments = 5;
    public float colliderRadius = 0.5f;
    private WhiskerCollisionSegment[] collisionSegs;

    public float thetaWDeg { get; private set; } = 0f;
    public float SContact { get; private set; }

    [Tooltip("Right-eye camera used to render this whisker.")]
    public Camera eyeCamera;

    [Range(0.5f, 4f)] public float minPixelsBase = 4.0f;
    [Range(0.5f, 4f)] public float minPixelsTip = 4.0f;
    [Range(0.1f, 2f)] public float pixelThicknessGain = 2.0f;
    public float maxWorldWidth = 0.1f;
    public int targetPixelHeight = 64;

    float PixelsToWorldWidth(float pixels, float distance) {
        float vfovRad = (eyeCamera ? eyeCamera.fieldOfView : 140f) * Mathf.Deg2Rad;
        float worldPerPixel = 2f * distance * Mathf.Tan(vfovRad * 0.5f) / Mathf.Max(1, targetPixelHeight);
        return pixels * worldPerPixel;
    }

    public void Initialize(List<Vector3[]> allFrames, Material material) {
        if (headOrigin == null)
            headOrigin = GameObject.Find("HeadOrigin")?.transform;

        framePoints = allFrames;

        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.useWorldSpace = true;

        lineRenderer.alignment = LineAlignment.View;
        lineRenderer.numCornerVertices = 6;
        lineRenderer.numCapVertices    = 6;
        lineRenderer.textureMode = LineTextureMode.Stretch;

        if (material == null) {
            material = new Material(Shader.Find("Unlit/Color"));
            material.color = Color.black;
        }
        lineRenderer.material = material;

        UpdateVisuals(0, 0, 0);
        CreateCollisionSegments();
    }

    public void UpdateFrame(int frameIndex, float blend, int whiskerIndex) {
        float framePosition = Mathf.Clamp(
            frameIndex + Mathf.Clamp01(blend),
            0f,
            framePoints.Count - 1f
        );
        int baseFrame = Mathf.Min(
            Mathf.FloorToInt(framePosition),
            framePoints.Count - 2
        );
        float localBlend = framePosition - baseFrame;

        UpdateVisuals(baseFrame, localBlend, whiskerIndex);
        thetaWDeg = framePosition - 10f;
    }

    public void SetExactFrame(int frameIndex, int whiskerIndex)
    {
        frameIndex = Mathf.Clamp(frameIndex, 0, framePoints.Count - 1);
        UpdateVisuals(frameIndex, 0f, whiskerIndex);
        thetaWDeg = frameIndex - 10f;
    }

    void UpdateVisuals(int frameIndex, float blend, int whiskerIndex)
    {
        frameIndex = Mathf.Clamp(frameIndex, 0, framePoints.Count - 1);
        int nextIndex = Mathf.Min(frameIndex + 1, framePoints.Count - 1);
        if (frameIndex == nextIndex) blend = 0f;

        Vector3[] currentFramePositions = framePoints[frameIndex];
        Vector3[] nextFramePositions    = framePoints[nextIndex];

        // 1) Interpolate in HEAD frame (CSV frame)
        int N = 100;
        Vector3[] headPositions = new Vector3[N];
        for (int i = 0; i < N; i++)
            headPositions[i] = Vector3.Lerp(currentFramePositions[i], nextFramePositions[i], blend);

        currentShapeHead = headPositions;
        currentFrameIndex = frameIndex;

        // 2) Convert HEAD -> WORLD for rendering + colliders
        Vector3[] worldPositions = new Vector3[N];
        for (int i = 0; i < N; i++)
            worldPositions[i] = headOrigin.TransformPoint(headPositions[i]);

        currentShapeWorld = worldPositions;

        // 3) Render in WORLD
        if (lineRenderer != null)
        {
            lineRenderer.useWorldSpace = true;
            lineRenderer.positionCount = worldPositions.Length;
            lineRenderer.SetPositions(worldPositions);

            Vector3 p0World = worldPositions[0];
            Vector3 pNWorld = worldPositions[worldPositions.Length - 1];

            float d0 = (eyeCamera ? Vector3.Distance(eyeCamera.transform.position, p0World) : 0.2f);
            float dN = (eyeCamera ? Vector3.Distance(eyeCamera.transform.position, pNWorld) : 0.2f);

            float scale = 1f; // world space: don't divide by lossyScale here

            float wBase = PixelsToWorldWidth(minPixelsBase * pixelThicknessGain, d0) / scale;
            float wTip  = PixelsToWorldWidth(minPixelsTip  * pixelThicknessGain, dN) / scale;

            wBase = Mathf.Min(wBase, maxWorldWidth);
            wTip  = Mathf.Min(wTip,  maxWorldWidth);

            var wc = new AnimationCurve();
            wc.AddKey(0f, wBase);
            wc.AddKey(1f, Mathf.Max(0.0005f, wTip));
            lineRenderer.widthCurve = wc;
        }

        // 4) Update collision capsules in WORLD
        UpdateCollisionSegments(currentShapeWorld);
    }

    public float GetSContact() { return SContact; }

    private float ComputeSFromContact(Vector3 contactHead, Vector3[] ptsHead, out float bestDist)
    {
        bestDist = float.PositiveInfinity;
        if (ptsHead == null || ptsHead.Length < 2) return -1f;

        // Precompute segment lengths + total length
        float totalLen = 0f;
        int segN = ptsHead.Length - 1;
        float[] segLen = new float[segN];

        for (int i = 0; i < segN; i++)
        {
            float L = Vector3.Distance(ptsHead[i], ptsHead[i + 1]);
            segLen[i] = L;
            totalLen += L;
        }
        if (totalLen < 1e-8f) return -1f;

        // Find closest projection onto any segment
        int bestIdx = 0;
        float bestT = 0f;
        float bestDist2 = float.PositiveInfinity;

        for (int i = 0; i < segN; i++)
        {
            Vector3 a = ptsHead[i];
            Vector3 b = ptsHead[i + 1];
            Vector3 ab = b - a;
            float ab2 = Vector3.Dot(ab, ab);
            if (ab2 < 1e-10f) continue;

            float t = Vector3.Dot(contactHead - a, ab) / ab2;
            t = Mathf.Clamp01(t);

            Vector3 proj = a + t * ab;
            float d2 = (contactHead - proj).sqrMagnitude;

            if (d2 < bestDist2)
            {
                bestDist2 = d2;
                bestIdx = i;
                bestT = t;
            }
        }

        bestDist = Mathf.Sqrt(bestDist2);

        // Arc length up to projection point
        float arc = 0f;
        for (int i = 0; i < bestIdx; i++) arc += segLen[i];
        arc += bestT * segLen[bestIdx];

        float u = arc / totalLen; // u=0 at ptsHead[0], u=1 at ptsHead[last]

        // Want base=1, tip=0
        bool baseIsIndex0 = ptsHead[0].sqrMagnitude < ptsHead[ptsHead.Length - 1].sqrMagnitude;
        float s = baseIsIndex0 ? (1f - u) : u;

        return Mathf.Clamp01(s);
    }

    void CreateCollisionSegments()
    {
        collisionSegments = Mathf.Max(1, collisionSegments);
        collisionSegs = new WhiskerCollisionSegment[collisionSegments];

        for (int i = 0; i < collisionSegments; i++)
        {
            var child = new GameObject($"{gameObject.name}_Seg{i}");
            child.transform.SetParent(transform, worldPositionStays: false);

            // segment collision component
            var seg = child.AddComponent<WhiskerCollisionSegment>();

            // forwarder must be on the same GO that receives physics callbacks
            if (child.GetComponent<WhiskerSegmentForwarder>() == null)
                child.AddComponent<WhiskerSegmentForwarder>();

            // Rigidbody also must be on that GO (kinematic is fine)
            var rb = child.GetComponent<Rigidbody>();
            if (rb == null) rb = child.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;

            collisionSegs[i] = seg;
        }
    }

    private static string GetPath(Transform t)
    {
        var parts = new System.Collections.Generic.List<string>();
        while (t != null) { parts.Add(t.name); t = t.parent; }
        parts.Reverse();
        return string.Join("/", parts);
    }

    public void RegisterContactFromSegment(Collider other, Vector3 segmentWorldPos)
    {
        hasContact = true;
        lastContactObjectName = other.name;

        if (currentShapeHead == null || currentShapeHead.Length < 2)
        {
            Debug.LogWarning("currentShapeHead not ready yet.");
            SContact = 0f;
            return;
        }

        Vector3 worldContact = other.ClosestPoint(segmentWorldPos);            // ClosestPoint is returned in WORLD space
        Vector3 contactHead = headOrigin.InverseTransformPoint(worldContact);  // convert world to head coordiantes

        lastContactPoint = contactHead;

        SContact = ComputeSFromContact(contactHead, currentShapeHead, out float dist);
    }

    public bool RegisterRefinedContact(
        Collider other,
        float searchStart,
        float searchEnd,
        float contactRadius)
    {
        if (other == null || currentShapeWorld == null || currentShapeWorld.Length < 2)
            return false;

        searchStart = Mathf.Clamp01(searchStart);
        searchEnd = Mathf.Clamp01(searchEnd);
        if (searchEnd < searchStart)
        {
            float tmp = searchStart;
            searchStart = searchEnd;
            searchEnd = tmp;
        }

        // Sample twice per source-curve interval within this broad-phase capsule.
        // Only triggered capsules run this refinement, so the collider count and
        // normal per-frame cost stay bounded.
        int curveIntervals = currentShapeWorld.Length - 1;
        int sampleCount = Mathf.Max(
            2,
            Mathf.CeilToInt((searchEnd - searchStart) * curveIntervals * 2f)
        );

        float threshold = Mathf.Max(0.0001f, contactRadius);
        float previousU = searchStart;
        float previousDistance = DistanceToCollider(other, SampleAlong(currentShapeWorld, previousU));

        if (previousDistance <= threshold)
            return CommitRefinedContact(other, previousU);

        float bestU = previousU;
        float bestDistance = previousDistance;

        for (int i = 1; i <= sampleCount; i++)
        {
            float u = Mathf.Lerp(searchStart, searchEnd, (float)i / sampleCount);
            float distance = DistanceToCollider(other, SampleAlong(currentShapeWorld, u));

            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestU = u;
            }

            if (distance <= threshold)
            {
                // Refine the first base-to-tip threshold crossing. Eight bounded
                // iterations are enough to make s continuous at CSV precision.
                float outsideU = previousU;
                float insideU = u;
                for (int iteration = 0; iteration < 8; iteration++)
                {
                    float midpointU = 0.5f * (outsideU + insideU);
                    float midpointDistance = DistanceToCollider(
                        other,
                        SampleAlong(currentShapeWorld, midpointU)
                    );

                    if (midpointDistance <= threshold)
                        insideU = midpointU;
                    else
                        outsideU = midpointU;
                }

                return CommitRefinedContact(other, insideU);
            }

            previousU = u;
            previousDistance = distance;
        }

        // Trigger overlap and curve sampling can disagree at capsule end caps.
        // Accept the best curve point only when it is still within the physical
        // whisker radius.
        if (bestDistance <= threshold)
            return CommitRefinedContact(other, bestU);

        return false;
    }

    private static float DistanceToCollider(Collider other, Vector3 worldPoint)
    {
        if (other is BoxCollider box)
        {
            Vector3 localPoint = box.transform.InverseTransformPoint(worldPoint) - box.center;
            Vector3 halfSize = box.size * 0.5f;
            Vector3 closestLocal = new Vector3(
                Mathf.Clamp(localPoint.x, -halfSize.x, halfSize.x),
                Mathf.Clamp(localPoint.y, -halfSize.y, halfSize.y),
                Mathf.Clamp(localPoint.z, -halfSize.z, halfSize.z)
            ) + box.center;
            Vector3 closestWorld = box.transform.TransformPoint(closestLocal);
            return Vector3.Distance(worldPoint, closestWorld);
        }

        Vector3 closest = other.ClosestPoint(worldPoint);
        return Vector3.Distance(worldPoint, closest);
    }

    private bool CommitRefinedContact(Collider other, float curveU)
    {
        Vector3 contactWorld = SampleAlong(currentShapeWorld, curveU);
        Vector3 contactHead = headOrigin.InverseTransformPoint(contactWorld);
        float s = ComputeSFromContact(contactHead, currentShapeHead, out float distance);

        if (s < 0f) return false;

        hasContact = true;
        lastContactObjectName = other.name;
        lastContactPoint = contactHead;

        // s=1 is the strongest/base-most contact. If more than one surface box
        // overlaps this whisker, retain the contact closest to the head.
        SContact = Mathf.Max(SContact, s);
        return true;
    }

    public void ClearContact()
    {
        hasContact = false;
        SContact = 0f;
        lastContactObjectName = "";
    }

    public void ResetContactInfo() {
        hasContact = false;
        lastContactPoint = Vector3.zero;
        lastContactNormal = Vector3.zero;
        lastContactObjectName = "";

        SContact = 0f;
    }

    public void EvaluateContacts(int stimulusLayerMask)
    {
        if (collisionSegs == null) return;

        foreach (WhiskerCollisionSegment segment in collisionSegs)
        {
            if (segment != null)
                segment.EvaluateContacts(this, stimulusLayerMask);
        }
    }

    Vector3 SampleAlong(Vector3[] pts, float s)
    {
        s = Mathf.Clamp01(s);
        int n = pts.Length;
        float f = s * (n - 1);
        int i0 = Mathf.Min(n - 2, Mathf.FloorToInt(f));
        int i1 = i0 + 1;
        float u = f - i0;
        return Vector3.Lerp(pts[i0], pts[i1], u);
    }

    void UpdateCollisionSegments(Vector3[] interpolatedPositions)
    {
        if (collisionSegs == null || collisionSegs.Length == 0) return;
        if (interpolatedPositions == null || interpolatedPositions.Length < 2) return;

        int N = collisionSegs.Length;

        for (int i = 0; i < N; i++)
        {
            // arc length fraction covered by this segment along the whisker
            float s0 = (i + 0.1f) / (N + 0.2f);
            float s1 = (i + 0.9f) / (N + 0.2f);

            Vector3 p0 = SampleAlong(interpolatedPositions, s0);
            Vector3 p1 = SampleAlong(interpolatedPositions, s1);

            float searchS0 = (float)i / N;
            float searchS1 = (float)(i + 1) / N;

            collisionSegs[i].SetFromEndpoints(
                p0,
                p1,
                colliderRadius,
                searchS0,
                searchS1
            );
        }
    }

    void OnDestroy() {
        if (bakedMesh != null) {
            Destroy(bakedMesh);
        }
    }

    public bool HasContact() {
        return hasContact;
    }

    public Vector3 GetLastContactPoint() {
        return lastContactPoint;
    }

    // For generating objects for behavioural analysis
    public Vector3 GetTipWorld()
    {
        if (lineRenderer != null && lineRenderer.positionCount > 0)
            return lineRenderer.GetPosition(lineRenderer.positionCount - 1);

        if (currentShapeWorld != null && currentShapeWorld.Length > 0)
            return currentShapeWorld[currentShapeWorld.Length - 1];

        return transform.position;
    }

    public Vector3 GetBaseWorld()
    {
        if (lineRenderer != null && lineRenderer.positionCount > 0)
            return lineRenderer.GetPosition(0);

        if (currentShapeWorld != null && currentShapeWorld.Length > 0)
            return currentShapeWorld[0];

        return transform.position;
    }

    public Vector3 GetDirectionWorld() {
        Vector3 a = GetBaseWorld();
        Vector3 b = GetTipWorld();
        Vector3 d = b - a;
        return d.sqrMagnitude > 1e-8f ? d.normalized : transform.forward;
    }

    void Start() {
        if (eyeCamera == null) {
            var rightCam = GameObject.Find("Right Eye Camera");
            if (rightCam == null)
                rightCam = GameObject.Find("RightEyeCamera");
            if (rightCam) eyeCamera = rightCam.GetComponent<Camera>();
        }
    }
}

