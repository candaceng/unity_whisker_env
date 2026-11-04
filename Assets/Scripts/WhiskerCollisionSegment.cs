using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(CapsuleCollider))]
public class WhiskerCollisionSegment : MonoBehaviour
{
    private Rigidbody rb;
    private CapsuleCollider col;

    private Vector3 p0World, p1World;

    private float searchS0, searchS1;

    private float radius;

    public float SearchS0 => searchS0;
    public float SearchS1 => searchS1;
    public float Radius => radius;

    void Awake()
    {
        int whiskerLayer = LayerMask.NameToLayer("Whisker");
        if (whiskerLayer != -1) gameObject.layer = whiskerLayer;

        rb = GetComponent<Rigidbody>();
        col = GetComponent<CapsuleCollider>();

        rb.isKinematic = true;
        rb.useGravity = false;

        col.isTrigger = true;
        col.direction = 1; // capsule height along local Y
        col.providesContacts = true;
    }

    public void SetFromEndpoints(
        Vector3 p0,
        Vector3 p1,
        float colliderRadius,
        float searchS0_,
        float searchS1_)
    {
        p0World = p0;
        p1World = p1;
        radius = colliderRadius;

        searchS0 = Mathf.Clamp01(searchS0_);
        searchS1 = Mathf.Clamp01(searchS1_);

        Vector3 mid = 0.5f * (p0World + p1World);
        Vector3 dir = (p1World - p0World);
        float dist = dir.magnitude;
        if (dist < 1e-6f) dist = 1e-6f;

        transform.position = mid;
        transform.rotation = Quaternion.FromToRotation(Vector3.up, dir / dist);

        col.radius = radius;
        col.height = dist + 2f * radius; // slight extension past endpoints
    }

    public void EvaluateContacts(Whisker whisker, int stimulusLayerMask)
    {
        if (whisker == null) return;

        Collider[] overlaps = Physics.OverlapCapsule(
            p0World,
            p1World,
            radius,
            stimulusLayerMask,
            QueryTriggerInteraction.Collide
        );

        foreach (Collider other in overlaps)
        {
            whisker.RegisterRefinedContact(
                other,
                searchS0,
                searchS1,
                radius
            );
        }
    }
}
