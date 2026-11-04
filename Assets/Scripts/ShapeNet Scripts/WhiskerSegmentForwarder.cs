using UnityEngine;

public class WhiskerSegmentForwarder : MonoBehaviour
{
    private Whisker whisker;
    private WhiskerCollisionSegment collisionSegment;
    [SerializeField] private bool verboseLogging = false;

    void Awake()
    {
        whisker = GetComponentInParent<Whisker>();
        collisionSegment = GetComponent<WhiskerCollisionSegment>();
    }

    static string GetPath(Transform t)
    {
        var parts = new System.Collections.Generic.List<string>();
        while (t != null) { parts.Add(t.name); t = t.parent; }
        parts.Reverse();
        return string.Join("/", parts);
    }

    void OnTriggerEnter(Collider other) => Forward(other);
    void OnTriggerStay(Collider other)  => Forward(other);

    void Forward(Collider other)
    {
        if (verboseLogging)
            Debug.Log($"[Forwarder RAW] seg={name} other={other.name} otherTag={other.tag} otherLayer={LayerMask.LayerToName(other.gameObject.layer)}");

        if (whisker == null)
        {
            Debug.LogWarning($"[Forwarder] whisker NULL on {name} path={GetPath(transform)}");
            return;
        }

        if (!HasStimulusTag(other.transform))
            return;

        if (verboseLogging)
            Debug.Log($"[Forwarder HIT] seg={name} whisker={whisker.name} other={other.name} tag={other.tag}");

        if (collisionSegment != null && whisker.RegisterRefinedContact(
                other,
                collisionSegment.SearchS0,
                collisionSegment.SearchS1,
                collisionSegment.Radius))
            return;

        // Fallback for older/non-primitive stimulus colliders.
        whisker.RegisterContactFromSegment(other, transform.position);
    }

    private static bool HasStimulusTag(Transform candidate)
    {
        while (candidate != null)
        {
            if (candidate.CompareTag("Stimulus")) return true;
            candidate = candidate.parent;
        }
        return false;
    }
}
