using UnityEngine;

public class PassiveSweep : MonoBehaviour
{
    public float sceneUnitsPerMm = 0.25f;
    public Vector3 headOffsetUnits = new Vector3(0f, -4f, -20f);

    public float x_mm, y_mm, startZ_mm, endZ_mm, speed_mm_s = 30f, yawDeg;
    public float preHold = 0.5f, postHold = 0.5f;
    public PassiveSweepLogger logger;

    public bool IsRunning { get; private set; }

    bool configured = false;
    Vector3 A, B;
    float t0, pathLenUnits;

    Rigidbody rb;
    Quaternion sweepRot;

    enum State { Pre, Sweep, Post, Done } State state = State.Pre;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }
        rb.isKinematic = true;   // Motion is controlled kinematically.
        rb.useGravity = false;
    }

    public void Configure(
        float sceneUnitsPerMm, Vector3 headOffsetUnits,
        float H_mm,
        float y_mm,
        float startZ_mm, float endZ_mm,
        float speed_mm_s, float yawDeg,
        float preHold, float postHold,
        PassiveSweepLogger logger)
    {
        this.sceneUnitsPerMm = sceneUnitsPerMm;
        this.headOffsetUnits = headOffsetUnits;
        this.x_mm = H_mm;
        this.y_mm = y_mm;
        this.startZ_mm = startZ_mm;
        this.endZ_mm   = endZ_mm;
        this.speed_mm_s = speed_mm_s;
        this.yawDeg     = yawDeg;
        this.preHold    = preHold;
        this.postHold   = postHold;
        this.logger     = logger;

        float u = sceneUnitsPerMm;

        A = headOffsetUnits + new Vector3(H_mm * u, y_mm * u, startZ_mm * u);
        B = headOffsetUnits + new Vector3(H_mm * u, y_mm * u,   endZ_mm * u);

        sweepRot = Quaternion.AngleAxis(yawDeg, Vector3.up);

        // Initialize state & timers properly
        state = State.Pre;
        t0 = Time.time;
        IsRunning = true;
        configured = true;
        enabled = true;

        // Set initial pose through physics
        rb.position = A;
        rb.rotation = sweepRot;

        pathLenUnits = Vector3.Distance(A, B);

    }

    void FixedUpdate()
    {
        if (!configured) return;

        float now = Time.time;

        switch (state)
        {
            case State.Pre:
                // keep object at A during pre-hold
                rb.MovePosition(A);
                rb.MoveRotation(sweepRot);

                if (now - t0 >= preHold)
                {
                    state = State.Sweep;
                    t0 = now;
                }
                break;

            case State.Sweep:
                float v = speed_mm_s * sceneUnitsPerMm; // units/sec
                float a = (pathLenUnits > 1e-6f)
                    ? Mathf.Clamp01(v * (now - t0) / pathLenUnits)
                    : 1f;

                Vector3 pos = Vector3.Lerp(A, B, a);
                rb.MovePosition(pos);
                rb.MoveRotation(sweepRot);

                if (a >= 1f)
                {
                    state = State.Post;
                    t0 = now;
                }
                break;

            case State.Post:
                // keep object at B during post-hold
                rb.MovePosition(B);
                rb.MoveRotation(sweepRot);

                if (now - t0 >= postHold)
                {
                    state = State.Done;
                    IsRunning = false;
                    enabled = false;
                }
                break;
        }
    }
}
