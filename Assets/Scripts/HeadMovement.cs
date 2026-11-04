using UnityEngine;

public class HeadMovement : MonoBehaviour
{
    public float rotationAmplitude = 15f;
    public float bobAmplitude = 0.5f;
    public float rotationSpeed = 1f;
    public float moveAmplitude = 1f;
    public float moveSpeed = 0.5f;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    void Start()
    {
        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    void Update()
    {
        float t = Time.time * rotationSpeed;

        // Bobbing (up and down)
        float bobOffset = Mathf.Sin(t) * bobAmplitude;

        // Rotating (tilting)
        float rotationAngle = Mathf.Sin(t) * rotationAmplitude;
        Quaternion rotation = initialRotation * Quaternion.Euler(0f, rotationAngle, 0f);

        float moveX = Mathf.Sin(t * moveSpeed) * moveAmplitude;
        float moveZ = Mathf.Cos(t * moveSpeed) * moveAmplitude;
        Vector3 movementOffset = new Vector3(moveX, 0f, moveZ);

        // Apply
        transform.SetPositionAndRotation(initialPosition + movementOffset + new Vector3(0f, bobOffset, 0f), rotation);
    }
}
