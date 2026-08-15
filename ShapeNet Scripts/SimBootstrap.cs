using UnityEngine;

public class SimBootstrap : MonoBehaviour
{
    [Range(1f, 10f)] public float timeScale = 3f;

    void Awake()
    {
        Application.runInBackground = true;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;      // as fast as possible
        Time.timeScale = timeScale;            // 3–6x is usually stable
        Time.maximumDeltaTime = 0.1f;
        Debug.Log($"[SimBootstrap] timeScale={Time.timeScale}");
    }
}