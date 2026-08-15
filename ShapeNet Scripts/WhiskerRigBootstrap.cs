using System.Collections;
using UnityEngine;

public class WhiskerRigBootstrap : MonoBehaviour
{
    public WhiskerManager whiskerManager;

    IEnumerator Start()
    {
        if (whiskerManager == null)
        {
            Debug.LogError("[WhiskerRigBootstrap] Assign WhiskerManager in the Inspector.");
            yield break;
        }

        while (!whiskerManager.IsReady && !whiskerManager.InitializationFailed)
            yield return null;

        if (whiskerManager.InitializationFailed)
        {
            Debug.LogError("[WhiskerRigBootstrap] Right-whisker initialization failed.");
            yield break;
        }

        Debug.Log(
            $"[WhiskerRigBootstrap] Right-only rig ready " +
            $"({whiskerManager.WhiskerCount} whiskers)."
        );
    }
}
