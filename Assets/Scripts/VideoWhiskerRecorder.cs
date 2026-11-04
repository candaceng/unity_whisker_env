using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public class VideoWhiskerRecorder : MonoBehaviour
{
    public const string RightImageFileName = "frame_right_0000.png";
    public const string TactileFileName = "whiskers.csv";
    public const int ImageCaptureSourceFrame = 35;

    public Camera rightEyeCamera;

    public int captureW = 256;
    public int captureH = 256;
    public int outW = 64;
    public int outH = 64;
    public int frameCap = WhiskerManager.SourceFrameCount;

    public WhiskerManager whiskerManager;

    private RenderTexture captureRT;
    private Texture2D outTex;
    private StreamWriter tactileWriter;

    private readonly List<string> whiskerNames =
        new List<string>(WhiskerManager.RightWhiskerCount);
    private readonly Dictionary<string, Whisker> whiskerMap =
        new Dictionary<string, Whisker>(WhiskerManager.RightWhiskerCount);

    private int currentFrame;
    private bool isRecording;
    private bool manuallyDriven;
    private bool imageCaptured;
    private string savePath;

    public bool IsRecording => isRecording;
    public int CurrentFrame => currentFrame;

    void LateUpdate()
    {
        if (isRecording && !manuallyDriven)
            CaptureFrame();
    }

    public void SetOutputPath(string path)
    {
        savePath = Path.GetFullPath(path);
    }

    public bool StartRecording(bool manualCapture = false)
    {
        if (isRecording)
        {
            Debug.LogError("[Recorder] A recording is already active.");
            return false;
        }

        if (rightEyeCamera == null || whiskerManager == null)
        {
            Debug.LogError(
                "[Recorder] RightEyeCamera and WhiskerManager references are required."
            );
            return false;
        }

        if (!whiskerManager.IsReady ||
            whiskerManager.whiskerNames.Count != WhiskerManager.RightWhiskerCount)
        {
            Debug.LogError(
                $"[Recorder] Expected a ready {WhiskerManager.RightWhiskerCount}-whisker " +
                "right-only rig."
            );
            return false;
        }

        if (string.IsNullOrWhiteSpace(savePath))
        {
            Debug.LogError("[Recorder] Output path is not set.");
            return false;
        }

        Directory.CreateDirectory(savePath);
        if (!EnsureCaptureBuffers())
            return false;
        ConfigureRightEye();

        whiskerNames.Clear();
        whiskerNames.AddRange(whiskerManager.whiskerNames);
        whiskerMap.Clear();

        foreach (Whisker whisker in FindObjectsByType<Whisker>(FindObjectsSortMode.None))
        {
            if (!whisker.name.StartsWith("R", StringComparison.Ordinal))
                continue;

            whisker.eyeCamera = rightEyeCamera;
            whisker.targetPixelHeight = outH;
            whiskerMap[whisker.name] = whisker;
        }

        foreach (string whiskerName in whiskerNames)
        {
            if (!whiskerMap.ContainsKey(whiskerName))
            {
                Debug.LogError($"[Recorder] Missing right whisker {whiskerName}.");
                return false;
            }
        }

        string tactilePath = Path.Combine(savePath, TactileFileName);
        tactileWriter = new StreamWriter(
            tactilePath,
            false,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            bufferSize: 16 * 1024
        );

        currentFrame = 0;
        manuallyDriven = manualCapture;
        imageCaptured = false;
        isRecording = true;
        return true;
    }

    private bool EnsureCaptureBuffers()
    {
        if (captureRT == null || captureRT.width != captureW ||
            captureRT.height != captureH)
        {
            ReleaseRenderTextures();

            captureRT = new RenderTexture(
                captureW,
                captureH,
                24,
                RenderTextureFormat.ARGB32
            )
            {
                useMipMap = false,
                autoGenerateMips = false,
                filterMode = FilterMode.Bilinear
            };
            captureRT.Create();
        }

        if (outTex == null || outTex.width != outW || outTex.height != outH)
        {
            if (outTex != null)
                Destroy(outTex);
            outTex = new Texture2D(outW, outH, TextureFormat.RGB24, false);
        }

        return true;
    }

    private void ConfigureRightEye()
    {
        float aspect = (float)captureW / Mathf.Max(1, captureH);
        rightEyeCamera.aspect = aspect;
        rightEyeCamera.allowMSAA = false;

        rightEyeCamera.forceIntoRenderTexture = true;
        rightEyeCamera.enabled = false;

        UniversalAdditionalCameraData urp = rightEyeCamera.GetUniversalAdditionalCameraData();
        urp.renderPostProcessing = false;
        urp.antialiasing = AntialiasingMode.None;
        urp.SetRenderer(0);
        urp.cameraStack.Clear();

        int uiLayer = LayerMask.NameToLayer("UI");
        if (uiLayer >= 0)
            rightEyeCamera.cullingMask &= ~(1 << uiLayer);
    }

    public void ResetContacts()
    {
        foreach (Whisker whisker in whiskerMap.Values)
        {
            if (whisker != null)
                whisker.ResetContactInfo();
        }
    }

    /// <summary>
    /// Writes one frame-major block of 30 ordered (s, theta_deg) rows.
    /// The right-eye image is rendered once at the +25-degree source pose.
    /// </summary>
    public bool CaptureFrame()
    {
        if (!isRecording || tactileWriter == null)
            return false;

        if (currentFrame >= frameCap)
        {
            StopRecording();
            return false;
        }

        if (!imageCaptured && currentFrame == ImageCaptureSourceFrame)
        {
            string imagePath = Path.Combine(savePath, RightImageFileName);
            if (!CaptureRightEyeAndSave(imagePath))
            {
                StopRecording();
                return false;
            }
            imageCaptured = true;
        }

        foreach (string whiskerName in whiskerNames)
        {
            Whisker whisker = whiskerMap[whiskerName];
            float s = whisker.HasContact() ? whisker.SContact : 0f;
            float theta = whisker.thetaWDeg;

            tactileWriter.Write(s.ToString("F4", CultureInfo.InvariantCulture));
            tactileWriter.Write(',');
            tactileWriter.WriteLine(theta.ToString("F4", CultureInfo.InvariantCulture));
        }

        ResetContacts();
        currentFrame++;

        if (currentFrame >= frameCap)
            StopRecording();

        return true;
    }

    /// <summary>
    /// Renders one image through the same right-eye buffers, camera settings,
    /// renderer, and downsampling path used by the paired training trials.
    /// This does not open a tactile file or change the recorder frame counter.
    /// </summary>
    public bool CaptureRightEyeImage(string filePath)
    {
        if (rightEyeCamera == null)
        {
            Debug.LogError("[Recorder] RightEyeCamera is required for image capture.");
            return false;
        }

        string parent = Path.GetDirectoryName(Path.GetFullPath(filePath));
        if (!string.IsNullOrEmpty(parent))
            Directory.CreateDirectory(parent);
        if (!EnsureCaptureBuffers())
            return false;
        ConfigureRightEye();
        return CaptureRightEyeAndSave(Path.GetFullPath(filePath));
    }

    private bool CaptureRightEyeAndSave(string filePath)
    {
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = rightEyeCamera.targetTexture;
        float previousAspect = rightEyeCamera.aspect;
        RenderTexture downsampled = null;

        try
        {
            rightEyeCamera.targetTexture = captureRT;
            rightEyeCamera.aspect = (float)captureW / Mathf.Max(1, captureH);
            rightEyeCamera.Render();

            downsampled = RenderTexture.GetTemporary(
                outW,
                outH,
                0,
                RenderTextureFormat.ARGB32
            );
            downsampled.filterMode = FilterMode.Bilinear;
            Graphics.Blit(captureRT, downsampled);

            RenderTexture.active = downsampled;
            outTex.ReadPixels(new Rect(0, 0, outW, outH), 0, 0);
            outTex.Apply();
            File.WriteAllBytes(filePath, outTex.EncodeToPNG());
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError($"[Recorder] Failed to write {filePath}: {exception}");
            return false;
        }
        finally
        {
            rightEyeCamera.aspect = previousAspect;
            rightEyeCamera.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
            if (downsampled != null)
                RenderTexture.ReleaseTemporary(downsampled);
        }
    }

    public void StopRecording()
    {
        if (!isRecording && tactileWriter == null)
            return;

        isRecording = false;
        manuallyDriven = false;

        tactileWriter?.Flush();
        tactileWriter?.Dispose();
        tactileWriter = null;
    }

    private void ReleaseRenderTextures()
    {
        if (captureRT != null)
        {
            captureRT.Release();
            Destroy(captureRT);
            captureRT = null;
        }
    }

    void OnDestroy()
    {
        StopRecording();
        ReleaseRenderTextures();

        if (outTex != null)
        {
            Destroy(outTex);
            outTex = null;
        }
    }
}
