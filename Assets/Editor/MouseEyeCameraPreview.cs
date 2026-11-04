using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

/// <summary>
/// Isolated, editor-only visual check for the right-eye camera. This utility
/// never saves the opened scene and does not participate in dataset generation.
/// </summary>
public static class MouseEyeCameraPreview
{
    private const string ScenePath = "Assets/Scenes/ShapeNetScene.unity";
    private const string OutputDirectory = "Data/CameraPreview";

    private static readonly Vector3 EyeLocalPosition =
        new Vector3(7.2068f, 5.5667f, -14.3873f);

    private const float CurrentYawDegrees = 55f;
    private const float CurrentHorizontalFovDegrees = 90f;
    private const float ProposedYawDegrees = 64f;
    private const float ProposedElevationDegrees = 22f;
    private const float ProposedHorizontalFovDegrees = 180f;
    private const float ProposedVerticalFovDegrees = 140f;

    [MenuItem("Tools/Camera Preview/Generate Mouse Eye Comparison")]
    public static void Generate()
    {
        Scene scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        Transform head = scene.GetRootGameObjects()
            .SelectMany(root => root.GetComponentsInChildren<Transform>(true))
            .FirstOrDefault(candidate => candidate.name == "HeadOrigin");
        if (head == null)
            throw new InvalidOperationException("HeadOrigin was not found.");

        string outputRoot = Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", OutputDirectory)
        );
        Directory.CreateDirectory(outputRoot);

        var disabledStimuli = DisableExistingStimuli(scene);
        GameObject previewRoot = new GameObject("MouseEyeCameraPreview (temporary)");

        try
        {
            AddPreviewLighting(previewRoot.transform);
            AddGroundAndReferenceObjects(previewRoot.transform, head);

            Vector3 eyeWorldPosition = head.TransformPoint(EyeLocalPosition);
            Quaternion currentRotation =
                head.rotation * Quaternion.Euler(0f, CurrentYawDegrees, 0f);
            Quaternion proposedRotation = GetProposedRotation(head);

            Camera renderCamera = CreateRenderCamera(
                "MouseEyePreviewCamera",
                previewRoot.transform
            );
            renderCamera.transform.position = eyeWorldPosition;
            renderCamera.transform.rotation = currentRotation;
            RenderPerspective(
                renderCamera,
                Path.Combine(outputRoot, "current_90deg_eye_view.png"),
                768,
                768,
                CurrentHorizontalFovDegrees
            );

            RenderAngularEyeView(
                renderCamera,
                eyeWorldPosition,
                proposedRotation,
                Path.Combine(outputRoot, "proposed_180x140_eye_view.png"),
                900,
                700
            );

            AddCameraMarkers(
                previewRoot.transform,
                eyeWorldPosition,
                currentRotation,
                proposedRotation
            );

            renderCamera.transform.position = head.TransformPoint(
                new Vector3(38f, 31f, -42f)
            );
            Vector3 overviewTarget = head.TransformPoint(new Vector3(6f, 5f, -2f));
            renderCamera.transform.rotation = Quaternion.LookRotation(
                overviewTarget - renderCamera.transform.position,
                head.up
            );
            RenderPerspective(
                renderCamera,
                Path.Combine(outputRoot, "camera_placement_overview.png"),
                900,
                700,
                42f
            );

            File.WriteAllText(
                Path.Combine(outputRoot, "README.txt"),
                "Mouse-eye camera preview (editor-only; scene not saved).\n" +
                "Red marker/ray: current camera, yaw 55 deg, 90 deg horizontal FOV.\n" +
                "Cyan marker/ray: proposed camera, yaw 64 deg, elevation 22 deg.\n" +
                "Proposed eye image: angular 180 deg horizontal x 140 deg vertical projection.\n" +
                "Both views use the existing head-local eye position " +
                EyeLocalPosition.ToString("F4") + ".\n"
            );

            Debug.Log(
                "[MouseEyeCameraPreview] Generated previews at " + outputRoot
            );
        }
        finally
        {
            foreach (KeyValuePair<GameObject, bool> entry in disabledStimuli)
            {
                if (entry.Key != null)
                    entry.Key.SetActive(entry.Value);
            }

            if (previewRoot != null)
                UnityEngine.Object.DestroyImmediate(previewRoot);
        }
    }

    private static Dictionary<GameObject, bool> DisableExistingStimuli(Scene scene)
    {
        var states = new Dictionary<GameObject, bool>();
        int stimulusLayer = LayerMask.NameToLayer("Stimulus");
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            foreach (Transform item in root.GetComponentsInChildren<Transform>(true))
            {
                GameObject gameObject = item.gameObject;
                if (gameObject == null || states.ContainsKey(gameObject))
                    continue;

                bool isStimulus =
                    (stimulusLayer >= 0 && gameObject.layer == stimulusLayer) ||
                    gameObject.CompareTag("Stimulus");
                if (!isStimulus)
                    continue;

                states.Add(gameObject, gameObject.activeSelf);
                gameObject.SetActive(false);
            }
        }
        return states;
    }

    private static void AddPreviewLighting(Transform parent)
    {
        GameObject lightObject = new GameObject("Preview Directional Light");
        lightObject.transform.SetParent(parent, false);
        lightObject.transform.rotation = Quaternion.Euler(42f, -28f, 0f);
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.25f;
        light.color = new Color(1f, 0.95f, 0.88f);

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.34f, 0.38f, 0.46f);
    }

    private static void AddGroundAndReferenceObjects(Transform parent, Transform head)
    {
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Preview Ground";
        ground.transform.SetParent(parent, false);
        ground.transform.position = head.TransformPoint(new Vector3(8f, -3f, 4f));
        ground.transform.rotation = head.rotation;
        ground.transform.localScale = new Vector3(6f, 1f, 6f);
        SetColor(ground, new Color(0.22f, 0.25f, 0.20f));

        var placements = new[]
        {
            new PreviewObject(PrimitiveType.Sphere, new Vector3(6f, 0f, -5f), 3.2f,
                new Color(0.95f, 0.45f, 0.12f)),
            new PreviewObject(PrimitiveType.Cube, new Vector3(10f, 0f, 0f), 4.5f,
                new Color(0.16f, 0.64f, 0.95f)),
            new PreviewObject(PrimitiveType.Sphere, new Vector3(15f, 0f, 5f), 6f,
                new Color(0.72f, 0.28f, 0.86f)),
            new PreviewObject(PrimitiveType.Cube, new Vector3(19f, 0f, -7f), 3.4f,
                new Color(0.20f, 0.78f, 0.42f)),
            new PreviewObject(PrimitiveType.Cube, new Vector3(7f, 0f, 9f), 5f,
                new Color(0.95f, 0.78f, 0.16f))
        };

        for (int index = 0; index < placements.Length; index++)
        {
            PreviewObject placement = placements[index];
            GameObject item = GameObject.CreatePrimitive(placement.type);
            item.name = $"Preview Obstacle {index + 1}";
            item.transform.SetParent(parent, false);
            item.transform.position = head.TransformPoint(placement.localPosition);
            item.transform.rotation = head.rotation;
            item.transform.localScale = Vector3.one * placement.scale;
            SetColor(item, placement.color);
        }
    }

    private static Quaternion GetProposedRotation(Transform head)
    {
        float azimuth = ProposedYawDegrees * Mathf.Deg2Rad;
        float elevation = ProposedElevationDegrees * Mathf.Deg2Rad;
        Vector3 localDirection = new Vector3(
            Mathf.Sin(azimuth) * Mathf.Cos(elevation),
            Mathf.Sin(elevation),
            Mathf.Cos(azimuth) * Mathf.Cos(elevation)
        );
        Vector3 worldDirection = head.TransformDirection(localDirection).normalized;
        return Quaternion.LookRotation(worldDirection, head.up);
    }

    private static Camera CreateRenderCamera(string name, Transform parent)
    {
        GameObject cameraObject = new GameObject(name);
        cameraObject.transform.SetParent(parent, true);
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.enabled = false;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.52f, 0.68f, 0.83f, 1f);
        camera.nearClipPlane = 0.01f;
        camera.farClipPlane = 120f;
        camera.allowHDR = false;
        camera.allowMSAA = true;
        return camera;
    }

    private static void RenderPerspective(
        Camera camera,
        string path,
        int width,
        int height,
        float horizontalFov)
    {
        float aspect = (float)width / height;
        camera.aspect = aspect;
        camera.fieldOfView = Camera.HorizontalToVerticalFieldOfView(
            horizontalFov,
            aspect
        );

        RenderTexture texture = new RenderTexture(
            width,
            height,
            24,
            RenderTextureFormat.ARGB32
        );
        texture.antiAliasing = 4;
        texture.Create();
        camera.targetTexture = texture;
        camera.Render();
        SaveRenderTexture(texture, path);
        camera.targetTexture = null;
        texture.Release();
        UnityEngine.Object.DestroyImmediate(texture);
    }

    private static void RenderAngularEyeView(
        Camera camera,
        Vector3 eyePosition,
        Quaternion eyeRotation,
        string path,
        int width,
        int height)
    {
        Shader shader = Shader.Find("Hidden/MouseEyeEquirectPreview");
        if (shader == null)
            throw new InvalidOperationException(
                "Hidden/MouseEyeEquirectPreview shader was not found."
            );

        RenderTexture cube = new RenderTexture(
            512,
            512,
            24,
            RenderTextureFormat.ARGB32
        );
        cube.dimension = TextureDimension.Cube;
        cube.volumeDepth = 6;
        cube.useMipMap = false;
        cube.autoGenerateMips = false;
        cube.Create();

        camera.transform.position = eyePosition;
        camera.transform.rotation = Quaternion.identity;
        camera.fieldOfView = 90f;
        camera.aspect = 1f;
        camera.RenderToCubemap(cube);

        Material material = new Material(shader);
        material.SetTexture("_Cube", cube);
        material.SetMatrix("_EyeToWorld", Matrix4x4.Rotate(eyeRotation));
        material.SetFloat(
            "_HorizontalHalfRadians",
            0.5f * ProposedHorizontalFovDegrees * Mathf.Deg2Rad
        );
        material.SetFloat(
            "_VerticalHalfRadians",
            0.5f * ProposedVerticalFovDegrees * Mathf.Deg2Rad
        );

        RenderTexture angular = new RenderTexture(
            width,
            height,
            0,
            RenderTextureFormat.ARGB32
        );
        angular.Create();
        Graphics.Blit(null, angular, material);
        SaveRenderTexture(angular, path);

        RenderTexture.active = null;
        angular.Release();
        cube.Release();
        UnityEngine.Object.DestroyImmediate(angular);
        UnityEngine.Object.DestroyImmediate(cube);
        UnityEngine.Object.DestroyImmediate(material);
    }

    private static void SaveRenderTexture(RenderTexture source, string path)
    {
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = source;
        Texture2D texture = new Texture2D(
            source.width,
            source.height,
            TextureFormat.RGB24,
            false
        );
        texture.ReadPixels(
            new Rect(0f, 0f, source.width, source.height),
            0,
            0
        );
        texture.Apply();
        File.WriteAllBytes(path, texture.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(texture);
        RenderTexture.active = previous;
    }

    private static void AddCameraMarkers(
        Transform parent,
        Vector3 eyePosition,
        Quaternion currentRotation,
        Quaternion proposedRotation)
    {
        Color currentColor = new Color(0.95f, 0.12f, 0.10f);
        Color proposedColor = new Color(0.00f, 0.88f, 0.92f);

        AddMarkerSphere(parent, eyePosition, currentColor, 0.7f);
        AddDirectionRay(
            parent,
            eyePosition,
            currentRotation * Vector3.forward,
            currentColor,
            15f
        );
        AddDirectionRay(
            parent,
            eyePosition + Vector3.up * 0.16f,
            proposedRotation * Vector3.forward,
            proposedColor,
            17f
        );
    }

    private static void AddMarkerSphere(
        Transform parent,
        Vector3 position,
        Color color,
        float scale)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.name = "Eye Position Marker";
        marker.transform.SetParent(parent, false);
        marker.transform.position = position;
        marker.transform.localScale = Vector3.one * scale;
        SetColor(marker, color);
    }

    private static void AddDirectionRay(
        Transform parent,
        Vector3 origin,
        Vector3 direction,
        Color color,
        float length)
    {
        GameObject ray = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        ray.name = "Camera Direction Ray";
        ray.transform.SetParent(parent, false);
        Vector3 endpoint = origin + direction.normalized * length;
        ray.transform.position = 0.5f * (origin + endpoint);
        ray.transform.rotation = Quaternion.FromToRotation(Vector3.up, direction);
        ray.transform.localScale = new Vector3(0.12f, 0.5f * length, 0.12f);
        SetColor(ray, color);
    }

    private static void SetColor(GameObject gameObject, Color color)
    {
        Renderer renderer = gameObject.GetComponent<Renderer>();
        if (renderer == null)
            return;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        Material material = new Material(shader);
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        renderer.sharedMaterial = material;
    }

    private readonly struct PreviewObject
    {
        public readonly PrimitiveType type;
        public readonly Vector3 localPosition;
        public readonly float scale;
        public readonly Color color;

        public PreviewObject(
            PrimitiveType type,
            Vector3 localPosition,
            float scale,
            Color color)
        {
            this.type = type;
            this.localPosition = localPosition;
            this.scale = scale;
            this.color = color;
        }
    }
}
