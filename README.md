# Unity Whisker Environment

A Unity environment for simulating whisker-based sensing and collecting paired visual and tactile data. The main scene combines a mouse head, a right-side whisker array, an eye camera, and procedural obstacles.

## Requirements

- Unity **6000.0.35f1**, installed through Unity Hub.
- The packages declared in `Packages/manifest.json`; Unity restores them when the project is first opened.

## Getting started

1. Clone this repository and add its root folder as a project in Unity Hub.
2. Open `Assets/Scenes/ShapeNetScene.unity`.
3. Select `WhiskerRig` and set a small `TrialManager.totalTrials` value for an initial run. The saved scene is configured for **100,000 trials**, archived in batches of 1,000.
4. Set `TrialManager.outputDirectory` to a fresh project-relative directory under `Data/`, then enter Play Mode.

The main pipeline uses 30 right whiskers across 51 source poses. Each completed trial contains `frame_right_0000.png`, a 64 × 64 grayscale right-eye image, and `whiskers.csv`, containing 1,530 rows of `s,theta_deg` in pose-major order. Contact position `s` runs from 1 at the base to 0 at the tip; 0 also denotes no contact. The image is captured at source pose 35 (+25 degrees). Batches include a format manifest, and a completion marker is written after the run finishes. Use separate output directories for different configurations.

## Project layout

- `Assets/Scripts/`: whisker animation, contact detection, head movement, and paired recording.
- `Assets/Scripts/ShapeNet Scripts/`: existing sweep, trial-plan, and six-condition evaluation utilities.
- `Assets/Editor/`: setup validation, batch launchers, and camera preview utilities.
- `Assets/Scenes/ShapeNetScene.unity`: main environment and build scene.
- `Assets/Resources/`: mouse head, whisker names, and the 51 right-whisker motion CSVs required by the main pipeline.
- `Assets/Settings/`, `Packages/`, and `ProjectSettings/`: rendering assets, dependencies, and Unity configuration.

The three conversion notebooks at the repository root are retained as supporting utilities. Their input data and Python dependencies are separate from the Unity runtime. ShapeNet model files are not bundled; modes that load external ShapeNet models require a dataset at the configured `shapenetRoot` path. The main procedural-obstacle pipeline does not require ShapeNet downloads.

## Validation

Use **Whisker Simulation → Validate Right-Only Setup** to check the scene and input resources. From a terminal, with the project closed in the Editor:

```powershell
& '<Unity installation>/Editor/Unity.exe' -batchmode -projectPath '<repository path>' -executeMethod RightOnlySimulationValidator.ValidateOnly -logFile '<validation log path>'
& '<Unity installation>/Editor/Unity.exe' -batchmode -projectPath '<repository path>' -executeMethod RightOnlySimulationSmokeLauncher.Run -logFile '<smoke log path>'
```

The smoke launcher generates 30 temporary trials, verifies image dimensions, tactile row counts, and physical contacts, then removes its temporary output. These entry points exit Unity when complete; omit `-quit` so asynchronous smoke generation can finish.

Guyoton and controlled-spatial experiment scripts, experiment datasets, recordings, training artifacts, and Unity caches are excluded from this project snapshot.
