# vision-core

Not a skill — the floor every skill stands on. Anything here is true about the
**camera** or the **field**, never about one task.

| Module | Owns |
| --- | --- |
| `field.py` | **The swap point.** `Field`, `CameraFieldTransform`, `SyntheticFieldTransform`, `ReferenceTagFieldTransform`. |
| `camera.py` | `open_camera` (always hands back a camera on auto), `list_cameras`, `lock_camera` (manual exposure matched to auto's brightness, auto-WB off unless that tints the picture — each half reverts to auto if it does harm), `unlock_camera` (the lock outlives the process — call it on exit), `read_key` (case-folded `waitKey`). `uv run unlock-camera` for runs that died locked. |
| `intrinsics.py` | `load`, `save`, `from_fov`, `hfov_of`, `undistort_maps` (precomputed `cv2.remap` tables — whole-frame undistortion, for a detector that searches the whole image rather than one already-known point). Keyed on resolution. |
| `charuco.py` | *Measures* the camera matrix, from a real `cv2.aruco.CharucoBoard` — `make_board`, `calibrate_intrinsics` (Zhang's method via `cv2.calibrateCamera`, with a degenerate-view guard), plus `SyntheticChArucoCamera` for a self-test with no hardware. Camera-only, no AprilTag dependency, which is why it lives here and not in `tag-tracking`. |
| `paths.py` | `repo_root()`, `calib_path()` — so `calib/` resolves regardless of cwd. |
| `kalman.py` | `KalmanFilter` (Mahalanobis gate, Joseph update, angle-wrapped residuals) plus constant-velocity `F`/`Q` builders. The physics stays in each skill. |
| `planview.py` | `project` and `draw_field` for the video overlay; `PlanView` for the top-down panel. |

Import from the submodules — there is no re-exported surface in `__init__.py` to
keep in sync.

```
tag-tracking      ──→  vision-core        one direction, always
ball-tracking     ──→  vision-core        tag-tracking and ball-tracking never import each other
combined-tracking ──→  vision-core        the one exception: an integration that
                  ──→  tag-tracking       depends on both existing skills, reusing
                  ──→  ball-tracking      their detectors instead of reimplementing them
```

## Why `ReferenceTagFieldTransform` lives here

It reads AprilTags, so on the face of it it belongs in `tag-tracking`. It does
not: reference tags are **field infrastructure**, not tag business. The ball
tracker has no interest in AprilTags and still needs the field *located* before
it can say where a ball is on it — it loads the very same
`calib/field_pose.json`, through `field_pose_path()` here.

That is also why this repo is a workspace rather than independent folders — see
the [root README](../README.md).
