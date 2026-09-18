# vision-core

Not a skill — the floor every skill stands on. Anything here is true about the
**camera** or the **field**, never about one task.

| Module | Owns |
| --- | --- |
| `field.py` | **The swap point.** `Field`, `CameraFieldTransform`, `SyntheticFieldTransform`, `ReferenceTagFieldTransform`. |
| `camera.py` | `open_camera`, `list_cameras`, `lock_camera` (auto-exposure / auto-WB off), `read_key` (case-folded `waitKey`). |
| `intrinsics.py` | `load`, `save`, `from_fov`, `hfov_of`. Keyed on resolution. |
| `paths.py` | `repo_root()`, `calib_path()` — so `calib/` resolves regardless of cwd. |
| `kalman.py` | `KalmanFilter` (Mahalanobis gate, Joseph update, angle-wrapped residuals) plus constant-velocity `F`/`Q` builders. The physics stays in each skill. |
| `planview.py` | `project` and `draw_field` for the video overlay; `PlanView` for the top-down panel. |

Import from the submodules — there is no re-exported surface in `__init__.py` to
keep in sync.

```
tag-tracking  ──→  vision-core           one direction, always
ball-tracking ──→  vision-core           and the skills never import each other
```

## Why `ReferenceTagFieldTransform` lives here

It reads AprilTags, so on the face of it it belongs in `tag-tracking`. It does
not: reference tags are **field infrastructure**, not tag business. The ball
tracker has no interest in AprilTags and still needs the field *located* before
it can say where a ball is on it — it loads the very same
`calib/field_pose.json`, through `field_pose_path()` here.

That is also why this repo is a workspace rather than independent folders — see
the [root README](../README.md).
