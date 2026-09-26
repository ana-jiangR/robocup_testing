# calib

Calibration describing the **camera**, not any one skill. Shared by every skill
and resolved from the repo root by `vision_core.paths.repo_root()`, so it does
not matter which folder you launch from. Created on first write — nothing to set
up.

| File | Made by | What it is |
| --- | --- | --- |
| `intrinsics.json` | `s` in `track`, or `calibrate-camera` | The camera matrix: how zoomed-in the lens is, and (from `calibrate-camera`) real lens distortion coefficients. Stores the resolution it was measured at, and is ignored on load if that does not match. |
| `field_pose.json` | `calibrate-field`, or `track --calibrate-live` | Where the field is, relative to the camera -- an `R`/`t` solved from four reference AprilTags. Both `track` and `track-ball` load it automatically in place of the made-up field. |
| `ball_color.json` | `calibrate-ball` | Named ball profiles: a measured hue/saturation histogram plus each ball's radius. Pick one with `--ball-profile`. |

Committed, because this team shares one rig.

> ⚠️ `ball_color.json` is the most fragile of the three. A histogram measured
> under one set of lights does not transfer to another room, or to daylight
> versus evening. Each profile carries an `h_spread`/`s_spread` (how far it
> reaches beyond what was sampled) that absorbs a ball rolling between lamps
> and shadows in the *same* room; a committed profile that predates those
> fields gets the defaults on load. If the mask looks wrong, re-run
> `calibrate-ball` rather than assuming a committed profile still applies.

> ⚠️ If you end up on **different webcams**, gitignore `intrinsics.json`. The
> loader only rejects a mismatched *resolution*, so two different cameras both at
> 1280x720 would silently share one wrong `fx` — and every reported distance
> would be off with nothing on screen to flag it.
