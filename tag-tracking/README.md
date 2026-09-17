# Virtual-field AprilTag tracking

Track an AprilTag held on a phone, and report where it is on a **field that does
not physically exist** — a rectangle defined purely in software as sitting some
distance in front of the webcam.

Built as a proof of concept for a RoboCup setup where the field will be real and
located by four reference AprilTags. The camera-to-field transform is isolated in
one file so that swap costs one subclass and changes nothing downstream.

**Skill 1** of this repo. The geometry it stands on lives in
[`vision-core`](../vision-core/), ready to be shared with the skills that follow;
everything below runs from the repo root.

## Run it in order

### 0 · Check the maths (no camera, no phone)

```powershell
uv run selfcheck
```

Renders synthetic pictures of a tag at known field positions, pushes them through
the real detector and the real `vision_core/field.py`, and checks the numbers come
back.

You should see three sections, all `[ok ]`, ending in *"All checks passed."*
Section 3 is the one that matters for RoboCup: it puts a tag at a known spot,
recovers the field pose from four reference tags, and prints the same tag under
both transforms:

```
same tag, invented transform: x=0.750 y=0.546 th=+29.8
same tag, measured transform: x=0.749 y=0.546 th=+29.8
```

Same numbers, same format, two completely different ways of deciding where the
field is. That is the property this project exists to protect.

If this fails, the problem is in the geometry. If it passes and the live view
looks wrong, the problem is the camera, the intrinsics, or the tag size.

### 1 · Put a tag on the phone at a known size

```powershell
uv run serve-tag                      # tag id 0, 80 mm
uv run serve-tag --tag-id 7 --size-mm 100
```

It prints a URL like `http://10.48.151.49:8000/`. Open that on the phone — both
devices need to be on the same wifi.

The page sizes the tag in **real millimetres**, with no printer and no ruler: you
drag a slider until an on-screen box matches any bank card. Cards are made to
ISO/IEC 7810 ID-1, 85.60 × 53.98 mm, worldwide. That gives the page an exact
pixels-per-millimetre for your phone, and it renders the tag at the size you ask
for.

What you should see: the calibration box, a size field, and a line reading
`--tag-size 0.0800`. That is the number the tracker needs. There is also a bar
that should measure exactly 50 mm if you happen to have a ruler and want to
confirm the card trick worked.

Tap **Full screen tag** for a clean white screen with just the tag. Turn
brightness up and auto-rotate off. Tap the top-right corner to get back.

> The size that matters is the **black square**, edge to edge, including the
> black border ring but not the white margin. That is what the detector measures.
> The page's number is already that measurement.

### 2 · Track it

Leave the server running, open a second terminal:

```powershell
uv run track --tag-size 0.080
```

You should see one window, camera on the left and a plan view on the right:

- a **green rectangle** floating in the scene — the virtual field, with corners
  labelled `(0,0) (W,0) (W,H) (0,H)`, a 20 cm grid, and red/green arrows for
  field +X and +Y at the origin corner;
- your tag **outlined in color** when detected, labelled with its field position;
- a **plan view** with the tag as a dot, a heading arrow, and a fading trail;
- a **black strip along the bottom** with one line per tag:

```
id  0   x +0.612   y +0.402   th   +3.4   off-plane +1.190   IN
```

Move the phone left and the `x` falls; move it up and `y` rises; rotate it and
`th` turns. Carry it past the edge of the green rectangle and the line turns red
and reads `OUT`, and the plan-view dot goes hollow.

Keys: `q` quit · `g` grid · `t` trails · `r` reset trails · `w` wall/floor ·
`[` `]` nudge fx · `s` save intrinsics.

Useful flags:

```powershell
uv run track --field 1.2 0.8 --distance 1.8      # field size, and how far out it floats
uv run track --mode floor                        # field flat, camera overhead 1.5 m up
uv run track --list-cameras                      # if it grabs the wrong one
uv run track --camera 2 --display-width 1300     # pick a camera, shrink the window
uv run track --print-poses                       # stream id/x/y/theta to stdout
```

## What `off-plane` is telling you

`x` and `y` are the tag's true 3D position expressed in field coordinates — not a
projection onto the field plane. `off-plane` is the third coordinate: how far the
tag is from the plane, along the field normal.

Held at arm's length with the field floating at 1.8 m, you will see `off-plane`
around `+1.2`. That is correct and it is not an error: the phone really is about
1.2 m in front of the imaginary rectangle. Since the position is genuinely 3D,
pushing the phone toward the camera while keeping it in the same spot on screen
*does* change `x` and `y`, because parallax is real. That is the tracking working,
not drifting.

If you want the phone to physically sweep the plane, use
`--distance 0.6 --field 0.5 0.3`: a smaller field close enough to reach. A
1.2 × 0.8 m field simply does not fit in a 60° view at arm's length.

## Does webcam calibration matter yet?

**Not yet. Fake the intrinsics for now.** But it is worth knowing exactly what
you are buying and what you are deferring.

`fx` converts apparent tag size in pixels into metric distance. Get it wrong by
20% and every reported *distance from the camera* is wrong by 20%.

The reason that is survivable today: **the virtual field is defined in the same
made-up camera frame the tag is measured in.** The field plane is at 1.8 m in the
fake camera's units; the tag's depth is computed in those same units. The error
is common to both, so "where is the tag on the field" and "is it inside the
bounds" stay self-consistent even with a sloppy `fx`. The demo is internally
honest; it is just not metrically true.

What does *not* cancel, even now:

- **Lens distortion.** Assumed zero here. On a cheap wide webcam this bends
  straight lines by a few percent near the frame edges, so the tag drifts when you
  move it to the corners of view.
- **Absolute scale.** If you tape-measure the phone at 50 cm and the numbers
  disagree, that gap is your `fx` error.

Two cheap upgrades, in order:

1. **Tell it the FOV**: `uv run track --hfov 70` if you know your webcam's spec.
2. **One-parameter fix, which is most of the benefit.** Hold the phone at a
   tape-measured distance, watch the reported depth, and press `[` / `]` until it
   matches. Press `s` to save to `calib/intrinsics.json` **at the repo root**,
   which is loaded automatically next run — and shared with the ball tracker,
   since it describes the camera rather than this skill. This directly calibrates `fx` — the parameter that
   actually matters — usually to within a couple of percent.

> One catch worth knowing: **tag-size error and `fx` error are the same error.**
> Only their product sets the depth. You cannot separate them by looking at one
> tag, so measure the tag size independently (step 1's card trick) and *then*
> tune `fx`. Tuning `fx` to compensate for a mis-measured tag will silently break
> the moment you use a differently sized tag.

**When it stops being optional: the real RoboCup field.** With four reference
tags, the camera pose is solved from their measured image positions against real
surveyed field coordinates. There is no longer a shared fiction for errors to
cancel into — intrinsic error becomes real position error, and distortion becomes
a position-dependent bias that is worst at the frame edges, which is exactly where
corner reference tags live. Do a proper chessboard calibration then
(`cv2.calibrateCamera`, ~20 views, and keep the distortion coefficients:
`ReferenceTagFieldTransform.from_detections` already accepts them).

## Swapping in the real field

Everything about where the field is lives in `vision-core/src/vision_core/field.py`
— not in this skill, because the ball tracker needs exactly the same answer.

The contract is `CameraFieldTransform`: a rigid transform holding `R` and `t`
such that `p_camera = R @ p_field + t`. Subclasses only decide **how those are
obtained**; every conversion, and the `TagFieldPose` that comes out, is
implemented once in the base class.

- `SyntheticFieldTransform` — today. Invents the pose from a distance and an
  angle. Nothing physical corresponds to it.
- `ReferenceTagFieldTransform` — the RoboCup path. Already written. Give it the
  detections, a `layout` of `tag_id -> (x, y)` in metres, and the camera matrix;
  it runs `solvePnP` and fills in the same `R` and `t`.

To switch, change `build_transform()` in `track.py`:

```python
layout = {0: (0.0, 0.0), 1: (1.2, 0.0), 2: (1.2, 0.8), 3: (0.0, 0.8)}
transform = ReferenceTagFieldTransform.from_detections(dets, layout, K, dist)
```

Downstream — `tag_field_pose`, the overlay, the plan view, the `(id, x, y, theta)`
output — does not change, and `selfcheck` section 3 already exercises exactly this
substitution. `ReferenceTagFieldTransform` also has `.save()` / `.load()` so you
can solve the pose once from a good frame and reuse it for a fixed camera.

## Conventions

**Camera frame** (OpenCV): +X right, +Y down, +Z forward out of the lens. Metres.

**Field frame**: origin at one corner, +X along `width`, +Y along `height`,
+Z along the surface normal. Right-handed. `(0,0)` is the origin corner and
`(width, height)` the far one.

**theta**: rotation about the field normal, degrees in `(-180, 180]`, measured
from field +X, counter-clockwise positive, **0 when the tag is upright**. Use
`--heading-offset` if a robot's tag is not mounted square to the way it drives.

The detector's own tag frame has +X pointing to the tag's visual *left* — a 180°
rotation from what you would guess. That is handled once, in `TAG_VISUAL_RIGHT`
in `src/tag_tracking/pose.py`, and `selfcheck` fails loudly if it ever changes.

The camera view is **not mirrored**. It shows what the camera sees, so the
overlay lands on the phone exactly, and the plan view agrees with the video.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| No detection at all | Screen too dim, or glare. Max brightness, tilt away from windows. |
| Detection flickers | Phone auto-dimming, or motion blur — move slower, add light. |
| Detected but position jumps | Tag too small in frame. Use `--size-mm 100`, or move closer. |
| Distances feel wrong | `fx` and/or tag size. See the calibration section. |
| Wrong camera opens | `uv run track --list-cameras`, then `--camera N`. |
| Camera opens but no frames | Another app has it (Teams, Zoom), or Windows camera privacy settings. |
| Window too big for the screen | `--display-width 1300` |
| Field not visible in frame | `--distance` too small crops it, too large shrinks it. Default 1.8 m fits 1.2 × 0.8. |

## Layout

```
src/tag_tracking/
  pose.py        TagFieldPose, tag_field_pose, TAG_VISUAL_RIGHT -- the tag-shaped bits
  track.py       live tracking, overlay, plan view
  serve_tag.py   serves the tag to the phone at a known physical size
  selfcheck.py   synthetic end-to-end verification, no hardware
```

and, from the shared floor one level up:

```
../vision-core/src/vision_core/
  field.py       the swap point: Field, CameraFieldTransform, Synthetic/ReferenceTag
  camera.py      open_camera, list_cameras, lock_camera
  intrinsics.py  load / save / from_fov / hfov_of
  planview.py    draw_field on the video, PlanView beside it
../calib/        intrinsics.json lands here when you press 's'
```

`pose.py` holds what is *about AprilTags*; `field.py` holds what is *about the
field*. The line matters: when reference tags get mounted, `field.py` changes and
nothing in this folder does.
