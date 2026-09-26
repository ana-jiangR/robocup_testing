# ball-tracking

Track a colored ball and report where it is, and **where it is going**, in field
coordinates — on the same field the [tag tracker](../tag-tracking/) uses.

**Skill 2.** The field geometry it stands on lives in
[`vision-core`](../vision-core/); everything below runs from the repo root.

## A ball is not a tag

A tag gives you four corners, so one detection yields a full pose — position
*and* orientation. A ball gives you **one point and one scalar**: a centroid and
an apparent radius, with no orientation at all.

So `direction_deg` here is not a heading. It is the direction of **travel**,
estimated by the filter, and a stationary ball correctly has none.

## Run it

### 1 · Settle the color

```powershell
uv run calibrate-ball                                  # profile 'test'
uv run calibrate-ball --profile mine --radius-mm 40    # YOUR ball's real radius
uv run calibrate-ball --profile mine --s-spread 6      # ball lives under changing light: reach further
uv run calibrate-ball --list-profiles
```

Drag a box over the ball; the right panel shows the mask live. Press `h` for a
hue histogram of the whole scene — pick the color **empirically against your
actual room**, not from a table. Then **roll the ball into shade and under
every lamp it will meet** before saving: the mask should fade, not vanish. If
it vanishes, press `>` to widen the saturation spread, `'` for hue (or lower the
saturation floor with `[`) until it survives. Press `s` to save.

Keys: drag = sample · `a` add another drag to the same sample · `h` scene hue
histogram · `-` `+` back-projection threshold · `[` `]` saturation floor ·
`<` `>` saturation spread · `;` `'` hue spread · `s` save · `q`/`Esc` quit.

**Sample under the light the ball will actually play in.** A profile is a
measurement of the ball *through this camera under this light*; the spread
(below) buys tolerance for the ball moving between lamps and shadows, not for
calibrating in a different room. Press `a` over the ball in its darkest and
brightest spots to fold those into the same sample, which is cheaper than any
amount of spread.

A **multi-color ball works**, because a profile is a whole histogram rather than
one hue: drag over one panel, then press `a` over each of the others to add them.
White and black panels are the exception — they carry no hue and are dropped by
design, so the mask will show holes there. That is fine as long as colored
patches still reach the ball's outline all the way round, since the outline is
what the radius is measured from.

Profiles are named, stored in `calib/ball_color.json`, and selected with
`--ball-profile`. Each carries the ball's **radius**, because every geometric
step downstream needs it — selecting a profile selects the whole ball.

### 2 · Track it

```powershell
uv run track-ball --ball-profile mine                 # floor mode, camera overhead: the default
uv run track-ball --ball-profile mine --cam-height 1.8  # match YOUR mounting height
uv run track-ball --mode wall            # free test harness for the airborne path
uv run track-ball --print-states         # stream x/y/z/vx/vy/grounded to stdout
uv run track-ball --synthetic-field      # ignore a saved calib/field_pose.json
uv run track-ball --list-cameras         # if it grabs the wrong one
```

Once the window is open, in this order:

1. **Press `m`** for the mask. One solid disc on the ball and nothing else is
   what you want. This view is the verdict on your color profile.
2. **Roll the ball.** The plan-view dot follows it and the speed and direction
   in the bottom strip change. The trail should be smooth, not jagged.
3. **Lift it 20 cm** and the state flips to airborne. Put it down, it flips back.
4. **Cover it** with your hand: the marker greys out and coasts, then recovers.
   Keep it covered past about a third of a second and the readout says `LOST`
   and the marker stops where it is, instead of drifting off on its last
   velocity.

If `calib/field_pose.json` exists (from `uv run calibrate-field` in tag-tracking,
with four reference tags on the field corners), it is loaded automatically and
`--cam-height`/`--pitch` are ignored: the camera's position is then measured,
not assumed. Until then the made-up rig below applies.

Keys: `q` quit · `g` grid · `t` trails · `m` mask · `w` wall/floor · `p` pause ·
`[` `]` nudge fx · `r` reset trail and track · `s` save intrinsics.

## The assumed rig

Camera **over the middle of the field, looking straight down**, 1.5 m up —
`--cam-height 1.5 --pitch 90`, the default.

⚠️ Nothing measures this; you are *telling* the software where the camera is. If
yours is really at 1.8 m and you leave the default, every position is
confidently wrong with nothing on screen to flag it. Measure once, pass it.

Below about **1.3 m** a 1.2 × 0.8 m field no longer fits in a 60° view. Lower is
otherwise better — the ball is bigger, so the airborne check gets more sensitive
(~20 cm of ball height at 1.4 m, 23 cm at 1.5 m, 33 cm at 1.8 m).

All of this disappears once four reference tags go up and the pose is *measured*.

## Buy a matte ball

Lean magenta/pink (OpenCV H ≈ 150–172): skin, wood, carpet and cardboard all
cluster at H 0–30. **Matte matters more than the hue** — a specular highlight is
a white blob that slides across the ball as it rolls, punching a moving hole in
the mask and destabilising the apparent radius, which is what the height estimate
depends on.

## How it works

**1 · Lock the camera.** Auto-exposure and auto-WB off. The single biggest
robustness win available: a camera left on auto re-exposes whenever something
bright crosses frame, and the ball's color drifts out from under the profile.
Both properties are advisory, so `lock_camera` reads them back and reports what
actually stuck, and checks each for damage: going manual on exposure jumps to
a stale value the driver remembers, so the lock measures brightness first and
walks the manual value until it matches; manual white balance on some cameras
is a fixed temperature that tints the picture green, so color balance is
measured either side and the lock rolled back if it moved. Whatever can't be
held safely is left on auto and reported. The lock lives in the driver and
outlives the process, so both tools unlock on exit; if a run dies before it
can, `uv run unlock-camera` puts the camera back.

**2 · Color mask.** `cv2.calcBackProject` over a measured H/S histogram, not a
hard `inRange` — `inRange` is a cliff where one degree of hue drift makes the
ball vanish, while back-projection lets it fade. Brightness is deliberately
excluded and applied as a wide-open gate: a ball rolling into shade should not
stop existing.

The measured histogram is a tight island, though: a handful of H/S bins, and
zero everywhere else. That is its own cliff. Shade and a bluish ambient pull
saturation *down*, a warmer bulb pushes it *up*, a re-balanced camera nudges
the hue — and a few bins of drift lands the same ball on zero likelihood. So
the profile is **spread** before use (`BallColor.working_hist`): a Gaussian
blur across the histogram, wide along saturation (`--s-spread`, 4 bins ≈ 32
levels) and narrow along hue (`--h-spread`, 1 bin = 6°), with hue wrapped so
red does not stop at the end of the axis, then re-normalised so `threshold`
means the same thing. The raw measurement is what gets saved; the spread is
applied on load, so old profiles benefit without re-sampling and a profile
can be re-tuned without re-measuring the ball. Hue barely moves with light,
saturation is what light changes, hence the asymmetry.

**3 · Blob.** Morphology OPEN then CLOSE, then contours scored on area,
circularity, fill ratio, and distance from the filter's prediction. Scored, not
filtered — a chain of hard rejects can discard the only candidate in a bad frame.
The centre is the min-enclosing-circle centre, not the moment centroid, which
occlusion would drag sideways.

**4 · Pixel → field metres.** Ray/plane intersection using the shared transform:

```python
d_cam = inv(K) @ [u, v, 1.0]             # z == 1, so s == depth
a = transform.direction_to_field(d_cam)[0]
b = transform.camera_origin_in_field()
s = (BALL_RADIUS - b[2]) / a[2]          # plane at z = radius, NOT z = 0
x, y = (b + s * a)[:2]
```

The plane sits at `z = radius` because the image gives you the ball's *centre*,
one radius above the surface. Intersecting at `z = 0` overshoots by
`radius / tan(elevation of the view ray)` — measured, for a 40 mm ball:

| camera | centre | edge | far corner |
| --- | --- | --- | --- |
| **overhead, 1.5 m** (default) | 0.0 mm | 8.1 mm | 9.7 mm |
| tilted 35°, 1.0 m up | 29 mm | 29 mm | 29 mm |

Small, but systematic and in a fixed direction, so unlike noise it never averages
out. `cv2.undistortPoints` is already wired in, a no-op until real `dist_coeffs`
exist.

**5 · Airborne detection, free.** Cross-check `Z ≈ fx · R / radius_px` against
the ray/plane depth. Worth it because the silent failure is bad: an airborne ball
still yields a plausible ground position — the ray just carries on to the floor.
In the synthetic check, ignoring it put the ball up to **45 cm** out.

The decision is a Schmitt trigger, a high bar to enter and 40% of it to leave. A
single threshold either misses a ball genuinely 20 cm up, or flaps frame to frame
— which is worse than either answer, because it keeps swapping the measurement
model out from under the filter.

**6 · Velocity.** A 6-state Kalman filter `[x, y, z, vx, vy, vz]`, built on
`vision_core.kalman` (the same core the tag tracker
uses; the ball physics lives here). Never difference raw positions: one pixel of
jitter at 2 m is ~0.1 m/s of pure noise, as fast as the ball rolls. Measured —
raw differencing σ = 0.095 m/s, filtered σ = 0.049 m/s, mean unbiased to 2 mm/s.

| | measurement | noise | out-of-plane |
| --- | --- | --- | --- |
| **grounded** | ray/plane `(x, y)` | small — real geometry against a known plane | `z` pinned to radius, `vz` to 0 |
| **airborne** | radius-based 3D point | large, plus larger process noise (`--sigma-a-air`) | free — and deliberately **no gravity** |

No gravity term, on purpose. A parabola is right for a ball in flight and wrong
for a ball in a hand — and the hand is how you test this. With `g` in the
prediction, a held ball's range is known to a centimetre, so the gate rejected
the "it didn't fall" measurement within two frames, the state fell away at
9.8 m/s² unopposed, and the track timed out and restarted in a loop. The ranger
is also too weak to see a parabola over the few frames a bounce lasts, so the
model bought nothing. Airborne acceleration is process noise instead, sized to
cover gravity.

Also buys coasting through motion-blur dropouts, a Mahalanobis gate against false
positives, and range-dependent noise `R ∝ (σ_px · Z / fx)²`. Tune `--sigma-a`
(2–4 m/s² for carpet) and `--sigma-a-air` (10 m/s²: gravity, a bounce, a hand). `dt` is **measured** from `perf_counter` right after
`cap.read()` — feeding it a nominal 1/30 against an actual 24 fps mis-scales every
velocity, and the error looks like drift rather than a bug.

> **If you touch the filter**, two non-obvious things, both found by testing:
> the airborne covariance is a cigar down the view ray, not a field-axis diagonal
> (`ray_covariance`); and every regime change must re-inflate the *position*
> covariance, not just `z`. Get either wrong and the gate rejects correct
> measurements while the filter coasts on a confident, stale estimate. The
> docstrings in [`ball.py`](src/ball_tracking/ball.py) explain why.

## Honest limits

**Radius-based height is weak** — for a 40 mm ball on 720p at 60° HFOV, about
**±15 cm at 1.5 m** (the default mounting height) and **±27 cm at 2 m**. The
error grows as `Z²`, because one pixel of radius error is worth `Z/r` metres and
`r` itself shrinks with distance. Enough to say *"airborne, stop trusting the
ground position"*. **Not** enough to predict a landing point; that needs a second
camera.

- `BallFieldState` exposes `vx`/`vy` but not `vz`, by design — in-plane velocity
  is the part you can trust. `vz` is in the filter state if you need it.
- One ball only. Two of the same color will fight.
- `--mode wall` is not physical for a rolling ball, but a hand-held ball is
  off-plane nearly always, so the airborne path runs constantly at your desk.

## Output contract

```python
@dataclass(frozen=True)
class BallFieldState:
    x, y, z: float            # metres, field coords; z == radius when grounded
    vx, vy: float             # metres/second
    speed: float              # in-plane, m/s
    direction_deg: float      # direction of TRAVEL, -180..180; 0 when stationary
    grounded: bool            # x/y are the precise ray/plane answer
    visible: bool             # False means the filter is coasting
    age: float                # seconds since actually seen
    lost: bool                # unseen past max_coast_s (0.35 s): position held, velocity 0
```

Once `lost`, the position stops being extrapolated. It holds where the coast
ran out, with `vx`/`vy`/`speed` at 0, until the ball is seen again, and that
sighting starts a fresh track. Before this, a hidden ball coasted on its last
velocity for as long as it stayed hidden, which is metres off the field
within seconds.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Ball vanishes in shade, or under a different lamp | Saturation drifted out of the profile. Widen `--s-spread` (`>` in `calibrate-ball`), lower the saturation floor with `[`, or re-sample with `a` in the dark spot too. If it only happens after the *camera* moved rooms, re-run `calibrate-ball` there; no spread covers a different white balance. |
| Mask has holes in the ball | White/black panels carry no hue. Lower the saturation floor with `[`, or accept it if colored patches still reach the outline. |
| Mask lights up the room too | The background shares the ball's hue. Re-sample a tighter box, raise the threshold with `+`, or use a ball in an emptier hue — press `h` to see which are free. |
| Ball found, but positions wrong | `--cam-height` doesn't match reality, or the lens is uncalibrated. Run `calibrate-field`, or measure and pass the height. |
| Height/airborne is noisy | Expected — see Honest limits. It answers *airborne yes/no*, not *how high*. |
| Never flips to airborne | Lift it higher, or lower the camera. The check gets more sensitive the closer the camera is. |
| Keys do nothing, mouse works | Click the video window; if letters are still ignored, switch your input method to English, or use `Esc`. |
| Picture is dark, or green, in every app | A run crashed with the camera locked, and the driver kept the manual exposure. `uv run unlock-camera`, or just start either tool — `open_camera` puts it back on auto first. |
| `camera lock:` says `REVERTED` | That half of the lock damaged the picture (manual exposure couldn't reach auto's brightness, or manual WB tinted it green), so it was left on auto. Tracking still works; the hue may drift a little if the lighting changes. |
| Mask fine at first, then drifts over minutes | The camera is on auto (`--no-lock`, or the lock `REVERTED`) and re-exposing as things move through frame. Fix the lock, or widen the spread to absorb it. |
| `calibrate-ball`/`track-ball` not found | `ball-tracking` isn't in the workspace. Check the root `pyproject.toml`, then `uv sync`. |

## Layout

```
src/ball_tracking/
  ball_color.py   named profiles, the mask, and the calibration tool
  ball.py         detector, ray/plane geometry, the ball's Kalman filter
  track_ball.py   live tracking, mask overlay, plan view
```

The filter's matrix algebra is `vision_core.kalman`, shared with the tag
tracker; what lives here is the ball physics — switching between the grounded
and airborne measurement models, sized so the switch itself doesn't get gated out.
