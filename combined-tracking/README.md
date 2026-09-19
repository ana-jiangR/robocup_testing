# Combined tracking: robot tag + ball, one camera

Track the robot's AprilTag and the ball at the same time, in one window, from
one camera. Not a third independent skill the way `tag-tracking` and
`ball-tracking` are — an **integration** of both. It depends on both of them
and calls straight into their existing detectors and filters
(`TagTracker`, `BallTracker`, `draw_tag_marks`, `draw_ball_marks`, ...)
rather than reimplementing any of it, so improvements to either tracker show
up here automatically.

**`uv run track` and `uv run track-ball` still work exactly as before,
independently** — this does not touch either of them. Use this one when you
actually need both at once; use the other two when you only need one.

## Run it

You need both a field calibration (or the made-up field, which works fine
too) and a ball color profile first — this doesn't calibrate either one
itself:

```powershell
uv run calibrate-ball --profile test --radius-mm 20   # if you have not already
uv run track-combined --ball-profile test --tag-size 0.080
```

You'll see one window: camera feed on the left with the field overlay, every
detected tag outlined, and the ball circled and cross-marked; one combined
plan view on the right with both the robot (as a colored dot with a heading
and velocity arrow) and the ball (as a magenta dot with a velocity arrow,
ringed when airborne) on the same panel; and a combined readout strip — one
line per tag, then the ball's line, at the bottom.

Keys are the union of both trackers': `q`/Esc quit, `g` grid, `t` trails,
`m` mask overlay, `w` wall/floor (only when the field is not calibrated),
`[`/`]` nudge fx, `r` reset trails and tracks, `s` save intrinsics.

## Flags

Shared flags (`--field`, `--mode`, `--camera`, `--field-pose`, ...) are named
the same as in `track`/`track-ball`. Tag- and ball-specific tuning flags are
prefixed to keep them unambiguous when both are in play at once, since e.g.
each tracker has its own independent `sigma_a`:

| Flag | For |
| --- | --- |
| `--tag-size`, `--heading-offset`, `--tag-sigma-a`, `--tag-sigma-alpha` | the tag tracker |
| `--ball-profile`, `--ball-radius`, `--ball-sigma-a`, `--ball-sigma-a-air`, `--ball-sigma-px` | the ball tracker |
| everything else | shared: field, camera, calibration |

`--mode` defaults to `floor` here (not `wall`, unlike plain `track`) — this
skill exists for the real robot+ball rig, where `floor` is the physically
meaningful geometry.

## Why a separate skill, not a flag on `track` or `track-ball`

Making either existing skill import the other would mean `tag-tracking` and
`ball-tracking` depend on each other, which breaks the one rule this
workspace holds to: skills depend on `vision-core`, never on each other. A
third skill that depends on *both* keeps that rule intact — `tag-tracking`
and `ball-tracking` still don't know about each other, and still install and
run standalone.

## Layout

```
src/combined_tracking/
  track_combined.py   opens the camera once, runs both detectors on every frame
```

Everything else it uses lives in `tag-tracking`, `ball-tracking`, or
`vision-core` — this file is deliberately thin: argument parsing, one shared
camera/frame loop, and drawing functions that combine both trackers' output
onto one readout strip and one plan-view panel, since neither existing
`draw_readout`/`draw_plan` can be called twice without their fixed-size
screen regions colliding.
