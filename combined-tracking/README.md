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

## Feeding another program

Three independent, opt-in ways to publish what's being tracked to some other
process — a simulator, a robot's own control loop, anything in any language.
All three publish the exact same per-frame JSON shape (below); use whichever
fits the reader, or several at once.

### A local file — `--json-out`

```powershell
uv run track-combined --ball-profile test --tag-size 0.080 --json-out live_state.json
```

Writes the current frame's detections to that path, **overwritten every
frame** — a reader just re-reads the file whenever it wants the latest state.
Needs the reader to have filesystem access to this machine (same machine, or
a shared/mounted drive).

The write is **atomic**: each frame is built in memory, written to a sibling
`<path>.tmp`, then moved on top of the real path in one filesystem operation
(`os.replace`, atomic on both Windows and POSIX). A reader polling the file
on its own, unsynchronised schedule can never open it mid-write and see
truncated or half-updated JSON — it always sees either the previous complete
frame or the new complete one, never something in between.

### A tiny HTTP server — `--serve-http PORT`

```powershell
uv run track-combined --ball-profile test --tag-size 0.080 --serve-http 8000
```

Prints the URL to use (`http://localhost:8000/state`, and the LAN address for
a reader on a *different* machine). A reader does a plain `GET` whenever it
wants the latest state — works from any language with an HTTP client, and
needs no filesystem access to this machine at all, just network reachability.
`GET /` works too, same response as `/state`. CORS is wide open
(`Access-Control-Allow-Origin: *`), so a browser-based simulator can
`fetch()` it directly with no proxy.

Runs in a background thread inside the same process — starts when the window
does, stops cleanly when you quit. The state it serves is whatever the most
recent frame published; there is no history, no queue, just "ask and get the
latest."

### A growing log — `--json-log PATH`

```powershell
uv run track-combined --ball-profile test --tag-size 0.080 --json-log session.jsonl
```

The other two are both "current state, overwritten" — this one is history
instead: **appends** one JSON object per frame as its own line (a
[JSON Lines](https://jsonlines.org/) file, so a reader `readline()`s or
`tail -f`s it and gets one complete record per line, never a partial one).
Use this when the reader wants to replay or analyse a whole session rather
than only ever ask "where is it right now" — or when it isn't running at the
same time as the tracker at all, and reads the file afterward.

Each line is flushed to disk immediately, not buffered — a `tail -f` sees a
new line the moment that frame was processed, no lag waiting for a buffer to
fill. The file only grows; nothing here rotates or truncates it, so a long
session can produce a large file — that's on you to manage (delete it,
`gzip` it, whatever fits), same as any other log file.

Shape (one object, whether it's the single current state from `--json-out`/
`--serve-http`, or one line of `--json-log`):

```json
{
  "run_id": "20260919-162120", "frame": 1234,
  "timestamp": 1730000000.123,
  "field": { "width": 1.2, "height": 0.8 },
  "tags": [
    {
      "id": 5, "x": 0.600, "y": 0.400, "theta_deg": 19.98,
      "vx": 0.0, "vy": 0.0, "speed": 0.0, "direction_deg": 0.0, "omega_deg": 0.0,
      "inside": true, "visible": true, "age": 0.0, "off_plane_m": 0.004
    }
  ],
  "ball": {
    "x": 0.246, "y": -0.121, "z": 0.247,
    "vx": 0.0, "vy": 0.0, "speed": 0.0, "direction_deg": 0.0,
    "grounded": false, "visible": true, "inside": false, "age": 0.0
  }
}
```

`tags` is a list (zero or more — every currently-tracked tag id, not just
one). A tag the detector misses keeps its entry, with `visible: false` and a
growing `age`, for `--tag-coast` seconds (default 0.5) before it is dropped;
raise that if your reader would rather see a predicted position than a gap. `ball` is `null` when no ball has ever been seen yet, otherwise always
present (with `visible: false` while the filter is coasting through a
dropout, same meaning as everywhere else in this project). All positions are
in **metres, field coordinates** — same convention as every readout and
plan-view in this project (see the root README's Conventions section):
origin at one field corner, `+X` along `width`, `+Y` along `height`. `age` is
seconds since the ball was last actually seen (0 while it's currently
visible). `timestamp` is `time.time()` (Unix epoch seconds) at the moment
that frame was captured, in case the reader wants to compute its own latency
or discard a stale file. `run_id` is the same for every frame of one launch
(so an appended `--json-log` splits cleanly into runs) and `frame` counts
camera frames from 1 within it.

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
camera/frame loop, drawing functions that combine both trackers' output onto
one readout strip and one plan-view panel (since neither existing
`draw_readout`/`draw_plan` can be called twice without their fixed-size
screen regions colliding), and the three `--json-out`/`--serve-http`/
`--json-log` publishers (`build_state_doc` builds the one shared dict each of
the three then does something different with — write it atomically,
serve it over HTTP, or append it to a log).
