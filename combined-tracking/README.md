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
| `--json-out`, `--serve-http`, `--json-log`, `--zmq-pub`, `--no-window` | publishing to another program — see [Feeding another program](#feeding-another-program) |
| `--mjpg` | ask the camera for MJPG, which most USB webcams need to reach 720p at full frame rate — see [Latency](#latency-capture-thread-and---mjpg) |
| everything else | shared: field, camera, calibration |

`--mode` defaults to `floor` here (not `wall`, unlike plain `track`) — this
skill exists for the real robot+ball rig, where `floor` is the physically
meaningful geometry.

## Feeding another program

Four independent, opt-in ways to publish what's being tracked to some other
process — a simulator, a robot's own control loop, anything in any language.
All four publish the exact same per-frame JSON shape (below); use whichever
fits the reader, or several at once. For a reader that wants every frame as
it happens (a simulator mirroring the field in real time), use
[`--zmq-pub`](#a-push-socket----zmq-pub-endpoint), usually with
[`--no-window`](#headless----no-window).

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
`gzip` it, whatever fits), same as any other log file. `*.jsonl`, `*.json.tmp`
and the usual `--json-out` names are gitignored so a session does not end up
in a commit by accident (roughly 10 MB per ten minutes at 30 fps).

### A push socket — `--zmq-pub ENDPOINT`

```powershell
uv sync --all-packages --extra zmq     # once: pyzmq is an optional extra
uv run track-combined --ball-profile test --tag-size 0.080 --no-window --zmq-pub tcp://*:5556
```

The other three all wait to be asked — a reader polls a file or an HTTP
endpoint on its own schedule, and so can skip frames or read the same one
twice. This one **pushes**: every frame, as soon as it is processed, goes
out on a [ZeroMQ](https://zeromq.org/) `PUB` socket bound to `ENDPOINT`. A
subscriber gets each frame once, with no polling loop and no guessing at the
rate.

Each message is **one frame of UTF-8 JSON**, the same doc as below. There is
no topic frame, so subscribe to `""`. A reader that only ever wants the newest
frame (a simulator mirroring the field) should set `CONFLATE` so ZMQ keeps
just the latest for it instead of a queue:

```python
import json, zmq

sock = zmq.Context().socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, 1)          # keep only the newest; set before connect
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.connect("tcp://TRACKER-IP:5556")     # the tracker binds; readers connect
while True:
    doc = json.loads(sock.recv_string())
```

`PUB` never blocks the tracker. A subscriber that falls behind has frames
dropped for it after a few (the send high-water mark is 4, not ZMQ's default
1000, since a queue of stale frames is worse than no frame), and the gap
shows up in `seq`. As with any `PUB`, a subscriber that connects mid-run
starts at whatever frame comes next.

pyzmq is an **optional** dependency — nothing else needs it. Without it,
`--zmq-pub` exits straight away with the install command, before the camera
is opened. `uv sync --all-packages --extra zmq` installs it into the shared
`.venv`. A later plain `uv sync` removes it again, because `uv sync` makes the
venv match exactly what was asked for. For a one-off without syncing, run
`uv run --with pyzmq track-combined ...`.

### Headless — `--no-window`

```powershell
uv run track-combined --ball-profile test --tag-size 0.080 --no-window --zmq-pub tcp://*:5556
```

No window, no overlay, no plan view: **nothing is drawn at all**, and
`cv2.imshow` is never called. Use it when the only consumer is another
program, which then gets the frames that drawing would have cost (on a
laptop, about 30 fps headless against 24 with the window). There are no keys
without a window, so **Ctrl+C quits**, and still shuts down cleanly — camera
unlocked, log closed, sockets released. Ctrl+C works the same with the window
open too.

### Shape

One object per frame, whether it's the single current state from
`--json-out`/`--serve-http`, one line of `--json-log`, or one `--zmq-pub`
message:

```json
{
  "run_id": "20260919-162120", "frame": 1234,
  "seq": 1842,
  "timestamp": 1790437358.3149,
  "t_capture": 1790437358.2909,
  "t_publish": 1790437358.3149,
  "field": { "width": 1.2, "height": 0.8 },
  "tags": [
    {
      "id": 4, "x": 0.612, "y": 0.388, "theta_deg": 19.98,
      "vx": 0.051, "vy": -0.012, "speed": 0.052, "direction_deg": -13.2, "omega_deg": 1.4,
      "inside": true, "visible": true, "age": 0.0, "off_plane_m": 0.004, "age": 0.0, "lost": false
    }
  ],
  "ball": {
    "x": 0.246, "y": 0.121, "z": 0.020,
    "vx": 0.310, "vy": -0.042, "speed": 0.313, "direction_deg": -7.7,
    "grounded": true, "visible": true, "inside": true, "age": 0.0, "lost": false
  },
  "stats": {
    "fps": 29.98, "tag_ms": 14.87, "ball_ms": 8.86, "latency_ms": 23.96,
    "capture_dropped": 4, "ball_dropouts": 57
  }
}
```

All positions are in **metres, field coordinates**, the same convention as
every readout and plan view in this project (see the root README's
Conventions section): origin at one field corner, `+X` along `width`, `+Y`
along `height`. Velocities are metres/second and angles degrees, `-180..180`.
Every key above is always present; only `ball` can be `null`.

**Frame identity and timing.** All times are `time.time()`, Unix epoch
seconds, on the tracker's clock.

| Key | Meaning |
| --- | --- |
| `seq` | Frame counter, `+1` for every frame processed, from 0 at startup. Never repeats or goes backwards within a run. A gap is a frame *this reader* missed (a slow `--zmq-pub` subscriber, an HTTP poll that skipped one), never one the tracker skipped — those are in `stats.capture_dropped`. It restarts at 0 when the tracker restarts. |
| `t_capture` | Taken immediately after the camera read returned the frame. Anything before that — exposure, USB transfer, the driver — is not included, and can't be seen from here. |
| `t_publish` | When this doc was built, after detection and filtering. |
| `timestamp` | Kept for existing readers. Same value as `t_publish`: it has always been the publish time, not the capture time. New readers should use `t_capture` for "when was this true" and `t_publish` for "when was this sent". |

**`tags`** is a list: zero or more entries, one per currently-tracked tag id,
not just one.

| Key | Meaning |
| --- | --- |
| `x`, `y`, `theta_deg` | Filtered position, and heading read off the tag |
| `vx`, `vy`, `speed`, `direction_deg`, `omega_deg` | From the filter. `direction_deg` is direction of *travel*, 0 below 0.05 m/s. `omega_deg` is turn rate in deg/s, CCW positive |
| `inside` | Within the field rectangle |
| `visible` | Detected this frame. `false` while coasting on the last velocity through a short dropout |
| `off_plane_m` | Last measured height off the field plane, a diagnostic. A tag on the field reads within ~1 cm of 0; see [Off-plane readings](#off-plane-readings) |
| `age` | Seconds since last detected; 0 while visible |
| `lost` | `true` once `age` exceeds the tag tracker's coast limit (0.5 s). A lost tag is **dropped from the list** that same frame, so every tag in the list has `lost: false`. The key is there so a reader can handle tags and the ball with the same code. |

**`ball`** is `null` until the ball has been seen for the first time, and
then an object from that point on:

| Key | Meaning |
| --- | --- |
| `x`, `y`, `z` | Filtered position. `z` equals the ball radius when grounded |
| `vx`, `vy`, `speed`, `direction_deg` | In-plane velocity from the filter |
| `grounded` | On the field, so `x`/`y` come from the precise ray/plane intersection. `false` means airborne, and the position then comes from the much weaker range-by-size estimate |
| `visible` | Detected this frame |
| `inside` | Within the field rectangle |
| `age` | Seconds since last detected; 0 while visible. It keeps counting while lost |
| `lost` | `true` once `age` exceeds the ball filter's coast limit (0.35 s) |

A ball goes through three states:

- **visible** (`visible: true`, `lost: false`): measured this frame.
- **coasting** (`visible: false`, `lost: false`, `age` ≤ 0.35 s): a short
  dropout, such as motion blur or a hand passing over. Position is
  extrapolated on the last velocity.
- **lost** (`visible: false`, `lost: true`): unseen for longer than that. The
  tracker **holds the last position**: `x`/`y`/`z` freeze where the coast ran
  out, and `vx`/`vy`/`speed`/`direction_deg` read **0**, so a reader
  extrapolating on them stays put. It stays that way until the ball is seen
  again, which starts a fresh track wherever it turns up. A reader that wants
  "no ball" should treat `lost: true` as absent. `ball` does **not** go back
  to `null`, so "never seen" and "seen, then lost" stay distinguishable, and
  readers that only check for `null` keep working.

> Before `lost` existed, a hidden ball was extrapolated forever. In one
> recorded session it coasted at a constant 0.5 m/s for 13 s and was
> published at `y = -6.4` on a 0.8 m field. If you have older logs, filter
> out `visible: false` frames where `age > 0.35`: they are not positions.

**`stats`** is the tracker's health, measured on the frame this doc describes:

| Key | Meaning |
| --- | --- |
| `fps` | Frames processed per second, smoothed. This is not the camera's rate: when processing is slower than the camera, frames are dropped (next row) |
| `tag_ms` | AprilTag detection, field pose and tag filter, this frame |
| `ball_ms` | Ball mask, blob detection and ball filter, this frame |
| `latency_ms` | `t_publish − t_capture`: how old the frame was by the time this doc went out |
| `capture_dropped` | Cumulative. Frames the camera delivered that were overwritten by a newer one before the tracker got to them. This means the tracker is slower than the camera. It is not a detection failure |
| `ball_dropouts` | Cumulative. Frames processed while a ball track existed but the ball wasn't detected (coasting or lost). This is a detection failure, and is counted separately from the above |

### Latency: capture thread and `--mjpg`

The camera is read on its own thread, which keeps **only the newest frame**.
In a single read-then-process loop, frames that arrive while the previous one
is being processed queue up in the driver. Each read then returns an older
frame than the one the camera just took, and the lag grows to the depth of
that queue. Here a frame the tracker was too slow to take is overwritten and
counted in `stats.capture_dropped`, so the tracker always works on the most
recent picture. The filters are given the real time between the frames they
did process, so a dropped frame does not distort velocity.

The driver is also asked to queue at most one frame (`CAP_PROP_BUFFERSIZE=1`).
Many Windows webcams ignore this, and the startup lines say whether it was
honoured. The capture thread drains the queue either way.

`--mjpg` asks the camera for MJPG instead of its default uncompressed format.
Many USB webcams can't send 1280×720 uncompressed at 30 fps and quietly drop
to 10–15 fps; MJPG fixes that. Some drivers don't report the format they
ended up in, so check the `fps` stat to see whether it helped.

### Off-plane readings

`off_plane_m` is a tag's measured distance from the field plane. **Positive
is towards the camera**, so a tag lifted off the field reads positive. A tag
lying on the field reads within about a centimetre of zero. If a tag's median
`|off_plane_m|` over 60 sightings exceeds 5 cm, the tracker prints a warning
once for that tag.

A large **negative** value is not a tag held in the air. It means the tag
looks *further away* than the plane it is lying on, which is physically
impossible, so the plane or the tag's range is wrong. The usual cause is
running without `calib/field_pose.json`, on the made-up field: camera
`--cam-height 1.5` m straight down, 60° HFOV unless `intrinsics.json` exists.
In that case tag and ball positions are measured on **different scales**:

- A tag's `x`/`y` come from its own 3D pose, which is ranged from the tag's
  printed size. The assumed height and fx cancel out, so its sideways
  offsets are right in metres, as long as `--tag-size` matches the printed
  tag.
- The ball's `x`/`y` come from where its view ray meets the *assumed* plane.
  If the real floor is further away than 1.5 m, every ball position is pulled
  towards the point under the camera by `1.5 / real distance`.

In one recorded session, tags read `-0.66` m (tag 0) and `-1.4` m (tag 4).
If those tags were lying flat, the ball's offsets from the field centre came
out about 0.5–0.7× their true size: 20–30 cm of disagreement with the tags
at the field edge. The fix is `uv run calibrate-field` (and
`calibrate-camera`), not a code change.

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
camera/frame loop fed by a newest-frame-only capture thread
(`LatestFrameGrabber`), drawing functions that combine both trackers' output
onto one readout strip and one plan-view panel (since neither existing
`draw_readout`/`draw_plan` can be called twice without their fixed-size
screen regions colliding), and the four `--json-out`/`--serve-http`/
`--json-log`/`--zmq-pub` publishers (`build_state_doc` builds the one shared
dict each of the four then does something different with — write it
atomically, serve it over HTTP, append it to a log, or push it on a socket).
