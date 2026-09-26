# robocup-testing

A collection of robotics capabilities — **one folder per skill** — built as a
RoboCup proof of concept. Everything reports positions in **metres, in field
coordinates**, from a single webcam.

## The skills

| Skill | What it tracks | Output |
| --- | --- | --- |
| [`tag-tracking/`](tag-tracking/) | AprilTags on a phone or a robot shell | `TagFieldState` — position, heading, velocity, turn rate |
| [`ball-tracking/`](ball-tracking/) | A colored ball, by its hue rather than a marker | `BallFieldState` — position, height, velocity, grounded |

Both report **filtered** motion, not frame-to-frame differences. Each keeps its
own physics and shares one small Kalman filter core in `vision-core`.

And an **integration**, not a third independent skill — it depends on both of
the above instead of only on `vision-core`, and reuses their detectors and
filters rather than reimplementing anything:

| | What it tracks | Output |
| --- | --- | --- |
| [`combined-tracking/`](combined-tracking/) | The robot's tag *and* the ball, one camera, one window | both of the above, together |

And the floor a skill stands on, which is **not** itself a skill:

| | |
| --- | --- |
| [`vision-core/`](vision-core/) | Field geometry, the camera↔field transform, intrinsics, camera control, drawing |
| [`calib/`](calib/) | `intrinsics.json`, `field_pose.json`, `ball_color.json` — measured calibration, shared by every skill |

## Quick start

```powershell
uv sync                        # one lockfile, one .venv, whole workspace

uv run selfcheck               # prove the geometry, no camera needed
uv run serve-tag               # put a sized AprilTag on your phone
uv run track --tag-size 0.080  # track it

uv run track --synthetic-camera   # or: watch the whole pipeline work with no hardware at all
```

For the ball, the color profile is the one step with no default — nothing can
find a ball it has never been shown:

```powershell
uv run calibrate-ball --profile mine --radius-mm 40   # drag a box over YOUR ball, press s
uv run track-ball --ball-profile mine --cam-height 1.5
```

Need both the robot's tag and the ball at once, from the same camera?

```powershell
uv run track-combined --ball-profile mine --tag-size 0.080
```

Turning the made-up field into a real, measured one is a one-time calibration
step — `uv run calibrate-camera` and `uv run calibrate-field`, or
`uv run track --calibrate-live` to do both in one session. See
[`tag-tracking`'s README](tag-tracking/#one-time-calibration-making-it-real).

Every command works from the repo root. Each skill also has its own
`pyproject.toml`, entry points and README, so it still reads — and installs —
standalone.

## Command reference

**No hardware needed:**

| Command | What it does |
| --- | --- |
| `uv run selfcheck` | Proves the field math is correct |
| `uv run calibrate-camera --synthetic` | Proves lens-calibration math is correct |
| `uv run calibrate-field --synthetic` | Proves field-calibration math is correct |
| `uv run track --synthetic-camera` | Watch the whole pipeline work on a fake field |

**Prep — phone or printer:**

| Command | What it does |
| --- | --- |
| `uv run serve-tag` | One AprilTag on your phone, sized via the bank-card trick |
| `uv run serve-tag --field-sheet out.png` | All 4 reference tags laid out on one printable page |
| `uv run calibrate-camera --save-board-png board.png` | The ChArUco board to print, for the lens step |

**Calibration — real webcam:**

| Command | What it does | Writes |
| --- | --- | --- |
| `uv run calibrate-camera` | Measures the lens: wave a printed board at the webcam | `calib/intrinsics.json` |
| `uv run calibrate-field` | Measures the field position: point the webcam at your 4 tags | `calib/field_pose.json` |
| `uv run calibrate-field --sequential` | Same, but one tag moved to each corner in turn (camera fixed) instead of 4 at once | `calib/field_pose.json` |
| `uv run calibrate-ball` | Measures the ball's color: drag a box over it | `calib/ball_color.json` |
| `uv run track --calibrate-live` | Lens + field in one session, then starts tracking | `calib/field_pose.json` |
| `uv run track --calibrate-live --sequential` | Same, one tag moved to each corner instead of 4 at once | `calib/field_pose.json` |

All three are optional to *start*. Without them the software assumes a 60° lens
and believes whatever `--cam-height` you type. Positions still move correctly;
they just aren't measured. The ball's color profile is the exception — there is
no default ball, so `calibrate-ball` is required before `track-ball` runs.

**Tracking:**

| Command | What it does |
| --- | --- |
| `uv run track` | Tags — auto-loads whatever's already calibrated |
| `uv run track-ball --ball-profile NAME` | Ball — same, and needs a color profile |
| `uv run track-combined --ball-profile NAME` | Both, one camera, one window — see [`combined-tracking/`](combined-tracking/) |
| `uv run unlock-camera` | Repair: a run that crashed with the camera locked leaves every app with a dark green picture. The tools also do this on start and exit. |

One webcam serves one program at a time — `track` and `track-ball` cannot
both hold it open at once, so run those two in separate sessions, not
together. `track-combined` is the exception on purpose: it opens the camera
once and runs both detectors against the same frame, for exactly this case.

### Getting live positions into another program

`track-combined` can publish what it's tracking to another process — a
simulator, another repo, anything — three independent ways. All off unless
you ask; use one, or several at once. Full JSON shape and details in
[`combined-tracking`'s README](combined-tracking/#feeding-another-program).

| Command | What you get |
| --- | --- |
| `uv run track-combined --ball-profile NAME --json-out state.json` | One file, **overwritten** every frame — always "the current state" |
| `uv run track-combined --ball-profile NAME --serve-http 8000` | The same, over `GET http://localhost:8000/state` — works from another machine or language, no filesystem access needed |
| `uv run track-combined --ball-profile NAME --json-log session.jsonl` | One JSON line **appended** per frame — a full history, not just "right now". `*.jsonl` and the `--json-out` files are gitignored: they are run output, not source |

### Flags that matter

`--field` and `--tag-size` are what make the numbers mean real metres —
always pass your **ruler-measured** sizes, not the target you printed for.
Both default to a plausible demo size, so a bare command still runs, just
against the wrong physical size until you correct it.

| Flag | Meaning | Default | Used by |
| --- | --- | --- | --- |
| `--field W H` | Real field size, in metres | `1.2 0.8` | `calibrate-field`, `track` (incl. `--calibrate-live`, `--synthetic-camera`) |
| `--tag-size` | Real black-square tag size, in metres | `0.080` | `track` (incl. `--calibrate-live`, `--synthetic-camera`) |
| `--camera N` | Which webcam index to use | `0` | `calibrate-camera`, `calibrate-field`, `track` |
| `--out PATH` | Where to save the calibration result | `calib/...` at repo root | `calibrate-camera`, `calibrate-field` |
| `--field-pose PATH` | Field calibration file to load/save | `calib/field_pose.json` | `track` (incl. `--calibrate-live`) |
| `--layout ID:X,Y ...` | Custom reference-tag positions, overrides the default 4 corners | corners of `--field` | `calibrate-field` |
| `--sequential` | One tag moved to each corner in turn (camera fixed), instead of 4 tags at once | off | `calibrate-field`, `track --calibrate-live` |
| `--frames-per-corner` | Samples to average at each corner | `15` | `calibrate-field --sequential`, `track --calibrate-live --sequential` |
| `--tag-id` / `--size-mm` | Which tag id, and initial size in mm on the phone page | `0` / `80` | `serve-tag` |
| `--ball-profile` | Which named color profile to track | `test` | `track-ball`, `track-combined` |
| `--radius-mm` | The ball's real **radius**, stored in the profile | `20` | `calibrate-ball` |
| `--s-spread` / `--h-spread` | How far the color profile reaches beyond what was sampled, in histogram bins — widen `--s-spread` if the ball vanishes in shade or under another lamp | `4` / `1` | `calibrate-ball` |
| `--cam-height` | Camera height above the field, metres — ignored once `field_pose.json` exists | `1.5` | `track`, `track-ball`, `track-combined` |
| `--print-poses` / `--print-states` | Stream the numbers to the terminal | off | `track` / `track-ball`, both in `track-combined` |
| `--json-out PATH` | Write current detections to a file, overwritten every frame | off | `track-combined` |
| `--serve-http PORT` | Serve current detections over plain HTTP `GET` | off | `track-combined` |
| `--json-log PATH` | Append one JSON line per frame — a history, not just current state | off | `track-combined` |

Full per-command flag lists: `uv run <command> --help`, or see
[`tag-tracking`'s README](tag-tracking/) for the calibration workflow in detail.

### Keys, in every window

`q` or `Esc` quits, `s` saves. The rest are per-tool and listed in each README.
Keys only reach the **video window**, so click it first — and if letters do
nothing while the mouse still works, your input method is swallowing them.
Switch it to English, or press `Esc`, which no IME intercepts.

## Not using uv?

It works with plain pip, with three catches. **Installing uv is still the easier
answer** — it is a single binary, needs no Python of its own, and does not touch
an existing setup:

```powershell
winget install --id=astral-sh.uv        # or: pip install uv
```

If you would rather use pip:

```powershell
py -3.12 -m venv .venv                  # 3.12 or 3.13, not 3.14 — see below
.venv\Scripts\activate
pip install -e ./vision-core            # THIS ORDER MATTERS
pip install -e ./tag-tracking
pip install -e ./ball-tracking
pip install -e ./combined-tracking      # needs the two above already installed
```

1. **Python 3.12 or 3.13 — not 3.14.** `pupil-apriltags` ships prebuilt wheels
   for cp310-cp313 only. On 3.14 pip compiles it from source and the result is
   missing `apriltag.dll`, so the tag tracker dies at import. `requires-python`
   now caps this, so pip refuses up front instead of building something broken.
   `.python-version` pins 3.12; uv obeys it automatically, pip does not.
2. **Install `vision-core` first.** The skills depend on it through
   `[tool.uv.sources] workspace = true`, which only uv understands. pip goes
   looking on PyPI instead, finds nothing, and stops with
   `No matching distribution found for vision-core`. (A clean failure, at least —
   there is no package of that name to install by accident.)
3. **`pip install .` at the repo root does nothing useful.** The root is a uv
   workspace definition, not a package. Install the members directly.

`uv.lock` is committed and pins exact versions for uv users; pip users resolve
fresh and may get slightly different ones.

## Adding a skill

1. `mkdir new-skill/src/new_skill`, give it a `pyproject.toml` depending on
   `vision-core = { workspace = true }`.
2. Add it to `members` **and** `dependencies` in the root `pyproject.toml`, so
   `uv run <its-command>` works from the root.
3. Write a README that reads standalone.
4. Anything camera- or field-shaped you find yourself writing belongs in
   `vision-core`, not in the skill.

## Conventions

**Camera frame** (OpenCV): +X right, +Y down, +Z forward out of the lens. Metres.

**Field frame**: origin at one corner, +X along `width`, +Y along `height`,
+Z along the surface normal. Right-handed. `(0,0)` is the origin corner and
`(width, height)` the far one.

**`--mode wall|floor`**: whether the virtual field stands up facing the camera or
lies flat. `floor` is the physically meaningful one and assumes the camera is
**hung over the field centre looking straight down**, 1.5 m up — set
`--cam-height` to your real mounting height. `wall` is easier to demo by hand.
The maths works at any camera pose; both modes are covered by `selfcheck`.


