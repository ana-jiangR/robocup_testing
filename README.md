# simulatedRobotApplication

A collection of robotics capabilities — **one folder per skill** — built as a
RoboCup proof of concept. Everything reports positions in **metres, in field
coordinates**, from a single webcam.

## The skills

| Skill | What it tracks | Output |
| --- | --- | --- |
| [`tag-tracking/`](tag-tracking/) | AprilTags on a phone or a robot shell | `TagFieldPose` — `x, y, theta_deg` |

Color-based ball tracking is the next skill, and is not in this branch yet.

And the floor a skill stands on, which is **not** itself a skill:

| | |
| --- | --- |
| [`vision-core/`](vision-core/) | Field geometry, the camera↔field transform, intrinsics, camera control, drawing |
| [`calib/`](calib/) | `intrinsics.json` / `field_pose.json` — the **camera's** and **field's** calibration, shared by every skill |

## Quick start

```powershell
uv sync                        # one lockfile, one .venv, whole workspace

uv run selfcheck               # prove the geometry, no camera needed
uv run serve-tag               # put a sized AprilTag on your phone
uv run track --tag-size 0.080  # track it

uv run track --synthetic-camera   # or: watch the whole pipeline work with no hardware at all
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

**Calibration — real webcam:**

| Command | What it does |
| --- | --- |
| `uv run calibrate-camera` | Measures the lens: wave a printed board at the webcam |
| `uv run calibrate-field` | Measures the field position: point the webcam at your 4 tags |
| `uv run track --calibrate-live` | Does both of the above, then starts tracking right away |

**Tracking:**

| Command | What it does |
| --- | --- |
| `uv run track` | Normal run — auto-loads whatever's already calibrated |

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
| `--tag-id` / `--size-mm` | Which tag id, and initial size in mm on the phone page | `0` / `80` | `serve-tag` |
| `--print-poses` | Also stream `id x y theta` to the terminal | off | `track` |

Full per-command flag lists: `uv run <command> --help`, or see
[`tag-tracking`'s README](tag-tracking/) for the calibration workflow in detail.

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


