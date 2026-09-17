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
| [`calib/`](calib/) | `intrinsics.json` — the **camera's** calibration, shared by every skill |

## Quick start

```powershell
uv sync                        # one lockfile, one .venv, whole workspace

uv run selfcheck               # prove the geometry, no camera needed
uv run serve-tag               # put a sized AprilTag on your phone
uv run track --tag-size 0.080  # track it
```

Every command works from the repo root. Each skill also has its own
`pyproject.toml`, entry points and README, so it still reads — and installs —
standalone.

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

## Why a uv workspace, and not sibling folders

Because of one promise, made in `vision-core/src/vision_core/field.py`:

> **This file is the swap point.** Moving from a field pose we *invented* to one
> *measured* from four reference AprilTags costs one subclass and changes nothing
> downstream.

`selfcheck` section 3 exercises exactly that substitution and prints the same tag
under both transforms.

The next skill — ball tracking — needs the identical transform. It has no
interest in AprilTags, but it very much needs the field to be *located* before it
can say where a ball is on it. Two independent folders would mean two copies of
`field.py`, and the promise would quietly expire the first day real reference
tags go up on the field. One workspace, one lockfile, one `.venv`, one copy.

```
tag-tracking  ──→  vision-core
```

One direction, always. `vision-core` never imports from a skill. That is also
why `vision-core` is a separate package rather than a folder inside
`tag-tracking`: nothing in it is about AprilTags.

## Layout

```
pyproject.toml     workspace root: package = false, depends on all members
uv.lock            one lockfile for everything
calib/             intrinsics.json — resolved via repo_root()
vision-core/       field.py, camera.py, intrinsics.py, paths.py, planview.py
tag-tracking/      pose.py, track.py, selfcheck.py, serve_tag.py
```

`calib/` sits at the root and is resolved by walking up from the source file
(`vision_core.paths.repo_root`), never from the current directory — otherwise
"which intrinsics did it load?" would depend on which folder you launched from,
and the failure is silent, because a missing file just falls back to a guessed
field of view.

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
