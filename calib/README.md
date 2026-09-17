# calib

Calibration describing the **camera**, not any one skill. Shared by every skill
and resolved from the repo root by `vision_core.paths.repo_root()`, so it does
not matter which folder you launch from. Created on first write — nothing to set
up.

| File | Made by | What it is |
| --- | --- | --- |
| `intrinsics.json` | `s` in either tracker | The camera matrix: how zoomed-in the lens is. Stores the resolution it was measured at, and is ignored on load if that does not match. |

Committed, because this team shares one rig. Later skills add their own files
here — ball tracking will drop a `ball_color.json` alongside it.

> ⚠️ If you end up on **different webcams**, gitignore `intrinsics.json`. The
> loader only rejects a mismatched *resolution*, so two different cameras both at
> 1280x720 would silently share one wrong `fx` — and every reported distance
> would be off with nothing on screen to flag it.
