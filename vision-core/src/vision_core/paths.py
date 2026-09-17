"""Where the repo is, so shared files resolve regardless of the current directory.

calib/ holds things that describe the *camera*, not any one skill: the
intrinsics, and the ball color profiles. A cwd-relative "calib/intrinsics.json"
would mean different files depending on which folder you launched from -- and the
failure is quiet, because a missing intrinsics file just falls back to a guessed
field of view.

So resolve it from this file's location instead. uv installs every workspace
member editable, so __file__ really is the checked-out source tree.
"""

from __future__ import annotations

from pathlib import Path

#: Only the workspace root has the lockfile; members share it.
_MARKER = "uv.lock"


def repo_root() -> Path:
    """The workspace root: the directory containing uv.lock and calib/."""
    for d in Path(__file__).resolve().parents:
        if (d / _MARKER).exists():
            return d
    raise RuntimeError(
        f"no {_MARKER} above {__file__}; vision_core is not installed from the "
        f"workspace source tree. Run 'uv sync' from the repo root."
    )


def calib_dir() -> Path:
    """The shared calibration directory. Not created until something writes."""
    return repo_root() / "calib"


def calib_path(name: str) -> Path:
    """A file inside calib/, e.g. calib_path("intrinsics.json")."""
    return calib_dir() / name
