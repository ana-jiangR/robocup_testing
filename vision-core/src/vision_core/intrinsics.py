"""Camera intrinsics: load, save, and a usable guess when there is no calibration.

fx turns apparent size in pixels into metric distance, so it decides whether
"2 metres" means two metres. A guessed fx is survivable while the field is
synthetic and stops being survivable once reference tags locate it -- see the
tag-tracking README.

Everything is keyed on resolution: a matrix for 1280x720 is wrong for 640x480,
and using it silently would be worse than guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .paths import calib_path

#: Shared across every skill -- it describes the camera, not the task.
DEFAULT_CALIB = "intrinsics.json"

#: What a generic webcam is, absent any better information.
DEFAULT_HFOV_DEG = 60.0


def intrinsics_path() -> Path:
    return calib_path(DEFAULT_CALIB)


def from_fov(width: int, height: int, hfov_deg: float = DEFAULT_HFOV_DEG) -> np.ndarray:
    """A pinhole guess from an assumed horizontal field of view.

    Square pixels, principal point at the centre, no distortion. Good enough to
    get a demo running and honest about being a guess.
    """
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], np.float64
    )


def hfov_of(K: np.ndarray, width: int) -> float:
    """The horizontal field of view a camera matrix implies, in degrees."""
    return float(np.degrees(2.0 * np.arctan((width / 2.0) / K[0, 0])))


def undistort_maps(K: np.ndarray, dist: np.ndarray, width: int, height: int):
    """Precomputed remap tables for cv2.remap, or None when there is no distortion.

    Whole-frame undistortion (as opposed to undistorting one point at a time)
    is worth the setup cost whenever a detector searches the whole image, not
    just a single already-known point -- an AprilTag detector finds corners
    everywhere, so the frame it sees has to be in the same undistorted pinhole
    space the drawn overlay is, or outlines drift off the tags towards the
    frame edges. Built once per K/dist/size; cv2.undistort() rebuilds the maps
    on every call, about 8x slower.
    """
    if not np.any(dist):
        return None
    return cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_16SC2)


def load(
    width: int, height: int, path: Path | None = None
) -> tuple[np.ndarray, np.ndarray, str]:
    """Real calibration if we have it for this resolution, otherwise a guess.

    Returns (camera_matrix, dist_coeffs, source), where `source` is a short
    human-readable note to put on screen so nobody mistakes a guess for a
    measurement.
    """
    path = path or intrinsics_path()
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("width") == width and d.get("height") == height:
            K = np.array(d["camera_matrix"], np.float64)
            dist = np.array(d.get("dist_coeffs", [0.0] * 5), np.float64).ravel()
            return K, dist, d.get("source", str(path))
        print(
            f"note: {path} is for {d.get('width')}x{d.get('height')}, "
            f"camera is {width}x{height} -- ignoring it"
        )
    return (
        from_fov(width, height, DEFAULT_HFOV_DEG),
        np.zeros(5, np.float64),
        f"assumed {DEFAULT_HFOV_DEG:g} deg HFOV",
    )


def save(
    K: np.ndarray,
    width: int,
    height: int,
    source: str,
    dist: np.ndarray | None = None,
    path: Path | None = None,
) -> Path:
    """Write a camera matrix to calib/, creating the directory if needed."""
    path = path or intrinsics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dist = np.zeros(5, np.float64) if dist is None else np.asarray(dist, np.float64)
    path.write_text(
        json.dumps(
            {
                "width": int(width),
                "height": int(height),
                "camera_matrix": np.asarray(K, np.float64).tolist(),
                "dist_coeffs": dist.ravel().tolist(),
                "source": source,
            },
            indent=2,
        )
    )
    return path
