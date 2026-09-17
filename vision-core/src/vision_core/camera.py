"""Opening a camera, and locking it so the picture stops changing under you.

The locking half matters far more than it sounds. A webcam left on auto will
re-expose and re-white-balance whenever something bright moves through frame, and
a color-based tracker is then chasing a target whose hue and saturation drift
every few seconds. Turning both off is the single biggest robustness win
available to the ball tracker, and it costs two property writes.

Tag detection does not care -- it works on a thresholded greyscale image -- so
the lock is opt-in rather than the default.
"""

from __future__ import annotations

import cv2

#: OpenCV's DSHOW backend wants 0.25 for "manual exposure" and 0.75 for "auto".
#: These are not documented constants, they are what the backend does.
_DSHOW_EXPOSURE_MANUAL = 0.25
_DSHOW_EXPOSURE_AUTO = 0.75

_BACKENDS = ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY"))


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows (MSMF is slow to start)."""
    last = None
    for api, name in _BACKENDS:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"camera {index} open via {name} at {got_w}x{got_h}")
                return cap
            last = f"{name} opened but returned no frame"
        cap.release()
    raise SystemExit(
        f"could not open camera {index} ({last or 'no backend worked'}).\n"
        f"Try --list-cameras, close Teams/Zoom, or check Windows camera privacy settings."
    )


def list_cameras(count: int = 5) -> None:
    print(f"probing camera indices 0-{count - 1} ...")
    for i in range(count):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"  index {i}: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  index {i}: opens but no frame (in use by another app?)")
        cap.release()


def lock_camera(cap: cv2.VideoCapture) -> list[str]:
    """Freeze exposure and white balance. Returns what actually stuck.

    These properties are advisory: plenty of webcams accept the write, carry on
    doing whatever they like, and OpenCV reports success either way. So read each
    value back and say honestly what happened, rather than printing "locked" and
    leaving you to wonder why the mask still breathes.

    Whatever the camera had settled on is what gets frozen, so point it at the
    scene and let it settle before calling this.
    """
    notes = []
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    notes.append("auto-WB off" if abs(cap.get(cv2.CAP_PROP_AUTO_WB)) < 0.5
                 else "auto-WB REFUSED")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _DSHOW_EXPOSURE_MANUAL)
    notes.append("auto-exposure off"
                 if abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - _DSHOW_EXPOSURE_AUTO) > 1e-6
                 else "auto-exposure REFUSED")
    return notes
