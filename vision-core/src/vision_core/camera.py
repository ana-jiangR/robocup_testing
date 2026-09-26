"""Opening a camera, and locking it so the picture stops changing under you.

The locking half matters far more than it sounds. A webcam left on auto will
re-expose and re-white-balance whenever something bright moves through frame, and
a color-based tracker is then chasing a target whose hue and saturation drift
every few seconds. Turning both off is the single biggest robustness win
available to the ball tracker, and it costs two property writes.

Tag detection does not care -- it works on a thresholded greyscale image -- so
the lock is opt-in rather than the default.

The lock is written into the driver, not into this process, so it outlives the
program that set it: lock and exit, and every other app inherits a frozen
picture. Whatever locks must unlock_camera() on the way out, and
`uv run unlock-camera` exists for the runs that die before they can.
"""

from __future__ import annotations

import argparse
import math
import sys

import cv2

#: OpenCV's DSHOW backend wants 0.25 for "manual exposure" and 0.75 for "auto".
#: These are not documented constants, they are what the backend does. Only
#: meaningful on Windows/DSHOW -- lock_camera() is a no-op in effect elsewhere.
_DSHOW_EXPOSURE_MANUAL = 0.25
_DSHOW_EXPOSURE_AUTO = 0.75


def _backends() -> list[tuple[int, str]]:
    """Backends to try, in a sensible order for the OS actually running this.

    DirectShow/MSMF are Windows-only; trying them elsewhere just wastes time
    falling through to CAP_ANY, which does not always resolve to a working
    backend on its own -- macOS in particular needs AVFoundation named
    explicitly, or camera access can fail silently before the OS even gets to
    prompt for permission.
    """
    if sys.platform == "darwin":
        return [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (cv2.CAP_ANY, "ANY")]
    if sys.platform.startswith("win"):
        return [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]
    return [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")]


def read_key(delay_ms: int = 1) -> int:
    """cv2.waitKey with the key folded to lowercase, so Caps Lock or Shift
    cannot silently make 'q' and 's' stop working."""
    key = cv2.waitKey(delay_ms) & 0xFF
    return key + 32 if ord("A") <= key <= ord("Z") else key


def _fourcc_str(value: float) -> str | None:
    """cv2's FOURCC property, a packed int returned as a float, as 4 letters.

    None when it is not 4 printable letters: DSHOW on some webcams answers
    every FOURCC read with the same junk number whatever the format really is.
    """
    v = int(value) & 0xFFFFFFFF
    s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.isascii() and s.isprintable() else None


def open_camera(
    index: int,
    width: int,
    height: int,
    *,
    fourcc: str | None = None,
    buffersize: int | None = None,
) -> cv2.VideoCapture:
    """Open a camera, trying backends in an OS-appropriate order.

    `fourcc` (e.g. "MJPG") asks for a pixel format, and has to be set before
    the resolution: DSHOW picks the format when the size is written, so a
    FOURCC written afterwards is ignored. MJPG is what lets most USB webcams
    reach 720p at 30 fps or better; raw YUY2 at that size saturates USB 2.

    `buffersize` asks the driver to queue at most that many frames. Only some
    backends honour it (V4L2 and some DSHOW drivers; plenty of DSHOW webcams
    do not even report it), and the result is printed either way rather than
    assumed.
    """
    last = None
    for api, name in _backends():
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            if fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if buffersize is not None:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
            ok, _ = cap.read()
            if ok:
                got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"camera {index} open via {name} at {got_w}x{got_h}")
                if fourcc is not None:
                    got = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
                    if got is None:
                        print(f"  asked for {fourcc}; pixel format not reported by this "
                              "backend (check the fps instead)")
                    else:
                        print(f"  pixel format {got}" + (
                            "" if got == fourcc else f" (asked for {fourcc}; camera refused)"))
                if buffersize is not None:
                    got_buf = cap.get(cv2.CAP_PROP_BUFFERSIZE)
                    print(f"  driver buffer {got_buf:g} frame(s)" if got_buf > 0 else
                          "  driver buffer size not reported by this backend")
                # A camera remembers the last run's lock, so this one can open
                # already frozen -- on a stale manual exposure, that is the dark
                # green picture nobody can account for. Start from auto every
                # time; anything that wants the lock asks for it right after.
                # Only the WB half is readable on every camera, so it stands in
                # as the tell-tale for the whole lock.
                if abs(cap.get(cv2.CAP_PROP_AUTO_WB)) < 0.5:
                    print("  (camera was left locked by an earlier run "
                          "-- putting it back on auto)")
                unlock_camera(cap)
                return cap
            last = f"{name} opened but returned no frame"
        cap.release()
    hint = (
        "check System Settings > Privacy & Security > Camera and make sure your "
        "terminal app is allowed"
        if sys.platform == "darwin" else
        "check Windows camera privacy settings"
    )
    raise SystemExit(
        f"could not open camera {index} ({last or 'no backend worked'}).\n"
        f"Try --list-cameras, close Teams/Zoom or other apps using the camera, or {hint}."
    )


def list_cameras(count: int = 5) -> None:
    print(f"probing camera indices 0-{count - 1} ...")
    api, _name = _backends()[0]
    for i in range(count):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"  index {i}: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  index {i}: opens but no frame (in use by another app?)")
        cap.release()


def _mean_brightness(cap: cv2.VideoCapture, frames: int = 8) -> float:
    """Average pixel value over a few frames, 0-255.

    Cheap enough to run either side of a property write, which is what makes
    it possible to tell whether the write helped or hurt.
    """
    total, n = 0.0, 0
    for _ in range(frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            total += float(frame.mean())
            n += 1
    return total / n if n else 0.0


def _color_balance(cap: cv2.VideoCapture, frames: int = 8) -> tuple[float, float]:
    """(G/R, G/B) over a few frames: the two numbers a white-balance shift moves.

    A neutral scene sits near (1.0, 1.0). Both ratios drifting up together is
    the green cast a stale manual white balance produces.
    """
    r = g = b = 0.0
    for _ in range(frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            b += float(frame[:, :, 0].mean())
            g += float(frame[:, :, 1].mean())
            r += float(frame[:, :, 2].mean())
    if r < 1.0 or b < 1.0:
        return 1.0, 1.0
    return g / r, g / b


def _settle(cap: cv2.VideoCapture, frames: int = 12) -> None:
    """Read and throw away frames so a property write can take effect.

    Writing a property is not the same as the sensor having acted on it. The
    picture crosses over a good dozen frames, so anything measured immediately
    after the write still describes the old state.
    """
    for _ in range(frames):
        cap.read()


def _exposure_note(cap: cv2.VideoCapture, *, want_auto: bool) -> str:
    """What the camera says about auto-exposure, without pretending to know.

    A negative value means the property is not reported at all, which is not
    the same as the write having worked -- say so rather than claiming success.
    """
    value = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    want = _DSHOW_EXPOSURE_AUTO if want_auto else _DSHOW_EXPOSURE_MANUAL
    word = "on" if want_auto else "off"
    if value < 0:
        return f"auto-exposure {word}: UNKNOWN (camera does not report it)"
    return f"auto-exposure {word}" if abs(value - want) < 0.2 else "auto-exposure REFUSED"


#: A picture this dark is not a scene, it is a camera in a bad state -- a
#: covered lens, a room with the lights off, or a manual exposure that could not
#: be matched. Worth saying out loud, since a color tracker will find nothing
#: and not explain why.
_TOO_DARK = 40

#: How far G/R or G/B may move when white balance goes manual before it counts
#: as a tint rather than noise. Frame-to-frame jitter on a still scene is about
#: 0.02; the green cast measured on a real camera moved G/B by 0.28.
_TINT_TOLERANCE = 0.06

#: DSHOW exposure is a log2 of seconds (-6 is 1/64 s), so one step doubles or
#: halves the light. Landing where auto was takes one or two corrections; four
#: is a ceiling against a camera that ignores the write, not a budget.
_EXPOSURE_MAX_STEPS = 4


def _match_exposure(cap: cv2.VideoCapture, target: float) -> tuple[float, float]:
    """Walk the manual exposure until the picture is as bright as `target`.

    Switching DSHOW to manual pins exposure at whatever stale value the driver
    holds, not at what auto had settled on -- on some cameras that is several
    stops darker, and the picture collapses to a dark green smear. The reported
    EXPOSURE value under auto is not to be trusted either, so rather than copy
    it, measure: each step of the log2 scale doubles the light, so the number of
    steps needed is just log2(target / current), rounded.

    Returns (exposure value, mean brightness) where it ended up.
    """
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    mean = _mean_brightness(cap)
    for _ in range(_EXPOSURE_MAX_STEPS):
        if mean < 1.0 or target < 1.0:
            break
        step = int(round(math.log2(target / mean)))
        if step == 0:
            break
        exposure += step
        if not cap.set(cv2.CAP_PROP_EXPOSURE, exposure):
            break
        _settle(cap)
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)  # cameras clamp; take theirs
        mean = _mean_brightness(cap)
    return exposure, mean


def lock_camera(cap: cv2.VideoCapture) -> list[str]:
    """Freeze exposure and white balance. Returns what actually stuck.

    These properties are advisory: plenty of webcams accept the write, carry on
    doing whatever they like, and OpenCV reports success either way. So read each
    value back and say honestly what happened, rather than printing "locked" and
    leaving you to wonder why the mask still breathes.

    Neither write is safe to trust. Going manual on exposure does not freeze
    the picture where it is; it jumps to a stale value the driver remembers,
    which can be stops darker. So the brightness auto had reached is measured
    first and the manual value walked until it matches. Manual white balance on
    some cameras is not a freeze either but a fixed temperature the auto
    balance was never on, which shows up as a green cast; that cannot be walked
    back, so the color balance is measured either side of the write and the
    lock rolled back if it moved. Either half that fails is handed back to auto
    and the note says so -- a slowly drifting picture costs the tracker some
    accuracy; a black or green frame costs it the ball.

    Whatever the camera had settled on is what gets frozen, so point it at the
    scene and let it settle before calling this. Pair it with unlock_camera on
    the way out -- these properties outlive the process that set them.
    """
    notes = []
    target = _mean_brightness(cap)
    gr0, gb0 = _color_balance(cap)

    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
    if abs(cap.get(cv2.CAP_PROP_AUTO_WB)) >= 0.5:
        notes.append("auto-WB REFUSED")
    else:
        _settle(cap, 20)
        gr1, gb1 = _color_balance(cap)
        if abs(gr1 - gr0) > _TINT_TOLERANCE or abs(gb1 - gb0) > _TINT_TOLERANCE:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
            _settle(cap, 20)
            notes.append(f"auto-WB REVERTED -- manual WB tints the picture "
                         f"(G/B {gb0:.2f} -> {gb1:.2f}), left on auto")
        else:
            notes.append("auto-WB off")

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _DSHOW_EXPOSURE_MANUAL)
    _settle(cap)
    exposure, mean = _match_exposure(cap, target)
    if target >= _TOO_DARK and mean < max(_TOO_DARK, target * 0.5):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _DSHOW_EXPOSURE_AUTO)
        _settle(cap)
        notes.append(f"auto-exposure REVERTED -- manual could not match auto's "
                     f"brightness ({target:.0f} -> {mean:.0f}), left on auto")
    else:
        note = _exposure_note(cap, want_auto=False)
        if note.startswith("auto-exposure off"):
            note = (f"auto-exposure off (EXPOSURE {exposure:g}, "
                    f"brightness {target:.0f} -> {mean:.0f})")
        notes.append(note)

    if _mean_brightness(cap) < _TOO_DARK:
        notes.append("WARNING picture is very dark -- check the lens cover "
                     "and the room")
    return notes


def unlock_camera(cap: cv2.VideoCapture) -> list[str]:
    """Hand the camera back to auto exposure and auto white balance.

    What lock_camera writes lives in the driver, not in this process. It
    outlasts the program that set it, so a tool that locks and exits leaves
    every other app -- and the next run of this one -- looking at a frozen
    picture, which on some cameras means a dark green one. Anything that locks
    unlocks on the way out.
    """
    notes = []
    cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
    notes.append("auto-WB on" if cap.get(cv2.CAP_PROP_AUTO_WB) >= 0.5
                 else "auto-WB REFUSED")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _DSHOW_EXPOSURE_AUTO)
    notes.append(_exposure_note(cap, want_auto=True))
    _settle(cap)  # give the camera a moment to re-expose
    return notes


def unlock_main() -> None:
    """`uv run unlock-camera` -- for when a tool died before it could unlock.

    Nothing else in the repo can reach a camera left locked by a crashed run,
    and the symptom (a dark green picture in every app) does not look like
    something this project caused.
    """
    ap = argparse.ArgumentParser(description=unlock_main.__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    args = ap.parse_args()

    cap = open_camera(args.camera, 1280, 720)
    try:
        print("camera unlock: " + ", ".join(unlock_camera(cap)))
        print(f"picture now averages {_mean_brightness(cap):.0f}/255 "
              f"(under about 40 is still too dark -- check the lens cover "
              f"and the room)")
    finally:
        cap.release()
