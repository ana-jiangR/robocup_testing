"""One-time field-pose calibration: point a calibrated camera at the fixed
reference AprilTags and solve where the field is, once.

    uv run calibrate-field                                    # live camera, default 4-corner layout
    uv run calibrate-field --field 1.2 0.8                    # a different field size
    uv run calibrate-field --layout 0:0,0 1:1.2,0 2:1.2,0.8 3:0,0.8
    uv run calibrate-field --synthetic                        # self-test, no hardware
    uv run calibrate-field --sequential                       # one tag, moved to each corner in turn

Wraps ReferenceTagFieldTransform (vision_core.field) -- the geometry already
lives there; from_detections() alone is a single noisy solvePnP call on
whatever frame you hand it. This makes running it a repeatable, checked,
one-shot command.

Two ways to capture the reference points, both solving the same underlying
PnP problem:

    simultaneous (default) -- all 4 reference tags fixed to the field and
        visible at once. Reads several whole frames, solves the pose from
        each independently, reports how much they agree, averages them.
    sequential (--sequential) -- one tag, carried to each corner in turn
        (e.g. mounted on a robot that drives there itself), camera fixed.
        Averages that one tag's pixel position at each corner, then solves
        once from the four averaged points.

Saves to calib/field_pose.json, at the repo root (vision_core.paths), same as
calib/intrinsics.json. track.py loads this automatically once it exists, in
place of the made-up field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pupil_apriltags as pa
from vision_core.camera import open_camera, read_frame, read_key
from vision_core.field import (
    Field,
    ReferenceTagFieldTransform,
    SyntheticFieldTransform,
    field_pose_path,
)

from .pose import tag_field_pose

MIN_FRAMES = 10
MAX_LIVE_FRAMES = 40
FRAMES_PER_CORNER = 15
#: --sequential: a new sample this far (px) from the running mean means the tag
#: is being moved, not held, so the corner's samples restart.
STILL_PX = 3.0
#: --sequential: a tag this close (px) to an already-confirmed corner is still
#: sitting where the last corner was captured, not at the next one yet.
MIN_CORNER_SEP_PX = 40.0


def default_field_layout(field: Field) -> dict[int, tuple[float, float]]:
    """Tags 0-3 at the four corners, in the same order Field.corners() uses."""
    w, h = field.width, field.height
    return {0: (0.0, 0.0), 1: (w, 0.0), 2: (w, h), 3: (0.0, h)}


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _average_rotations(Rs: list[np.ndarray]) -> np.ndarray:
    """Mean rotation via SVD re-orthonormalisation. Fine for the small spread
    expected between repeated views of a camera that has not moved -- this is
    not meant to average rotations that disagree by a lot.
    """
    M = np.mean(Rs, axis=0)
    U, _s, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


@dataclass
class FieldCalibrationResult:
    transform: ReferenceTagFieldTransform
    n_frames: int
    rotation_spread_deg: float
    translation_spread_m: float


def calibrate_field(
    frames: Iterable[np.ndarray],
    layout: dict[int, tuple[float, float]],
    K: np.ndarray,
    dist: np.ndarray | None = None,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_LIVE_FRAMES,
    tag_size: float | None = None,
) -> FieldCalibrationResult:
    """Solve the field pose from several frames of the reference tags and
    average, instead of trusting one noisy solvePnP call.

    Pass `tag_size` (metres) so each solve uses all four corners of every tag
    rather than just the centres. solvePnP inside
    ReferenceTagFieldTransform.from_detections applies `dist` itself, so the
    frames do not need to be undistorted first.
    """
    detector = pa.Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    seen = 0
    for frame in frames:
        seen += 1
        if len(Rs) >= max_frames:
            break
        grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(grey)
        try:
            t = ReferenceTagFieldTransform.from_detections(dets, layout, K, dist, tag_size)
        except LookupError:
            continue
        Rs.append(t.R)
        ts.append(t.t)
    if len(Rs) < min_frames:
        raise RuntimeError(
            f"only {len(Rs)}/{seen} frames saw all {len(layout)} reference tags "
            f"(ids {sorted(layout)}), need >= {min_frames}"
        )
    R_avg = _average_rotations(Rs)
    t_avg = np.mean(ts, axis=0)
    rot_spread = max(_rotation_angle_deg(R_avg.T @ R) for R in Rs)
    t_spread = max(float(np.linalg.norm(t - t_avg)) for t in ts)
    transform = ReferenceTagFieldTransform(
        R_avg, t_avg,
        f"measured, averaged over {len(Rs)} frames of {len(layout)} reference tags",
    )
    return FieldCalibrationResult(transform, len(Rs), rot_spread, t_spread)


def parse_layout(spec: list[str]) -> dict[int, tuple[float, float]]:
    """'0:0,0' '1:1.2,0' ... -> {0: (0.0, 0.0), 1: (1.2, 0.0), ...}"""
    layout: dict[int, tuple[float, float]] = {}
    for item in spec:
        tid_s, xy = item.split(":")
        x_s, y_s = xy.split(",")
        layout[int(tid_s)] = (float(x_s), float(y_s))
    return layout


def capture_reference_frames(
    read_bgr: Callable[[], np.ndarray],
    layout: dict[int, tuple[float, float]],
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_LIVE_FRAMES,
    window_name: str = "calibrate-field",
) -> list[np.ndarray]:
    """Show a live preview and collect grayscale frames until all reference
    tags have been seen `min_frames`+ times. 'q' to finish once enough are
    captured, Esc to abort (raises SystemExit).

    Takes a plain `read_bgr()` callable rather than a specific camera type, so
    both this module's own live capture and track.py's already-open
    cv2.VideoCapture can drive this identically -- this is the interactive
    half of what --calibrate-live on track re-uses directly, instead of
    duplicating the capture loop there.
    """
    detector = pa.Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    win = window_name
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    frames: list[np.ndarray] = []
    try:
        while True:
            bgr = read_bgr()
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            dets = detector.detect(grey)
            seen_ids = {d.tag_id for d in dets}
            good = set(layout).issubset(seen_ids)
            disp = bgr.copy()
            for d in dets:
                colour = (0, 255, 0) if d.tag_id in layout else (0, 165, 255)
                cv2.polylines(disp, [np.asarray(d.corners, np.int32)], True, colour, 2)
                cv2.putText(
                    disp, str(d.tag_id), tuple(np.asarray(d.center, int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
                )
            if good and len(frames) < max_frames:
                frames.append(grey)
            status_colour = (0, 255, 0) if good else (0, 0, 255)
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 84), (0, 0, 0), -1)
            cv2.putText(
                disp, f"captured {len(frames)}/{min_frames}   need ids {sorted(layout)}"
                      f"   see {sorted(seen_ids) or '(none)'}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_colour, 2, cv2.LINE_AA,
            )
            ready = len(frames) >= min_frames
            go_msg = "'q' TO FINISH -- enough captured" if ready else "'q' does nothing yet -- need more captures"
            cv2.putText(
                disp, f"{go_msg}   |   Esc to abort", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0) if ready else (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.imshow(win, disp)
            key = read_key()
            if key == 27:
                print("aborted")
                raise SystemExit(0)
            if key == ord("q") and len(frames) >= min_frames:
                break
    finally:
        cv2.destroyWindow(win)
    return frames


# --------------------------------------------------------------------------
# Sequential capture: one tag, carried to each corner in turn, camera fixed.
# --------------------------------------------------------------------------


class _AveragedDetection:
    """A fake pupil_apriltags detection standing in for one corner's averaged
    pixel centre and corners, so the averaged result can be solved through the
    exact same ReferenceTagFieldTransform.from_detections() the simultaneous
    path uses, instead of duplicating the solvePnP call-site.
    """

    __slots__ = ("tag_id", "center", "corners")

    def __init__(self, tag_id: int, center, corners) -> None:
        self.tag_id = tag_id
        self.center = np.asarray(center, np.float64).reshape(2)
        self.corners = np.asarray(corners, np.float64).reshape(4, 2)


@dataclass
class SequentialCalibrationResult:
    transform: ReferenceTagFieldTransform
    n_corners: int
    reproj_error_px: float
    max_reproj_error_px: float


def solve_sequential(
    averaged: dict[int, _AveragedDetection],
    layout: dict[int, tuple[float, float]],
    K: np.ndarray,
    dist: np.ndarray | None = None,
    tag_size: float | None = None,
) -> SequentialCalibrationResult:
    """Solve one field pose from one averaged pixel position per corner,
    instead of several tags seen at once -- the counterpart to
    calibrate_field() for a single tag carried to each corner in turn.

    There is only one solve here, not several to average, so the quality
    signal is different too: reprojection error (px) -- project each known
    field corner through the solved pose and compare to where it was actually
    seen, the same idea calibrate-camera's rms_reproj_px uses for the lens.

    Pass `tag_size` (the tag lying flat at each corner) to solve on all 16 tag
    corners rather than the 4 centres. Four points is the ambiguous worst case
    for planar PnP, and fits them near-exactly whatever the pose, so without it
    the reprojection error below says little.
    """
    fake_dets = [
        _AveragedDetection(tid, det.center, det.corners) for tid, det in averaged.items()
    ]
    transform = ReferenceTagFieldTransform.from_detections(fake_dets, layout, K, dist, tag_size)
    transform.source = f"measured, one tag moved to {len(averaged)} corners in turn"

    obj = np.array([[*layout[tid], 0.0] for tid in averaged], np.float64)
    img = np.array([averaged[tid].center for tid in averaged], np.float64)
    rvec, tvec = transform.rvec_tvec()
    dist_c = dist if dist is not None else np.zeros(5, np.float64)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist_c)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    return SequentialCalibrationResult(
        transform=transform,
        n_corners=len(averaged),
        reproj_error_px=float(np.sqrt(np.mean(err ** 2))),
        max_reproj_error_px=float(err.max()),
    )


def capture_sequential_corners(
    read_bgr: Callable[[], np.ndarray],
    layout: dict[int, tuple[float, float]],
    frames_per_corner: int = FRAMES_PER_CORNER,
    window_name: str = "calibrate-field (one tag)",
) -> dict[int, np.ndarray]:
    """Interactively capture one tag's pixel centre at each field position in
    `layout`, in turn -- for a single tag (e.g. mounted on a robot) driven to
    each corner while the camera stays fixed, instead of several tags visible
    at once.

    At each position: accumulates every frame with *exactly one* tag visible
    (zero or several are ambiguous, so skipped) up to `frames_per_corner`,
    averaging their pixel centres and corners to smooth out per-frame jitter.
    Only a tag held still counts: one that moves more than STILL_PX restarts
    the samples, and one still within MIN_CORNER_SEP_PX of a corner already
    confirmed is ignored -- otherwise the next corner fills up in half a
    second from wherever the tag was while being carried there. 'q' confirms
    the current corner and moves to the next once enough samples are in;
    'r' clears the current corner's samples to retry it; Esc aborts entirely.

    Returns {tag_id: averaged detection}, keyed the same way `layout` is --
    the physical tag's id at capture time does not have to match; only the
    order you visit `layout`'s positions in does.
    """
    detector = pa.Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    win = window_name
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    results: dict[int, _AveragedDetection] = {}
    items = list(layout.items())
    try:
        for i, (label_id, (fx, fy)) in enumerate(items):
            samples: list[np.ndarray] = []
            corner_samples: list[np.ndarray] = []
            while True:
                bgr = read_bgr()
                grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                dets = detector.detect(grey)
                disp = bgr.copy()
                for d in dets:
                    cv2.polylines(disp, [np.asarray(d.corners, np.int32)], True,
                                  (0, 165, 255), 2)
                good = len(dets) == 1
                hint = None
                if good:
                    c = np.asarray(dets[0].center, np.float64)
                    if any(np.linalg.norm(c - r.center) < MIN_CORNER_SEP_PX
                           for r in results.values()):
                        good = False
                        hint = "still at the previous corner -- move the tag"
                    elif samples and np.linalg.norm(c - np.mean(samples, axis=0)) > STILL_PX:
                        samples.clear()
                        corner_samples.clear()
                        hint = "tag moving -- set it down and let go"
                if good and len(samples) < frames_per_corner:
                    samples.append(c)
                    corner_samples.append(np.asarray(dets[0].corners, np.float64))
                ready = len(samples) >= frames_per_corner
                status_colour = (
                    (0, 255, 0) if ready else (0, 165, 255) if good else (0, 0, 255)
                )
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 84), (0, 0, 0), -1)
                cv2.putText(
                    disp,
                    f"corner {i + 1}/{len(items)}: place ONE tag at field "
                    f"({fx:g}, {fy:g})   captured {len(samples)}/{frames_per_corner}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 2, cv2.LINE_AA,
                )
                if hint is not None:
                    msg = hint
                elif len(dets) > 1:
                    msg = f"{len(dets)} tags visible -- show exactly one"
                elif not good:
                    msg = "no tag visible"
                else:
                    msg = "'q' to confirm and move on" if ready else "hold steady -- capturing"
                cv2.putText(
                    disp, f"{msg}   |   'r' retry this corner   |   Esc abort",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if ready else (200, 200, 200), 1, cv2.LINE_AA,
                )
                cv2.imshow(win, disp)
                key = read_key()
                if key == 27:
                    print("aborted")
                    raise SystemExit(0)
                if key == ord("r"):
                    samples.clear()
                    corner_samples.clear()
                if key == ord("q") and ready:
                    break
            avg = np.mean(samples, axis=0)
            px_std = float(np.std(np.linalg.norm(np.array(samples) - avg, axis=1)))
            results[label_id] = _AveragedDetection(
                label_id, avg, np.mean(corner_samples, axis=0)
            )
            print(
                f"  corner {i + 1}/{len(items)} at ({fx:g}, {fy:g}): "
                f"averaged {len(samples)} samples, pixel ({avg[0]:.1f}, {avg[1]:.1f}), "
                f"jitter {px_std:.2f} px"
            )
    finally:
        cv2.destroyWindow(win)
    return results


def _run_synthetic(args, field: Field, layout: dict[int, tuple[float, float]]):
    from vision_core.intrinsics import from_fov

    from .synthetic import SyntheticFieldCamera

    shape = (args.height, args.width)
    K = from_fov(args.width, args.height)
    truth = SyntheticFieldTransform.floor(field, height=1.1, pitch_deg=40.0)
    moving_truth = (round(field.width * 0.6, 3), round(field.height * 0.55, 3), 30.0)
    cam = SyntheticFieldCamera(
        layout, truth, tag_size=args.tag_size, shape=shape, K=K,
        extra_tags=((7, *moving_truth),),
    )
    print(f"Synthetic field -- ground truth: {truth.source}\n")

    frames = (cam.read() for _ in range(20))
    result = calibrate_field(frames, layout, K, min_frames=args.min_frames,
                             tag_size=args.tag_size)
    print(f"{result.transform.source}")
    print(
        f"agreement across frames: rotation spread {result.rotation_spread_deg:.3f} deg, "
        f"translation spread {result.translation_spread_m * 1000:.2f} mm"
    )
    dR = _rotation_angle_deg(result.transform.R.T @ truth.R)
    dt = float(np.linalg.norm(result.transform.t - truth.t))
    print(f"vs ground truth: rotation off {dR:.2f} deg, origin off {dt * 1000:.1f} mm")

    detector = pa.Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    check_frame = cam.read()
    dets = detector.detect(
        check_frame, estimate_tag_pose=True,
        camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]), tag_size=args.tag_size,
    )
    moving = next((d for d in dets if d.tag_id == 7), None)
    if moving is not None:
        pose = tag_field_pose(moving, result.transform, field)
        print(
            f"moving tag readback: x={pose.x:.3f} y={pose.y:.3f} th={pose.theta_deg:+.1f}  "
            f"(placed at x={moving_truth[0]:.3f} y={moving_truth[1]:.3f} th={moving_truth[2]:+.1f})"
        )
    return result, args.out


def _open_live(args):
    """Open the camera and load intrinsics for the size it ACTUALLY delivers.

    Not args.width/height: a camera that ignores the request (640x480 when
    asked for 1280x720) would otherwise miss the real calibration, fall back to
    a guessed lens for the wrong image size, and solve the field ~2x too far
    away -- with no error, just a field that lines up at one corner only.
    """
    from vision_core.intrinsics import load as load_intrinsics

    cap = open_camera(args.camera, args.width, args.height)
    ok, frm = read_frame(cap)
    if not ok:
        cap.release()
        raise RuntimeError("camera opened but returned no frame")
    h, w = frm.shape[:2]
    K, dist, intr_source = load_intrinsics(w, h)
    print(f"intrinsics: {intr_source}")
    if not intr_source.startswith("charuco"):
        print(f"\nWARNING: no measured lens calibration for {w}x{h}. The field pose will "
              f"be solved with a guessed lens and every distance can be off by tens of %.\n"
              f"Run 'uv run calibrate-camera --camera {args.camera} --width {w} --height {h}' "
              f"first unless this is only a rough test.\n")

    def _read_bgr() -> np.ndarray:
        ok, frm = read_frame(cap)
        if not ok:
            raise RuntimeError("camera stopped returning frames")
        return frm

    return cap, _read_bgr, K, dist, f"{w}x{h}, {intr_source}"


def _run_live(args, field: Field, layout: dict[int, tuple[float, float]]):
    cap, _read_bgr, K, dist, lens_note = _open_live(args)
    print(f"looking for reference tags {sorted(layout)} at {layout}")
    print(f"hold the camera steady on the field; capturing until {args.min_frames}+ good frames")
    print("'q' to finish once enough are captured, Esc to abort.\n")

    try:
        frames = capture_reference_frames(
            _read_bgr, layout, args.min_frames, MAX_LIVE_FRAMES, "calibrate-field"
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    result = calibrate_field(frames, layout, K, dist, min_frames=args.min_frames,
                             tag_size=args.tag_size)
    print(f"\n{result.transform.source}")
    print(
        f"agreement across frames: rotation spread {result.rotation_spread_deg:.3f} deg, "
        f"translation spread {result.translation_spread_m * 1000:.2f} mm"
    )
    result.transform.source += f" ({lens_note})"
    return result, (args.out or str(field_pose_path()))


def _run_synthetic_sequential(args, field: Field, layout: dict[int, tuple[float, float]]):
    from vision_core.intrinsics import from_fov

    from .synthetic import SyntheticFieldCamera

    shape = (args.height, args.width)
    K = from_fov(args.width, args.height)
    truth = SyntheticFieldTransform.floor(field, height=1.1, pitch_deg=40.0)
    print(f"Synthetic field -- ground truth: {truth.source}")
    print("simulating one tag carried to each corner in turn...\n")

    detector = pa.Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    averaged: dict[int, _AveragedDetection] = {}
    for tag_id, (cx, cy) in layout.items():
        cam = SyntheticFieldCamera(
            {}, truth, tag_size=args.tag_size, shape=shape, K=K,
            extra_tags=((tag_id, cx, cy, 0.0),),
        )
        samples: list[np.ndarray] = []
        corner_samples: list[np.ndarray] = []
        while len(samples) < args.frames_per_corner:
            dets = detector.detect(cam.read())
            if len(dets) == 1:
                samples.append(np.asarray(dets[0].center, np.float64))
                corner_samples.append(np.asarray(dets[0].corners, np.float64))
        avg = np.mean(samples, axis=0)
        averaged[tag_id] = _AveragedDetection(tag_id, avg, np.mean(corner_samples, axis=0))
        print(f"  corner id {tag_id} at ({cx:g}, {cy:g}): "
              f"averaged {len(samples)} synthetic samples, pixel ({avg[0]:.1f}, {avg[1]:.1f})")

    result = solve_sequential(averaged, layout, K, tag_size=args.tag_size)
    print(f"\n{result.transform.source}")
    print(f"reprojection error: {result.reproj_error_px:.3f} px rms, "
          f"{result.max_reproj_error_px:.3f} px max")
    dR = _rotation_angle_deg(result.transform.R.T @ truth.R)
    dt = float(np.linalg.norm(result.transform.t - truth.t))
    print(f"vs ground truth: rotation off {dR:.2f} deg, origin off {dt * 1000:.1f} mm")
    return result, args.out


def _run_live_sequential(args, field: Field, layout: dict[int, tuple[float, float]]):
    cap, _read_bgr, K, dist, lens_note = _open_live(args)
    print(f"one tag, moved to {len(layout)} positions in turn: {list(layout.values())}")
    print("camera stays fixed. At each position: hold the tag steady, then")
    print("'q' confirms and moves to the next corner ('r' retries this one), Esc aborts.\n")

    try:
        averaged = capture_sequential_corners(
            _read_bgr, layout, args.frames_per_corner, "calibrate-field (one tag)"
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    result = solve_sequential(averaged, layout, K, dist, tag_size=args.tag_size)
    print(f"\n{result.transform.source}")
    print(f"reprojection error: {result.reproj_error_px:.3f} px rms, "
          f"{result.max_reproj_error_px:.3f} px max")
    result.transform.source += f" ({lens_note})"
    return result, (args.out or str(field_pose_path()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--synthetic", action="store_true",
        help="self-test against a digital field, no hardware needed",
    )
    ap.add_argument(
        "--sequential", action="store_true",
        help="one tag, moved to each corner in turn (camera fixed) instead of "
             "all reference tags visible at once -- e.g. a robot carrying a "
             "single AprilTag, driven to each corner",
    )
    ap.add_argument(
        "--field", type=float, nargs=2, metavar=("W", "H"), default=[1.2, 0.8],
        help="field size in metres, used to build the default 4-corner layout",
    )
    ap.add_argument(
        "--layout", nargs="+", metavar="ID:X,Y",
        help="override the default corners, e.g. 0:0,0 1:1.2,0 2:1.2,0.8 3:0,0.8 "
             "(--sequential: visited in this order; the ids are just labels)",
    )
    ap.add_argument("--tag-size", type=float, default=0.080)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument(
        "--frames-per-corner", type=int, default=FRAMES_PER_CORNER,
        help="--sequential: samples to average at each corner (default 15)",
    )
    ap.add_argument(
        "--out", type=str, default=None,
        help="where to save (default calib/field_pose.json at the repo root; "
        "--synthetic saves nowhere unless you pass this)",
    )
    args = ap.parse_args()

    field = Field(args.field[0], args.field[1])
    layout = parse_layout(args.layout) if args.layout else default_field_layout(field)

    if args.synthetic:
        result, out = (
            _run_synthetic_sequential(args, field, layout) if args.sequential
            else _run_synthetic(args, field, layout)
        )
    else:
        result, out = (
            _run_live_sequential(args, field, layout) if args.sequential
            else _run_live(args, field, layout)
        )

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        result.transform.save(out)
        print(f"\nsaved {out}")
    else:
        print("\n(--synthetic run, nothing saved -- pass --out to save anyway)")


if __name__ == "__main__":
    main()
