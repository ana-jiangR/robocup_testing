"""One-time field-pose calibration: point a calibrated camera at the fixed
reference AprilTags and solve where the field is, once.

    uv run calibrate-field                                    # live camera, default 4-corner layout
    uv run calibrate-field --field 1.2 0.8                    # a different field size
    uv run calibrate-field --layout 0:0,0 1:1.2,0 2:1.2,0.8 3:0,0.8
    uv run calibrate-field --synthetic                        # self-test, no hardware

Wraps ReferenceTagFieldTransform (vision_core.field) -- the geometry already
lives there; from_detections() alone is a single noisy solvePnP call on
whatever frame you hand it. This makes running it a repeatable, checked,
one-shot command: it reads several frames, solves the pose from each
independently, reports how much they agree (a real quality signal, not just
"it ran"), averages them, and saves.

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
from vision_core.camera import open_camera
from vision_core.field import Field, ReferenceTagFieldTransform, SyntheticFieldTransform
from vision_core.paths import calib_path

from .pose import tag_field_pose

FIELD_POSE_NAME = "field_pose.json"
MIN_FRAMES = 10
MAX_LIVE_FRAMES = 40


def field_pose_path() -> Path:
    return calib_path(FIELD_POSE_NAME)


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
) -> FieldCalibrationResult:
    """Solve the field pose from several frames of the reference tags and
    average, instead of trusting one noisy solvePnP call.

    Detection only needs det.center (pixel coordinates); solvePnP inside
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
            t = ReferenceTagFieldTransform.from_detections(dets, layout, K, dist)
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
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("aborted")
                raise SystemExit(0)
            if key == ord("q") and len(frames) >= min_frames:
                break
    finally:
        cv2.destroyWindow(win)
    return frames


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
    result = calibrate_field(frames, layout, K, min_frames=args.min_frames)
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


def _run_live(args, field: Field, layout: dict[int, tuple[float, float]]):
    from vision_core.intrinsics import load as load_intrinsics

    K, dist, intr_source = load_intrinsics(args.width, args.height)
    print(f"intrinsics: {intr_source}")
    print(f"looking for reference tags {sorted(layout)} at {layout}")
    print(f"hold the camera steady on the field; capturing until {args.min_frames}+ good frames")
    print("'q' to finish once enough are captured, Esc to abort.\n")

    cap = open_camera(args.camera, args.width, args.height)

    def _read_bgr() -> np.ndarray:
        ok, frm = cap.read()
        if not ok:
            raise RuntimeError("camera stopped returning frames")
        return frm

    try:
        frames = capture_reference_frames(
            _read_bgr, layout, args.min_frames, MAX_LIVE_FRAMES, "calibrate-field"
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    result = calibrate_field(frames, layout, K, dist, min_frames=args.min_frames)
    print(f"\n{result.transform.source}")
    print(
        f"agreement across frames: rotation spread {result.rotation_spread_deg:.3f} deg, "
        f"translation spread {result.translation_spread_m * 1000:.2f} mm"
    )
    return result, (args.out or str(field_pose_path()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--synthetic", action="store_true",
        help="self-test against a digital field, no hardware needed",
    )
    ap.add_argument(
        "--field", type=float, nargs=2, metavar=("W", "H"), default=[1.2, 0.8],
        help="field size in metres, used to build the default 4-corner layout",
    )
    ap.add_argument(
        "--layout", nargs="+", metavar="ID:X,Y",
        help="override the default corners, e.g. 0:0,0 1:1.2,0 2:1.2,0.8 3:0,0.8",
    )
    ap.add_argument("--tag-size", type=float, default=0.080)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument(
        "--out", type=str, default=None,
        help="where to save (default calib/field_pose.json at the repo root; "
        "--synthetic saves nowhere unless you pass this)",
    )
    args = ap.parse_args()

    field = Field(args.field[0], args.field[1])
    layout = parse_layout(args.layout) if args.layout else default_field_layout(field)

    result, out = (
        _run_synthetic(args, field, layout) if args.synthetic else _run_live(args, field, layout)
    )

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        result.transform.save(out)
        print(f"\nsaved {out}")
    else:
        print("\n(--synthetic run, nothing saved -- pass --out to save anyway)")


if __name__ == "__main__":
    main()
