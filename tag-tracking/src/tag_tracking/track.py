"""Live AprilTag tracking against a virtual field.

    uv run track --tag-size 0.080
    uv run track --synthetic-camera      # no webcam: watch it work on a fully simulated field
    uv run track --calibrate-live        # real webcam: calibrate from reference tags, then track

Shows the camera feed with the virtual field drawn on it, a top-down plan view,
and each tag's position in field coordinates. The field is not real: it is
whatever vision_core/field.py says it is -- and with --synthetic-camera, neither
is the camera feed; a fake, moving tag on a fake field is calibrated and
tracked live, with no hardware at all, through this exact same window.
--calibrate-live is the real-camera counterpart: point a real webcam at real
reference tags (e.g. 4 printed AprilTags on a paper field) and it solves the
field pose live, in this same window, before tracking begins -- no separate
calibrate-field run or saved calib/field_pose.json required first.

Keys:
    q / Esc  quit           g  grid on/off          t  trails on/off
    [ / ]    fx -/+ 2%      w  wall / floor         r  reset trails
    s        save intrinsics to calib/intrinsics.json
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pupil_apriltags as pa
from vision_core import intrinsics as intr
from vision_core.camera import list_cameras, open_camera
from vision_core.field import (
    CameraFieldTransform,
    Field,
    ReferenceTagFieldTransform,
    SyntheticFieldTransform,
)
from vision_core.planview import OUT_COLOR, PlanView, color_for, draw_field

from .calibrate_field import (
    MIN_FRAMES as CALIB_MIN_FRAMES,
    calibrate_field,
    capture_reference_frames,
    default_field_layout,
    field_pose_path,
)
from .pose import TagFieldPose, tag_field_pose

TRAIL_LEN = 90


@contextlib.contextmanager
def _suppressed_stderr():
    """Silence warnings pupil_apriltags prints straight to the C-level stderr
    file descriptor ("more than one new minima found", "Matrix is singular")
    -- benign notices from its per-tag pose solver when a flat tag is viewed
    close to head-on, where two tilts explain the image almost equally well.
    It picks the better one either way and keeps going; there is no real
    hardware issue underneath to want to see. Happens with real cameras too
    (a tag held flat-on is the same ambiguous geometry, camera or not), so
    this wraps every detect() call, not just --synthetic-camera. Python
    exceptions are unaffected -- this only hides that library's own fprintf
    output.
    """
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield  # stderr is not a real file (captured / IDE console): nothing to silence
        return
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _undistort_maps(K: np.ndarray, dist: np.ndarray, w: int, h: int):
    """Precomputed remap tables for cv2.remap, or None when there is no distortion."""
    if not np.any(dist):
        return None
    return cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)


def draw_tag_marks(frame: np.ndarray, det, pose: TagFieldPose, color) -> None:
    """Outline the detected tag in the video and label it."""
    quad = np.asarray(det.corners, np.int32)
    cv2.polylines(frame, [quad], True, color, 2, cv2.LINE_AA)
    c = np.asarray(det.center, int)
    cv2.circle(frame, tuple(c), 4, color, -1, cv2.LINE_AA)
    label = f"id {pose.tag_id}  {pose.x:+.2f}, {pose.y:+.2f} m"
    cv2.putText(frame, label, (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2, cv2.LINE_AA)


def draw_plan(
    plan: PlanView,
    poses: list[TagFieldPose],
    trails: dict[int, deque],
    transform: CameraFieldTransform,
    show_trails: bool,
) -> np.ndarray:
    """Plan view with one dot per tag, carrying a heading arrow."""
    panel = plan.base()
    plan.draw_camera(panel, transform)

    if show_trails:
        for tid, trail in trails.items():
            plan.draw_trail(panel, trail, color_for(tid))

    for pose in poses:
        color = color_for(pose.tag_id)
        p = plan.to_px(pose.x, pose.y)
        if not plan.on_panel(p):
            continue  # off the panel entirely
        # Heading arrow, 12 cm long, in field coords.
        plan.draw_arrow(panel, pose.x, pose.y, pose.theta_deg, 0.12, color)
        cv2.circle(panel, p, 7, color, -1 if pose.inside else 2, cv2.LINE_AA)
        cv2.putText(panel, str(pose.tag_id), (p[0] + 9, p[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    return panel


def draw_readout(
    frame: np.ndarray,
    poses: list[TagFieldPose],
    transform: CameraFieldTransform,
    K: np.ndarray,
    tag_size: float,
    intr_source: str,
    fps: float,
) -> None:
    """The text block: what the transform is, and one line per tag."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
    head = (
        f"{transform.source}  |  tag {tag_size * 1000:.0f} mm  |  "
        f"fx {K[0, 0]:.0f} ({intr.hfov_of(K, w):.0f} deg HFOV, {intr_source})"
        f"  |  {fps:4.1f} fps"
    )
    cv2.putText(frame, head, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)

    # A solid strip behind the numbers: a text halo turns to mush over a busy
    # camera image, and these are the numbers the whole exercise is about.
    n = max(len(poses), 1)
    strip_h = 20 + 24 * n
    cv2.rectangle(frame, (0, h - strip_h), (w, h), (0, 0, 0), -1)

    if not poses:
        cv2.putText(frame, "no tags detected", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 140, 210), 1, cv2.LINE_AA)
        return

    y = h - strip_h + 28
    for pose in sorted(poses, key=lambda p: p.tag_id):
        color = color_for(pose.tag_id)
        state = "IN " if pose.inside else "OUT"
        line = (
            f"id {pose.tag_id:2d}   x {pose.x:+6.3f}   y {pose.y:+6.3f}   "
            f"th {pose.theta_deg:+7.1f}   off-plane {pose.off_plane_m:+6.3f}   {state}"
        )
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    color if pose.inside else OUT_COLOR, 1, cv2.LINE_AA)
        y += 24


# --------------------------------------------------------------------------


def build_transform(args, field: Field, mode: str) -> CameraFieldTransform:
    """The one call that decides where the field is.

    Swapping in ReferenceTagFieldTransform for the real RoboCup field means
    changing this function and nothing else.
    """
    if mode == "floor":
        return SyntheticFieldTransform.floor(field, args.cam_height, args.pitch)
    return SyntheticFieldTransform.wall(field, args.distance, args.yaw, args.pitch_wall)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q quit, g grid, t trails, [ ] fx, w wall/floor, r reset, s save",
    )
    ap.add_argument("--tag-size", type=float, default=0.080,
                    help="black-square width in metres (default 0.080)")
    ap.add_argument("--field", type=float, nargs=2, metavar=("W", "H"),
                    default=[1.2, 0.8], help="field size in metres (default 1.2 0.8)")
    ap.add_argument("--mode", choices=["wall", "floor"], default=None,
                    help="virtual field standing up facing you, or lying flat "
                         "(default wall; --synthetic-camera defaults to floor instead, "
                         "since a fronto-parallel tag is the worst case for the "
                         "detector's per-tag pose solver)")
    ap.add_argument("--distance", type=float, default=1.8,
                    help="wall mode: metres from camera to the field plane. The "
                         "default fits a 1.2x0.8 m field in a 60-degree view; "
                         "closer crops the field, further shrinks it")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="wall mode: rotate the field about vertical, degrees")
    ap.add_argument("--pitch-wall", type=float, default=0.0,
                    help="wall mode: tip the field top-away, degrees")
    ap.add_argument("--cam-height", type=float, default=1.5,
                    help="floor mode: camera height above the field, metres. The "
                         "default frames a 1.2x0.8 m field in a 60-degree view "
                         "with room to spare; below about 1.3 m the field no "
                         "longer fits")
    ap.add_argument("--pitch", type=float, default=90.0,
                    help="floor mode: camera downward tilt from horizontal, "
                         "degrees. 90 is straight down, directly over the field "
                         "centre, which is the rig this is built for. Lower it "
                         "only if your camera really is off to one side")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=None,
                    help="assumed horizontal FOV in degrees, if not calibrated")
    ap.add_argument("--heading-offset", type=float, default=0.0,
                    help="degrees added to theta, if the tag is not mounted upright")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--display-width", type=int, default=1600,
                    help="shrink the window if the composite is wider than this")
    ap.add_argument("--print-poses", action="store_true",
                    help="also stream (id, x, y, theta) to stdout")
    ap.add_argument("--field-pose", type=str, default=None,
                    help="field pose file to load, or to save to with --calibrate-live "
                         "(default calib/field_pose.json at the repo root -- see calibrate-field)")
    ap.add_argument("--synthetic-field", action="store_true",
                    help="use the made-up field even if a calibrated one is saved")
    ap.add_argument("--synthetic-camera", action="store_true",
                    help="no webcam: track an animated tag on a simulated field instead, "
                         "calibrating live from simulated reference tags")
    ap.add_argument("--calibrate-live", action="store_true",
                    help="real webcam: calibrate the field pose live from reference tags "
                         "(e.g. 4 printed AprilTags on a paper field) before tracking, "
                         "instead of loading a previously saved calib/field_pose.json")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    field = Field(args.field[0], args.field[1])
    mode = args.mode or ("floor" if args.synthetic_camera else "wall")
    cap = None
    sim_cam = None
    calibrated_note = ""

    if args.synthetic_camera:
        from .synthetic import SyntheticFieldCamera, orbit_trajectory

        print("SIMULATED CAMERA -- no webcam opened. The field, the camera, and the")
        print("tracked tag are all synthetic; everything else on screen -- overlay,")
        print("plan view, readout -- is exactly what a real run would show.\n")

        h, w = args.height, args.width
        K = intr.from_fov(w, h)
        dist, intr_source = np.zeros(5), "synthetic camera (ground truth K)"
        layout = default_field_layout(field)
        ground_truth = build_transform(args, field, mode)
        sim_cam = SyntheticFieldCamera(
            layout, ground_truth, tag_size=args.tag_size, shape=(h, w), K=K,
            trajectory=orbit_trajectory(field),
        )

        print(f"calibrating field pose from {len(layout)} simulated reference tags...")
        calib_frames = [sim_cam.read() for _ in range(20)]
        result = calibrate_field(calib_frames, layout, K, min_frames=10,
                                 tag_size=args.tag_size)
        print(f"  {result.transform.source}")
        print(f"  agreement across frames: rotation spread "
              f"{result.rotation_spread_deg:.3f} deg, translation spread "
              f"{result.translation_spread_m * 1000:.2f} mm\n")
        transform = result.transform
        using_calibrated = True
        calibrated_note = "solved live from the simulated reference tags"
        if args.calibrate_live:
            print("note: --calibrate-live is ignored with --synthetic-camera "
                  "(the field pose is already solved live, from simulated tags)\n")
    else:
        cap = open_camera(args.camera, args.width, args.height)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera opened but the first frame failed")
        h, w = frame.shape[:2]

        if args.hfov is not None:
            K = intr.from_fov(w, h, args.hfov)
            dist, intr_source = np.zeros(5), f"assumed {args.hfov:g} deg HFOV"
        else:
            K, dist, intr_source = intr.load(w, h)

        if args.calibrate_live:
            layout = default_field_layout(field)
            print(f"calibrating live: point the camera at reference tags {sorted(layout)} "
                  f"at the corners of a {field.width:g} x {field.height:g} m field")
            print("(e.g. printed AprilTags taped to the corners of a paper rectangle).")
            print(f"hold steady; capturing until {CALIB_MIN_FRAMES}+ good frames, "
                  "'q' to finish, Esc to abort.\n")

            def _read_bgr() -> np.ndarray:
                ok2, frm = cap.read()
                if not ok2:
                    raise RuntimeError("camera stopped returning frames")
                return frm

            calib_frames = capture_reference_frames(
                _read_bgr, layout, min_frames=CALIB_MIN_FRAMES,
                window_name="calibrate-field (live)",
            )
            result = calibrate_field(calib_frames, layout, K, dist, min_frames=CALIB_MIN_FRAMES,
                                     tag_size=args.tag_size)
            print(f"\n{result.transform.source}")
            print(f"agreement across frames: rotation spread "
                  f"{result.rotation_spread_deg:.3f} deg, translation spread "
                  f"{result.translation_spread_m * 1000:.2f} mm")
            transform = result.transform
            using_calibrated = True
            calibrated_note = "solved live this session"

            out_path = Path(args.field_pose) if args.field_pose else field_pose_path()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            transform.save(out_path)
            print(f"saved {out_path}\n")
        else:
            out_path = Path(args.field_pose) if args.field_pose else field_pose_path()
            using_calibrated = out_path.exists() and not args.synthetic_field
            if using_calibrated:
                transform = ReferenceTagFieldTransform.load(out_path)
                calibrated_note = f"loaded from {out_path}"
                print(f"loaded calibrated field pose from {out_path}")
            else:
                transform = build_transform(args, field, mode)

    detector = pa.Detector(families="tag36h11", nthreads=4, quad_decimate=1.0,
                           decode_sharpening=0.25)

    show_grid, show_trails = True, True
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    plan = PlanView(field, height=h, mode=mode)
    fps, last_t = 0.0, time.perf_counter()

    print(f"\nfield {field.width:g} x {field.height:g} m, {transform.source}")
    print(f"tag size {args.tag_size * 1000:.1f} mm, intrinsics: {intr_source}")
    print("window open -- q or Esc to quit\n")

    win = "virtual field tracking"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # Undistort the whole frame, not just the grey copy the detector sees:
    # the overlay is drawn on `frame`, so both must live in the same (pinhole)
    # pixel space or the outlines drift off the tags towards the edges.
    # Maps are built once; cv2.undistort rebuilds them every call (~8x slower).
    undistort_maps = _undistort_maps(K, dist, w, h)

    while True:
        if sim_cam is not None:
            frame = cv2.cvtColor(sim_cam.read(), cv2.COLOR_GRAY2BGR)
        else:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped returning frames")
                break
        if undistort_maps is not None:
            frame = cv2.remap(frame, *undistort_maps, cv2.INTER_LINEAR)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        with _suppressed_stderr():
            dets = detector.detect(grey, estimate_tag_pose=True,
                                   camera_params=cam_params, tag_size=args.tag_size)

        poses = [
            tag_field_pose(d, transform, field, args.heading_offset) for d in dets
        ]

        draw_field(frame, field, transform, K, show_grid)
        for d, pose in zip(dets, poses):
            draw_tag_marks(frame, d, pose, color_for(pose.tag_id))
            trails[pose.tag_id].append((pose.x, pose.y))
        draw_readout(frame, poses, transform, K, args.tag_size, intr_source, fps)

        if args.print_poses:
            for p in sorted(poses, key=lambda p: p.tag_id):
                print(f"{p.tag_id}\t{p.x:.4f}\t{p.y:.4f}\t{p.theta_deg:.2f}", flush=True)

        panel = draw_plan(plan, poses, trails, transform, show_trails)
        composite = np.hstack([frame, panel])
        if composite.shape[1] > args.display_width:
            k = args.display_width / composite.shape[1]
            composite = cv2.resize(composite, None, fx=k, fy=k,
                                   interpolation=cv2.INTER_AREA)
        cv2.imshow(win, composite)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
        last_t = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("g"):
            show_grid = not show_grid
        elif key == ord("t"):
            show_trails = not show_trails
        elif key == ord("r"):
            trails.clear()
        elif key in (ord("["), ord("]")):
            K = K.copy()
            K[0, 0] *= 0.98 if key == ord("[") else 1.02
            K[1, 1] = K[0, 0]
            intr_source = "hand-tuned"
            undistort_maps = _undistort_maps(K, dist, w, h)
            print(f"fx {K[0, 0]:.1f}  -> {intr.hfov_of(K, w):.1f} deg HFOV")
        elif key == ord("w"):
            if using_calibrated:
                extra = ("restart without --synthetic-camera for a movable wall/floor demo"
                          if sim_cam is not None else
                          "pass --synthetic-field to demo the made-up field instead")
                print(f"field is calibrated ({calibrated_note}); {extra}")
            else:
                mode = "floor" if mode == "wall" else "wall"
                transform = build_transform(args, field, mode)
                plan = PlanView(field, height=h, mode=mode)
                trails.clear()
                print(f"transform: {transform.source}")
        elif key == ord("s"):
            if sim_cam is not None:
                print("no real camera to calibrate in --synthetic-camera mode")
            else:
                # Keep whatever distortion was loaded: saving K alone would
                # silently zero a calibrate-camera result.
                path = intr.save(K, w, h, intr_source, dist=dist)
                print(f"saved {path}")
                if not np.any(dist):
                    print("(fx only, no lens distortion -- run calibrate-camera for that)")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
