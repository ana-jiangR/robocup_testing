"""Live AprilTag tracking against a virtual field.

    uv run track --tag-size 0.080

Shows the camera feed with the virtual field drawn on it, a top-down plan view,
and each tag's position in field coordinates. The field is not real: it is
whatever vision_core/field.py says it is.

Keys:
    q / Esc  quit           g  grid on/off          t  trails on/off
    [ / ]    fx -/+ 2%      w  wall / floor         r  reset trails
    s        save intrinsics to calib/intrinsics.json
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pupil_apriltags as pa
from vision_core import intrinsics as intr
from vision_core.camera import list_cameras, open_camera
from vision_core.field import CameraFieldTransform, Field, SyntheticFieldTransform
from vision_core.planview import OUT_COLOR, PlanView, color_for, draw_field

from .pose import TagFieldPose, tag_field_pose

TRAIL_LEN = 90


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


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
    ap.add_argument("--mode", choices=["wall", "floor"], default="wall",
                    help="virtual field standing up facing you, or lying flat")
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
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    field = Field(args.field[0], args.field[1])
    mode = args.mode
    transform = build_transform(args, field, mode)

    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera opened but the first frame failed")
    h, w = frame.shape[:2]

    if args.hfov is not None:
        K = intr.from_fov(w, h, args.hfov)
        intr_source = f"assumed {args.hfov:g} deg HFOV"
    else:
        K, _dist, intr_source = intr.load(w, h)

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

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped returning frames")
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
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
            print(f"fx {K[0, 0]:.1f}  -> {intr.hfov_of(K, w):.1f} deg HFOV")
        elif key == ord("w"):
            mode = "floor" if mode == "wall" else "wall"
            transform = build_transform(args, field, mode)
            plan = PlanView(field, height=h, mode=mode)
            trails.clear()
            print(f"transform: {transform.source}")
        elif key == ord("s"):
            path = intr.save(K, w, h, f"{intr_source} (no lens distortion)")
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
