"""Live AprilTag tracking against a virtual field.

    uv run track --tag-size 0.080

Shows the camera feed with the virtual field drawn on it, a top-down plan view,
and each tag's position in field coordinates. The field is not real: it is
whatever field.py says it is.

Keys:
    q / Esc  quit           g  grid on/off          t  trails on/off
    [ / ]    fx -/+ 2%      w  wall / floor         r  reset trails
    s        save intrinsics to calib/intrinsics.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pupil_apriltags as pa

from .field import (
    CameraFieldTransform,
    Field,
    SyntheticFieldTransform,
    TagFieldPose,
    tag_field_pose,
)

CALIB_PATH = Path("calib/intrinsics.json")
PLAN_W = 440  # width of the plan-view panel, pixels
TRAIL_LEN = 90

# BGR. Distinct hues so several tags stay readable at once.
TAG_COLOURS = [
    (80, 200, 255),
    (120, 255, 140),
    (255, 160, 90),
    (200, 130, 255),
    (90, 230, 230),
    (255, 210, 120),
]
FIELD_COLOUR = (90, 230, 90)
OUT_COLOUR = (90, 110, 255)


# --------------------------------------------------------------------------
# Intrinsics
# --------------------------------------------------------------------------


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """A pinhole guess from an assumed horizontal field of view.

    Good enough to get the demo running: see the README on why faking this is
    fine now and not fine later.
    """
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], np.float64
    )


def load_intrinsics(path: Path, width: int, height: int) -> tuple[np.ndarray, str]:
    """Real calibration if we have it for this resolution, otherwise a guess."""
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("width") == width and d.get("height") == height:
            return np.array(d["camera_matrix"], np.float64), d.get("source", str(path))
        print(
            f"note: {path} is for {d.get('width')}x{d.get('height')}, "
            f"camera is {width}x{height} -- ignoring it"
        )
    return intrinsics_from_fov(width, height, 60.0), "assumed 60 deg HFOV"


def hfov_of(K: np.ndarray, width: int) -> float:
    return float(np.degrees(2.0 * np.arctan((width / 2.0) / K[0, 0])))


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def project(pts_field: np.ndarray, transform: CameraFieldTransform, K: np.ndarray):
    """Field points -> pixels, plus a flag for whether they are in front of us.

    cv2.projectPoints happily returns coordinates for points behind the camera,
    which then draw as wild lines across the frame. So check the depth ourselves.
    """
    cam = transform.field_to_camera(pts_field)
    in_front = cam[:, 2] > 1e-3
    rvec, tvec = transform.rvec_tvec()
    px, _ = cv2.projectPoints(
        np.asarray(pts_field, np.float64).reshape(-1, 1, 3), rvec, tvec, K, np.zeros(5)
    )
    return px.reshape(-1, 2), in_front


def draw_field(
    frame: np.ndarray,
    field: Field,
    transform: CameraFieldTransform,
    K: np.ndarray,
    show_grid: bool,
) -> None:
    """The virtual field outline, grid, origin and axes, overlaid on the video."""
    if show_grid:
        for seg in field.grid(0.2):
            px, ok = project(seg, transform, K)
            if ok.all():
                cv2.line(
                    frame,
                    tuple(px[0].astype(int)),
                    tuple(px[1].astype(int)),
                    (60, 120, 60),
                    1,
                    cv2.LINE_AA,
                )

    corners, ok = project(field.corners(), transform, K)
    if ok.all():
        cv2.polylines(
            frame, [corners.astype(np.int32)], True, FIELD_COLOUR, 2, cv2.LINE_AA
        )
        for i, name in enumerate(["(0,0)", "(W,0)", "(W,H)", "(0,H)"]):
            p = corners[i].astype(int)
            cv2.circle(frame, tuple(p), 4, FIELD_COLOUR, -1, cv2.LINE_AA)
            cv2.putText(
                frame, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, FIELD_COLOUR, 1, cv2.LINE_AA,
            )
    elif ok.any():
        cv2.putText(
            frame, "field partly behind camera", (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, OUT_COLOUR, 2, cv2.LINE_AA,
        )

    # Field axes at the origin corner: X red, Y green (0.2 m each).
    axes = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]], np.float64)
    px, ok = project(axes, transform, K)
    if ok.all():
        o = tuple(px[0].astype(int))
        cv2.arrowedLine(frame, o, tuple(px[1].astype(int)), (70, 70, 255), 2,
                        cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, o, tuple(px[2].astype(int)), (70, 255, 70), 2,
                        cv2.LINE_AA, tipLength=0.25)


def draw_tag_marks(frame: np.ndarray, det, pose: TagFieldPose, colour) -> None:
    """Outline the detected tag in the video and label it."""
    quad = np.asarray(det.corners, np.int32)
    cv2.polylines(frame, [quad], True, colour, 2, cv2.LINE_AA)
    c = np.asarray(det.center, int)
    cv2.circle(frame, tuple(c), 4, colour, -1, cv2.LINE_AA)
    label = f"id {pose.tag_id}  {pose.x:+.2f}, {pose.y:+.2f} m"
    cv2.putText(frame, label, (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 2, cv2.LINE_AA)


def draw_plan(
    field: Field,
    poses: list[TagFieldPose],
    trails: dict[int, deque],
    transform: CameraFieldTransform,
    height: int,
    show_trails: bool,
    mode: str = "wall",
) -> np.ndarray:
    """Plan view of the field: +X right, +Y up, origin at bottom-left.

    In floor mode this really is a top-down view. In wall mode the field is
    vertical, so it is a head-on view of the plane -- same axes either way.
    """
    panel = np.full((height, PLAN_W, 3), 24, np.uint8)
    pad, top = 34, 64
    # Fit the rectangle in the usable box, then centre it so the panel does not
    # end up with all its dead space at one end.
    usable_w, usable_h = PLAN_W - 2 * pad, height - top - pad
    # Leave a margin outside the rectangle so a tag that strays out of bounds is
    # still drawn somewhere visible instead of being clipped off the panel.
    scale = 0.86 * min(usable_w / field.width, usable_h / field.height)
    fw, fh = field.width * scale, field.height * scale
    ox = pad + (usable_w - fw) / 2.0  # origin pixel: bottom-left of the rect
    oy = top + (usable_h - fh) / 2.0 + fh

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(ox + x * scale)), int(round(oy - y * scale))

    title = "TOP-DOWN (field coords)" if mode == "floor" else "FIELD PLANE (head-on)"
    cv2.putText(panel, title, (pad - 12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{field.width:g} x {field.height:g} m", (pad - 12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1, cv2.LINE_AA)

    for seg in field.grid(0.2):
        cv2.line(panel, to_px(*seg[0][:2]), to_px(*seg[1][:2]), (48, 48, 48), 1)
    cv2.rectangle(panel, to_px(0, 0), to_px(field.width, field.height),
                  FIELD_COLOUR, 1, cv2.LINE_AA)

    # Origin corner and axes.
    cv2.circle(panel, to_px(0, 0), 4, FIELD_COLOUR, -1, cv2.LINE_AA)
    cv2.putText(panel, "0,0", (to_px(0, 0)[0] - 26, to_px(0, 0)[1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, FIELD_COLOUR, 1, cv2.LINE_AA)
    cv2.arrowedLine(panel, to_px(0, 0), to_px(0.18, 0), (70, 70, 255), 2,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(panel, to_px(0, 0), to_px(0, 0.18), (70, 255, 70), 2,
                    cv2.LINE_AA, tipLength=0.3)

    # Where the camera is. This panel only shows the two in-plane axes, so the
    # camera's distance out of the plane is spelled out rather than implied --
    # in wall mode it sits "over" the middle of the field but metres in front.
    cam = transform.camera_origin_in_field()
    cpx = to_px(float(cam[0]), float(cam[1]))
    if 0 <= cpx[0] < PLAN_W and 0 <= cpx[1] < height:
        cv2.drawMarker(panel, cpx, (170, 170, 170), cv2.MARKER_TRIANGLE_UP, 11, 2)
        cv2.putText(panel, f"cam ({cam[2]:+.2f} m out)", (cpx[0] + 9, cpx[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

    if show_trails:
        for tid, trail in trails.items():
            colour = TAG_COLOURS[tid % len(TAG_COLOURS)]
            pts = [to_px(x, y) for x, y in trail]
            for i in range(1, len(pts)):
                fade = i / len(pts)
                cv2.line(panel, pts[i - 1], pts[i],
                         tuple(int(c * fade * 0.7) for c in colour), 1, cv2.LINE_AA)

    for pose in poses:
        colour = TAG_COLOURS[pose.tag_id % len(TAG_COLOURS)]
        p = to_px(pose.x, pose.y)
        if not (0 <= p[0] < PLAN_W and 0 <= p[1] < height):
            continue  # off the panel entirely
        # Heading arrow, 12 cm long, in field coords.
        th = np.radians(pose.theta_deg)
        tip = to_px(pose.x + 0.12 * np.cos(th), pose.y + 0.12 * np.sin(th))
        cv2.arrowedLine(panel, p, tip, colour, 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(panel, p, 7, colour, -1 if pose.inside else 2, cv2.LINE_AA)
        cv2.putText(panel, str(pose.tag_id), (p[0] + 9, p[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)
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
        f"fx {K[0, 0]:.0f} ({hfov_of(K, w):.0f} deg HFOV, {intr_source})  |  {fps:4.1f} fps"
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
        colour = TAG_COLOURS[pose.tag_id % len(TAG_COLOURS)]
        state = "IN " if pose.inside else "OUT"
        line = (
            f"id {pose.tag_id:2d}   x {pose.x:+6.3f}   y {pose.y:+6.3f}   "
            f"th {pose.theta_deg:+7.1f}   off-plane {pose.off_plane_m:+6.3f}   {state}"
        )
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    colour if pose.inside else OUT_COLOUR, 1, cv2.LINE_AA)
        y += 24


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows (MSMF is slow to start)."""
    last = None
    for api, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")):
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


def list_cameras() -> None:
    print("probing camera indices 0-4 ...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"  index {i}: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  index {i}: opens but no frame (in use by another app?)")
        cap.release()


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
    ap.add_argument("--cam-height", type=float, default=1.0,
                    help="floor mode: camera height above the field, metres")
    ap.add_argument("--pitch", type=float, default=35.0,
                    help="floor mode: camera downward tilt, degrees")
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
        K, intr_source = intrinsics_from_fov(w, h, args.hfov), f"assumed {args.hfov:g} deg HFOV"
    else:
        K, intr_source = load_intrinsics(CALIB_PATH, w, h)

    detector = pa.Detector(families="tag36h11", nthreads=4, quad_decimate=1.0,
                           decode_sharpening=0.25)

    show_grid, show_trails = True, True
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
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
            draw_tag_marks(frame, d, pose, TAG_COLOURS[pose.tag_id % len(TAG_COLOURS)])
            trails[pose.tag_id].append((pose.x, pose.y))
        draw_readout(frame, poses, transform, K, args.tag_size, intr_source, fps)

        if args.print_poses:
            for p in sorted(poses, key=lambda p: p.tag_id):
                print(f"{p.tag_id}\t{p.x:.4f}\t{p.y:.4f}\t{p.theta_deg:.2f}", flush=True)

        plan = draw_plan(field, poses, trails, transform, h, show_trails, mode)
        composite = np.hstack([frame, plan])
        if composite.shape[1] > args.display_width:
            k = args.display_width / composite.shape[1]
            composite = cv2.resize(composite, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
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
            print(f"fx {K[0, 0]:.1f}  -> {hfov_of(K, w):.1f} deg HFOV")
        elif key == ord("w"):
            mode = "floor" if mode == "wall" else "wall"
            transform = build_transform(args, field, mode)
            trails.clear()
            print(f"transform: {transform.source}")
        elif key == ord("s"):
            CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
            CALIB_PATH.write_text(json.dumps({
                "width": w, "height": h,
                "camera_matrix": K.tolist(),
                "dist_coeffs": [0.0] * 5,
                "source": f"{intr_source} (no lens distortion)",
            }, indent=2))
            print(f"saved {CALIB_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
