"""Live tracking of the robot's AprilTag and the ball, together, one camera.

    uv run track-combined --ball-profile test --tag-size 0.080

Opens the camera once and runs both detectors against the same frame: the
AprilTag detector + per-tag Kalman filter from tag-tracking, and the
color-based ball detector + Kalman filter from ball-tracking. Same field,
same camera, same window -- one plan view with both the robot and the ball
on it, one combined readout strip.

This does not replace `uv run track` or `uv run track-ball` -- both still
work exactly as before, independently, for whichever one thing you actually
need. This is for when you need both at once. Nothing here reimplements
detection or filtering: it calls straight into TagTracker/BallTracker and the
same draw_tag_marks/draw_ball_marks each skill already uses, so improvements
to either tracker show up here too, automatically.

Keys:
    q / Esc  quit            g  grid on/off           t  trails on/off
    m        mask overlay    w  wall / floor          r  reset trails/tracks
    [ / ]    fx -/+ 2%       s  save intrinsics
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pupil_apriltags as pa
from ball_tracking.ball import BallFieldState, BallTracker
from ball_tracking.ball_color import color_file, load_all, load_profile
from ball_tracking.track_ball import AIR_COLOR, BALL_COLOR, draw_ball_marks
from ball_tracking.track_ball import GHOST_COLOR as BALL_GHOST_COLOR
from tag_tracking.filter import SPEED_EPS as TAG_SPEED_EPS, TagFieldState, TagTracker
from tag_tracking.pose import TagFieldPose, tag_field_pose
from tag_tracking.track import GHOST_COLOR as TAG_GHOST_COLOR
from tag_tracking.track import VELOCITY_COLOR, draw_tag_marks, suppressed_stderr
from vision_core import intrinsics as intr
from vision_core.camera import list_cameras, lock_camera, open_camera, read_key, unlock_camera
from vision_core.field import (
    CameraFieldTransform,
    Field,
    ReferenceTagFieldTransform,
    SyntheticFieldTransform,
    field_pose_path,
)
from vision_core.planview import OUT_COLOR, PlanView, color_for, draw_field

TAG_TRAIL_LEN = 90
BALL_TRAIL_LEN = 120


# --------------------------------------------------------------------------
# Drawing: one readout, one plan view, both tag(s) and ball on each.
# --------------------------------------------------------------------------


def draw_combined_readout(
    frame: np.ndarray,
    tag_states: list[TagFieldState],
    ball_state: BallFieldState | None,
    field: Field,
    transform: CameraFieldTransform,
    K: np.ndarray,
    tag_size: float,
    ball_name: str,
    ball_radius_m: float,
    intr_source: str,
    fps: float,
) -> None:
    """Header (what's calibrated, fps), then one line per tag, then the ball."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
    head = (
        f"{transform.source}  |  tag {tag_size * 1000:.0f}mm  |  "
        f"ball '{ball_name}' r={ball_radius_m * 1000:.0f}mm  |  "
        f"fx {K[0, 0]:.0f} ({intr.hfov_of(K, w):.0f} deg HFOV, {intr_source})  |  {fps:4.1f} fps"
    )
    cv2.putText(frame, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (220, 220, 220), 1, cv2.LINE_AA)

    n_tag_lines = max(len(tag_states), 1)
    n_ball_lines = 2
    strip_h = 16 + 22 * n_tag_lines + 22 * n_ball_lines
    cv2.rectangle(frame, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    y = h - strip_h + 20

    if not tag_states:
        cv2.putText(frame, "no tags detected", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (140, 140, 210), 1, cv2.LINE_AA)
        y += 22
    else:
        for s in tag_states:
            color = color_for(s.tag_id) if s.visible else TAG_GHOST_COLOR
            where = "IN " if s.inside else "OUT"
            line = (
                f"tag {s.tag_id:2d}  x {s.x:+6.3f}  y {s.y:+6.3f}  th {s.theta_deg:+7.1f}  "
                f"v {s.speed:4.2f} m/s  {where}{'' if s.visible else '  coasting'}"
            )
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color if s.inside else OUT_COLOR, 1, cv2.LINE_AA)
            y += 22

    if ball_state is None:
        cv2.putText(frame, "no ball detected (check: uv run calibrate-ball)",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 210), 1, cv2.LINE_AA)
        return

    status = (
        f"COASTING {ball_state.age * 1000:.0f}ms" if not ball_state.visible
        else "AIRBORNE" if not ball_state.grounded
        else "grounded"
    )
    color = (
        BALL_GHOST_COLOR if not ball_state.visible
        else AIR_COLOR if not ball_state.grounded
        else BALL_COLOR
    )
    if not ball_state.inside(field):
        color = OUT_COLOR
    line1 = (
        f"ball  x {ball_state.x:+6.3f}  y {ball_state.y:+6.3f}  z {ball_state.z:+6.3f}  "
        f"{'IN ' if ball_state.inside(field) else 'OUT'}  {status}"
    )
    line2 = (
        f"      speed {ball_state.speed:5.3f} m/s  dir {ball_state.direction_deg:+7.1f} deg"
    )
    cv2.putText(frame, line1, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, line2, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_combined_plan(
    plan: PlanView,
    tag_states: list[TagFieldState],
    tag_trails: dict[int, deque],
    ball_state: BallFieldState | None,
    ball_trail: deque,
    transform: CameraFieldTransform,
    show_trails: bool,
) -> np.ndarray:
    """One plan-view panel with every tag and the ball on it."""
    panel = plan.base()
    plan.draw_camera(panel, transform)

    if show_trails:
        for tid, trail in tag_trails.items():
            plan.draw_trail(panel, trail, color_for(tid))
        if len(ball_trail) > 1:
            plan.draw_trail(panel, ball_trail, BALL_COLOR)

    for s in tag_states:
        color = color_for(s.tag_id) if s.visible else TAG_GHOST_COLOR
        p = plan.to_px(s.x, s.y)
        if not plan.on_panel(p):
            continue
        plan.draw_arrow(panel, s.x, s.y, s.theta_deg, 0.12, color)
        if s.speed > TAG_SPEED_EPS:
            plan.draw_arrow(panel, s.x, s.y, s.direction_deg, 0.5 * s.speed, VELOCITY_COLOR, 1)
        cv2.circle(panel, p, 7, color, -1 if s.inside else 2, cv2.LINE_AA)
        cv2.putText(panel, str(s.tag_id), (p[0] + 9, p[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    if ball_state is not None:
        color = (
            BALL_GHOST_COLOR if not ball_state.visible
            else AIR_COLOR if not ball_state.grounded
            else BALL_COLOR
        )
        p = plan.to_px(ball_state.x, ball_state.y)
        if plan.on_panel(p):
            if ball_state.speed > 0.05:
                plan.draw_arrow(panel, ball_state.x, ball_state.y, ball_state.direction_deg,
                                min(0.20 * ball_state.speed, 0.45), color, 2)
            r = max(plan.metres(0.02), 5)
            cv2.circle(panel, p, r, color, -1 if ball_state.visible else 2, cv2.LINE_AA)
            if not ball_state.grounded:
                cv2.circle(panel, p, r + 5, AIR_COLOR, 1, cv2.LINE_AA)
    return panel


# --------------------------------------------------------------------------


def build_transform(args, field: Field, mode: str) -> CameraFieldTransform:
    """Same shape as both trackers' own -- swapping in a measured field pose
    means changing this function and nothing else."""
    if mode == "floor":
        return SyntheticFieldTransform.floor(field, args.cam_height, args.pitch)
    return SyntheticFieldTransform.wall(field, args.distance, args.yaw, args.pitch_wall)


def _new_ball_tracker(args, color, transform, K, dist) -> BallTracker:
    return BallTracker(
        color, transform, K, dist, sigma_px=args.ball_sigma_px,
        sigma_a=args.ball_sigma_a, sigma_a_air=args.ball_sigma_a_air,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q quit, g grid, t trails, m mask, w wall/floor, [ ] fx, r reset, s save",
    )
    # -- tag-specific --------------------------------------------------
    ap.add_argument("--tag-size", type=float, default=0.080,
                    help="black-square width in metres (default 0.080)")
    ap.add_argument("--heading-offset", type=float, default=0.0,
                    help="degrees added to tag theta, if not mounted upright")
    ap.add_argument("--tag-sigma-a", type=float, default=1.0,
                    help="tag filter process noise, m/s^2 (raise for a fast robot)")
    ap.add_argument("--tag-sigma-alpha", type=float, default=180.0,
                    help="tag filter process noise for turning, deg/s^2")
    # -- ball-specific ---------------------------------------------------
    ap.add_argument("--ball-profile", default="test",
                    help="named color profile from calib/ball_color.json "
                         "(default 'test'; make one with 'uv run calibrate-ball')")
    ap.add_argument("--ball-radius", type=float, default=None,
                    help="override the profile's ball RADIUS, in metres")
    ap.add_argument("--ball-sigma-a", type=float, default=3.0,
                    help="ball filter process noise while grounded, m/s^2")
    ap.add_argument("--ball-sigma-a-air", type=float, default=10.0,
                    help="ball filter process noise while airborne, m/s^2")
    ap.add_argument("--ball-sigma-px", type=float, default=1.5,
                    help="assumed pixel noise on the ball centre")
    # -- shared: field / camera -----------------------------------------
    ap.add_argument("--field", type=float, nargs=2, metavar=("W", "H"),
                    default=[1.2, 0.8], help="field size in metres (default 1.2 0.8)")
    ap.add_argument("--mode", choices=["wall", "floor"], default="floor",
                    help="floor is the physically meaningful one for a robot+ball rig; "
                         "wall is a hand-held test harness (default floor)")
    ap.add_argument("--distance", type=float, default=1.8,
                    help="wall mode: metres from camera to the field plane")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="wall mode: rotate the field about vertical, degrees")
    ap.add_argument("--pitch-wall", type=float, default=0.0,
                    help="wall mode: tip the field top-away, degrees")
    ap.add_argument("--cam-height", type=float, default=1.5,
                    help="floor mode: camera height above the field, metres")
    ap.add_argument("--pitch", type=float, default=90.0,
                    help="floor mode: camera downward tilt from horizontal, degrees "
                         "(90 = straight down, directly over the field centre)")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=None,
                    help="assumed horizontal FOV in degrees, if not calibrated")
    ap.add_argument("--field-pose", type=str, default=None,
                    help="measured field pose to load (default calib/field_pose.json "
                         "at the repo root, written by calibrate-field)")
    ap.add_argument("--synthetic-field", action="store_true",
                    help="use the made-up field from --cam-height/--pitch even if a "
                         "measured one is saved")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the auto-exposure / auto-WB lock the ball tracker wants "
                         "(not recommended: a camera left on auto re-exposes whenever "
                         "something bright moves through frame, and the ball's colour "
                         "drifts out from under its profile)")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--display-width", type=int, default=1600,
                    help="shrink the window if the composite is wider than this")
    ap.add_argument("--print-poses", action="store_true",
                    help="also stream tag id/x/y/theta to stdout")
    ap.add_argument("--print-states", action="store_true",
                    help="also stream ball x/y/z to stdout")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return
    if args.list_profiles:
        profiles = load_all()
        print(f"{color_file()}: {', '.join(sorted(profiles)) or '(none yet)'}")
        return

    color = load_profile(args.ball_profile)
    if args.ball_radius is not None:
        color.radius_m = float(args.ball_radius)

    field = Field(args.field[0], args.field[1])
    mode = args.mode
    pose_path = Path(args.field_pose) if args.field_pose else field_pose_path()
    using_calibrated = pose_path.exists() and not args.synthetic_field
    if using_calibrated:
        transform = ReferenceTagFieldTransform.load(pose_path)
        print(f"loaded calibrated field pose from {pose_path}")
    else:
        transform = build_transform(args, field, mode)

    cap = open_camera(args.camera, args.width, args.height)
    locked = False
    if not args.no_lock:
        # Let the camera settle on the scene before freezing it, or the lock
        # freezes whatever exposure it happened to open with. Tag detection
        # does not care about this either way (it thresholds greyscale), so
        # this is purely for the ball's color profile to hold still.
        for _ in range(20):
            cap.read()
        print("camera lock: " + ", ".join(lock_camera(cap)))
        locked = True

    try:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera opened but the first frame failed")
        h, w = frame.shape[:2]

        if args.hfov is not None:
            K = intr.from_fov(w, h, args.hfov)
            dist = np.zeros(5)
            intr_source = f"assumed {args.hfov:g} deg HFOV"
        else:
            K, dist, intr_source = intr.load(w, h)

        tag_detector = pa.Detector(families="tag36h11", nthreads=4, quad_decimate=1.0,
                                   decode_sharpening=0.25)
        tag_tracker = TagTracker(field, sigma_a=args.tag_sigma_a,
                                 sigma_alpha_deg=args.tag_sigma_alpha)
        # The frame handed to BallTracker is the whole-frame-undistorted one
        # below (when there is real distortion to correct), so it must not
        # also undistort per-point internally -- that would double-correct.
        # dist=None here is deliberate, not a placeholder.
        ball_tracker = _new_ball_tracker(args, color, transform, K, None)

        plan = PlanView(field, height=h, mode=mode)
        tag_trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TAG_TRAIL_LEN))
        ball_trail: deque = deque(maxlen=BALL_TRAIL_LEN)

        show_grid, show_trails, show_mask = True, True, True
        fps, last_t = 0.0, time.perf_counter()

        # Whole-frame undistortion: the tag detector searches the entire
        # image, so it (and the overlay drawn on top of it) need to share one
        # undistorted pinhole space -- see vision_core.intrinsics.undistort_maps.
        undistort_maps = intr.undistort_maps(K, dist, w, h)

        print(f"\nfield {field.width:g} x {field.height:g} m, {transform.source}")
        print(f"tag size {args.tag_size * 1000:.1f} mm, "
              f"ball '{color.name}' r={color.radius_m * 1000:.0f} mm")
        print(f"intrinsics: {intr_source}")
        print("window open -- q or Esc to quit\n")

        win = "combined tracking"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped returning frames")
                break
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)

            if undistort_maps is not None:
                frame = cv2.remap(frame, *undistort_maps, cv2.INTER_LINEAR)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            with suppressed_stderr():
                dets = tag_detector.detect(grey, estimate_tag_pose=True,
                                           camera_params=cam_params, tag_size=args.tag_size)
            tag_poses = [
                tag_field_pose(d, transform, field, args.heading_offset) for d in dets
            ]
            tag_states = tag_tracker.update(tag_poses, dt)

            ball_state = ball_tracker.update(frame, dt)
            if ball_state is not None and ball_state.visible:
                ball_trail.append((ball_state.x, ball_state.y))

            draw_field(frame, field, transform, K, show_grid)
            for d, pose in zip(dets, tag_poses):
                draw_tag_marks(frame, d, pose, color_for(pose.tag_id))
            for s in tag_states:
                if s.visible:
                    tag_trails[s.tag_id].append((s.x, s.y))
            draw_ball_marks(frame, ball_tracker, ball_state, show_mask)
            draw_combined_readout(frame, tag_states, ball_state, field, transform, K,
                                  args.tag_size, color.name, color.radius_m, intr_source, fps)

            if args.print_poses:
                for s in tag_states:
                    print(f"tag\t{s.tag_id}\t{s.x:.4f}\t{s.y:.4f}\t{s.theta_deg:.2f}",
                          flush=True)
            if args.print_states and ball_state is not None:
                print(f"ball\t{ball_state.x:.4f}\t{ball_state.y:.4f}\t{ball_state.z:.4f}",
                      flush=True)

            panel = draw_combined_plan(plan, tag_states, tag_trails, ball_state,
                                       ball_trail, transform, show_trails)
            composite = np.hstack([frame, panel])
            if composite.shape[1] > args.display_width:
                k = args.display_width / composite.shape[1]
                composite = cv2.resize(composite, None, fx=k, fy=k,
                                       interpolation=cv2.INTER_AREA)
            cv2.imshow(win, composite)

            key = read_key()
            if key in (ord("q"), 27):
                break
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("t"):
                show_trails = not show_trails
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r"):
                tag_trails.clear()
                ball_trail.clear()
                tag_tracker.reset()
            elif key in (ord("["), ord("]")):
                K = K.copy()
                K[0, 0] *= 0.98 if key == ord("[") else 1.02
                K[1, 1] = K[0, 0]
                intr_source = "hand-tuned"
                undistort_maps = intr.undistort_maps(K, dist, w, h)
                ball_tracker = _new_ball_tracker(args, color, transform, K, None)
                print(f"fx {K[0, 0]:.1f}  -> {intr.hfov_of(K, w):.1f} deg HFOV")
            elif key == ord("w"):
                if using_calibrated:
                    print(f"field is calibrated ({transform.source}); "
                          "pass --synthetic-field to demo the made-up field instead")
                else:
                    mode = "floor" if mode == "wall" else "wall"
                    transform = build_transform(args, field, mode)
                    plan = PlanView(field, height=h, mode=mode)
                    tag_trails.clear()
                    ball_trail.clear()
                    tag_tracker.reset()
                    ball_tracker = _new_ball_tracker(args, color, transform, K, None)
                    print(f"transform: {transform.source}")
            elif key == ord("s"):
                path = intr.save(K, w, h, intr_source, dist=dist)
                print(f"saved {path}")
                if not np.any(dist):
                    print("(fx only, no lens distortion -- run calibrate-camera for that)")

    finally:
        if locked:
            print("camera unlock: " + ", ".join(unlock_camera(cap)))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
