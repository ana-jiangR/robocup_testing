"""Live color-based ball tracking against the virtual field.

    uv run track-ball --ball-profile test --mode floor

Camera feed on the left with the mask and the detected ball drawn on it, plan
view on the right with the ball, its velocity arrow and its trail. The field is
not real: it is whatever vision_core/field.py says it is.

Keys:
    q / Esc  quit            g  grid on/off           t  trails on/off
    m        mask overlay    w  wall / floor          r  reset trail
    [ / ]    fx -/+ 2%       s  save intrinsics       p  pause
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from vision_core import intrinsics as intr
from vision_core.camera import list_cameras, lock_camera, open_camera, read_key
from vision_core.field import (
    CameraFieldTransform,
    Field,
    ReferenceTagFieldTransform,
    SyntheticFieldTransform,
    field_pose_path,
)
from vision_core.planview import OUT_COLOR, PlanView, draw_field

from .ball import BallFieldState, BallTracker
from .ball_color import color_file, load_all, load_profile

TRAIL_LEN = 120

BALL_COLOR = (200, 80, 240)  # BGR, magenta-ish: the ball itself
AIR_COLOR = (90, 200, 255)  # amber: airborne, stop trusting the ground position
GHOST_COLOR = (120, 120, 120)  # coasting, not actually seen this frame


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def draw_ball_marks(
    frame: np.ndarray,
    tracker: BallTracker,
    state: BallFieldState | None,
    show_mask: bool,
) -> None:
    """Mask overlay, the detected circle, and the airborne flag, on the video."""
    if show_mask and tracker.last_mask is not None:
        # Tint the masked pixels rather than replacing them: you want to see both
        # what the mask caught and what it caught it *on*, because the usual
        # failure is the mask quietly eating something the same color.
        tint = np.zeros_like(frame)
        tint[:] = (120, 0, 140)
        m = tracker.last_mask.astype(bool)
        frame[m] = cv2.addWeighted(frame, 0.45, tint, 0.55, 0)[m]

    blob = tracker.last_blob
    if blob is None:
        return
    airborne = state is not None and not state.grounded
    color = AIR_COLOR if airborne else BALL_COLOR
    c = (int(round(blob.u)), int(round(blob.v)))
    cv2.circle(frame, c, int(round(blob.radius_px)), color, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, c, color, cv2.MARKER_CROSS, 12, 1)

    obs = tracker.last_obs
    if obs is not None:
        # The two range estimates, side by side. When they disagree the ball is
        # in the air, and seeing both numbers is what makes that legible rather
        # than magic.
        txt = f"plane {obs.ground_depth:.2f} m | radius {obs.radius_depth:.2f} m"
        # Clear of the circle, and haloed: this label is the same color as the
        # ball it sits next to, so over the ball itself it is invisible.
        r = int(round(blob.radius_px))
        org = (min(c[0] + r + 10, frame.shape[1] - 260), max(c[1] - r - 8, 50))
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)


def draw_plan(
    plan: PlanView,
    state: BallFieldState | None,
    trail: deque,
    transform: CameraFieldTransform,
    show_trails: bool,
) -> np.ndarray:
    """Plan view with the ball, its velocity arrow and its trail."""
    panel = plan.base()
    plan.draw_camera(panel, transform)

    if show_trails and len(trail) > 1:
        plan.draw_trail(panel, trail, BALL_COLOR)

    if state is None:
        cv2.putText(panel, "no ball yet", (plan.pad - 12, plan.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        return panel

    color = (
        GHOST_COLOR if not state.visible
        else AIR_COLOR if not state.grounded
        else BALL_COLOR
    )
    p = plan.to_px(state.x, state.y)
    if plan.on_panel(p):
        # Velocity arrow, scaled so 1 m/s reads as 20 cm on the field. This is a
        # velocity, not a heading: a ball has no orientation, and a stationary
        # ball correctly gets no arrow at all.
        if state.speed > 0.05:
            plan.draw_arrow(panel, state.x, state.y, state.direction_deg,
                            min(0.20 * state.speed, 0.45), color, 2)
        r = max(plan.metres(0.02), 5)
        cv2.circle(panel, p, r, color, -1 if state.visible else 2, cv2.LINE_AA)
        if not state.grounded:
            # A ring around an airborne ball: its plan position is the weak
            # radius-based answer, so it deserves to look different.
            cv2.circle(panel, p, r + 5, AIR_COLOR, 1, cv2.LINE_AA)
    return panel


def draw_readout(
    frame: np.ndarray,
    state: BallFieldState | None,
    tracker: BallTracker,
    field: Field,
    transform: CameraFieldTransform,
    K: np.ndarray,
    profile_name: str,
    intr_source: str,
    fps: float,
) -> None:
    """Header, and the numbers the whole exercise is about."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
    head = (
        f"{transform.source}  |  ball '{profile_name}' "
        f"r={tracker.radius_m * 1000:.0f} mm  |  "
        f"fx {K[0, 0]:.0f} ({intr.hfov_of(K, w):.0f} deg HFOV, {intr_source})"
        f"  |  {fps:4.1f} fps"
    )
    cv2.putText(frame, head, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)

    strip_h = 68
    cv2.rectangle(frame, (0, h - strip_h), (w, h), (0, 0, 0), -1)

    if state is None:
        cv2.putText(frame, "no ball detected", (10, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 140, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, "check the color profile: uv run calibrate-ball",
                    (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (120, 120, 160), 1, cv2.LINE_AA)
        return

    if not state.visible:
        status, color = f"COASTING {state.age * 1000:3.0f} ms", GHOST_COLOR
    elif not state.grounded:
        status, color = "AIRBORNE", AIR_COLOR
    else:
        status, color = "grounded", BALL_COLOR
    if not state.inside(field):
        color = OUT_COLOR

    line1 = (
        f"x {state.x:+6.3f}   y {state.y:+6.3f}   z {state.z:+6.3f}   "
        f"{'IN ' if state.inside(field) else 'OUT'}   {status}"
    )
    line2 = (
        f"vx {state.vx:+6.3f}   vy {state.vy:+6.3f}   "
        f"speed {state.speed:5.3f} m/s   dir {state.direction_deg:+7.1f} deg"
    )
    cv2.putText(frame, line1, (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                color, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                color, 1, cv2.LINE_AA)

    if not state.grounded:
        cv2.putText(frame, "airborne: ground position is NOT reliable",
                    (w - 430, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    AIR_COLOR, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------


def build_transform(args, field: Field, mode: str) -> CameraFieldTransform:
    """The one call that decides where the field is.

    Same shape as the tag tracker's, and for the same reason: swapping in
    ReferenceTagFieldTransform for the real field means changing this function
    and nothing else. The ball tracker needs the field located just as much as
    the tag tracker does.
    """
    if mode == "floor":
        return SyntheticFieldTransform.floor(field, args.cam_height, args.pitch)
    return SyntheticFieldTransform.wall(field, args.distance, args.yaw, args.pitch_wall)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q quit, g grid, t trails, m mask, w wall/floor, "
               "[ ] fx, r reset, s save, p pause",
    )
    ap.add_argument("--ball-profile", default="test",
                    help="named color profile from calib/ball_color.json "
                         "(default 'test'; make one with 'uv run calibrate-ball')")
    ap.add_argument("--ball-radius", type=float, default=None,
                    help="override the profile's ball RADIUS, in metres")
    ap.add_argument("--field", type=float, nargs=2, metavar=("W", "H"),
                    default=[1.2, 0.8], help="field size in metres (default 1.2 0.8)")
    ap.add_argument("--mode", choices=["wall", "floor"], default="floor",
                    help="floor is the physically meaningful one for a rolling "
                         "ball; wall is a free test harness for the airborne path, "
                         "since a hand-held ball is off-plane nearly always")
    ap.add_argument("--distance", type=float, default=1.8,
                    help="wall mode: metres from camera to the field plane")
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
    ap.add_argument("--sigma-a", type=float, default=3.0,
                    help="process noise as an acceleration, m/s^2. 2-4 suits a "
                         "ball rolling on carpet; raise it if the ball gets "
                         "kicked hard and the filter lags")
    ap.add_argument("--sigma-px", type=float, default=1.5,
                    help="assumed pixel noise on the ball centre")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=None,
                    help="assumed horizontal FOV in degrees, if not calibrated")
    ap.add_argument("--field-pose", type=str, default=None,
                    help="measured field pose to load (default calib/field_pose.json "
                         "at the repo root, written by calibrate-field in tag-tracking)")
    ap.add_argument("--synthetic-field", action="store_true",
                    help="use the made-up field from --cam-height/--pitch even if a "
                         "measured one is saved")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the auto-exposure / auto-WB lock. Not recommended: "
                         "a camera left on auto re-exposes whenever something "
                         "bright moves through frame, and the ball's color "
                         "drifts out from under the profile")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--display-width", type=int, default=1600,
                    help="shrink the window if the composite is wider than this")
    ap.add_argument("--print-states", action="store_true",
                    help="also stream x/y/vx/vy/grounded to stdout")
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
    # The measured field pose, if calibrate-field has produced one, else the
    # made-up rig. Same file and same rule as the tag tracker; --mode still
    # says which way gravity points, since the pose file does not know.
    pose_path = Path(args.field_pose) if args.field_pose else field_pose_path()
    using_calibrated = pose_path.exists() and not args.synthetic_field
    if using_calibrated:
        transform = ReferenceTagFieldTransform.load(pose_path)
        print(f"loaded calibrated field pose from {pose_path}")
    else:
        transform = build_transform(args, field, mode)

    cap = open_camera(args.camera, args.width, args.height)
    if not args.no_lock:
        # Let the camera settle on the scene before freezing it, or we lock in
        # whatever exposure it happened to open with.
        for _ in range(20):
            cap.read()
        print("camera lock: " + ", ".join(lock_camera(cap)))

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

    tracker = BallTracker(color, transform, K, dist, mode=mode,
                          sigma_px=args.sigma_px, sigma_a=args.sigma_a)
    plan = PlanView(field, height=h, mode=mode)
    trail: deque = deque(maxlen=TRAIL_LEN)

    show_grid, show_trails, show_mask, paused = True, True, True, False
    fps = 0.0
    last_t = time.perf_counter()

    print(f"\nfield {field.width:g} x {field.height:g} m, {transform.source}")
    print(f"ball '{color.name}' radius {color.radius_m * 1000:.0f} mm, "
          f"peak hue H~{color.hue_peak()}")
    print(f"intrinsics: {intr_source}")
    print("window open -- q or Esc to quit\n")

    win = "ball tracking"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state: BallFieldState | None = None

    while True:
        if not paused:
            ok, frame = cap.read()
            # Sample the clock right after the grab, and hand the filter the dt
            # it actually got. A filter fed a nominal 1/30 while the camera
            # delivers a jittery 24 mis-scales every velocity it reports, and the
            # error looks like drift rather than like a bug.
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if not ok:
                print("camera stopped returning frames")
                break

            state = tracker.update(frame, dt)
            if state is not None and state.visible:
                trail.append((state.x, state.y))
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)

            if args.print_states and state is not None:
                print(f"{state.x:.4f}\t{state.y:.4f}\t{state.z:.4f}\t"
                      f"{state.vx:.4f}\t{state.vy:.4f}\t{int(state.grounded)}\t"
                      f"{int(state.visible)}", flush=True)

        shown = frame.copy()
        draw_field(shown, field, transform, K, show_grid)
        draw_ball_marks(shown, tracker, state, show_mask)
        draw_readout(shown, state, tracker, field, transform, K, color.name,
                     intr_source, fps)
        if paused:
            cv2.putText(shown, "PAUSED", (w // 2 - 60, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 200, 255), 2, cv2.LINE_AA)

        panel = draw_plan(plan, state, trail, transform, show_trails)
        composite = np.hstack([shown, panel])
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
        elif key == ord("p"):
            paused = not paused
            last_t = time.perf_counter()  # do not bill the pause to the filter
        elif key == ord("r"):
            trail.clear()
        elif key in (ord("["), ord("]")):
            K = K.copy()
            K[0, 0] *= 0.98 if key == ord("[") else 1.02
            K[1, 1] = K[0, 0]
            intr_source = "hand-tuned"
            tracker = BallTracker(color, transform, K, dist, mode=mode,
                                  sigma_px=args.sigma_px, sigma_a=args.sigma_a)
            trail.clear()
            print(f"fx {K[0, 0]:.1f}  -> {intr.hfov_of(K, w):.1f} deg HFOV")
        elif key == ord("w") and using_calibrated:
            print(f"field is calibrated ({transform.source}); "
                  "pass --synthetic-field to demo the made-up wall/floor instead")
        elif key == ord("w"):
            mode = "floor" if mode == "wall" else "wall"
            transform = build_transform(args, field, mode)
            plan = PlanView(field, height=h, mode=mode)
            tracker = BallTracker(color, transform, K, dist, mode=mode,
                                  sigma_px=args.sigma_px, sigma_a=args.sigma_a)
            trail.clear()
            print(f"transform: {transform.source}")
        elif key == ord("s"):
            # Keep whatever distortion was loaded: saving K alone would
            # silently zero a calibrate-camera result.
            path = intr.save(K, w, h, intr_source, dist=dist)
            print(f"saved {path}")
            if not np.any(dist):
                print("(fx only, no lens distortion -- run calibrate-camera for that)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
