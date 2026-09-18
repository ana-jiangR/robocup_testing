"""Tag readings over time: velocity, turn rate, and coasting through dropouts.

tag_field_pose() turns one frame into one TagFieldPose, and nothing there
remembers the previous frame. This is the layer that does: one small Kalman
filter per tag id (vision_core.kalman), fed those poses as measurements.

Why bother, when the pose is measured directly each frame? Two reasons.

- Velocity. Differencing consecutive poses gives noise the size of the pose
  jitter divided by the frame time: a few mm of jitter at 30 fps is 0.1 m/s of
  pure noise, about as fast as a robot moves. The filter averages it out.
- Dropouts. A tag blurred by motion or briefly covered stops being detected
  for a few frames. The filter coasts on its velocity and reports the track as
  not visible, instead of having it vanish and reappear.

A robot never leaves the floor and gravity does not apply, so there is none of
the ball tracker's regime switching here. The one wrinkle is heading, which
wraps at 180 degrees; the filter is told which state is an angle and wraps the
residual itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vision_core.field import Field
from vision_core.kalman import KalmanFilter, cv_process_noise, cv_transition

from .pose import TagFieldPose

#: Below this speed the direction of travel is noise, not a direction.
SPEED_EPS = 0.05


@dataclass(frozen=True)
class TagFieldState:
    """Where one tag is and how it is moving, in field coordinates.

    The filtered counterpart of TagFieldPose, shaped like ball_tracking's
    BallFieldState. theta_deg is still the heading read off the tag, now
    smoothed; direction_deg is the direction of *travel*, which a robot driving
    sideways does not share with its heading.
    """

    tag_id: int
    x: float  # metres along field +X
    y: float  # metres along field +Y
    theta_deg: float  # heading, -180..180
    vx: float  # metres/second along field +X
    vy: float  # metres/second along field +Y
    speed: float  # in-plane speed, m/s
    direction_deg: float  # direction of travel, -180..180; 0 when not moving
    omega_deg: float  # turn rate, degrees/second, counter-clockwise positive
    inside: bool  # within the field rectangle
    visible: bool  # seen this frame; False while coasting on the prediction
    age: float  # seconds since the last accepted sighting
    off_plane_m: float  # last measured height off the field plane, diagnostic


class TagFilter:
    """State [x, y, theta, vx, vy, omega]: metres, radians, seconds."""

    def __init__(
        self,
        pose: TagFieldPose,
        sigma_a: float,
        sigma_alpha: float,
        gate: float,
        max_coast_s: float,
    ) -> None:
        x0 = [pose.x, pose.y, np.radians(pose.theta_deg), 0.0, 0.0, 0.0]
        P0 = np.diag([0.02, 0.02, np.radians(5.0), 0.5, 0.5, np.radians(180.0)]) ** 2
        self.kf = KalmanFilter(x0, P0, gate=gate, angle_dims=(2,))
        self.sigma = np.array([sigma_a, sigma_a, sigma_alpha], np.float64)
        self.max_coast_s = float(max_coast_s)
        self.age = 0.0
        self.off_plane_m = float(pose.off_plane_m)

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self.kf.predict(cv_transition(dt, 3), cv_process_noise(dt, self.sigma))
        self.age += dt

    def update(self, pose: TagFieldPose, sigma_xy: float, sigma_theta: float) -> bool:
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.diag([sigma_xy**2, sigma_xy**2, sigma_theta**2])
        z = [pose.x, pose.y, np.radians(pose.theta_deg)]
        ok = self.kf.update(z, H, R)
        if ok:
            self.age = 0.0
            self.off_plane_m = float(pose.off_plane_m)
        return ok

    def lost(self) -> bool:
        return self.age > self.max_coast_s

    def state(self, tag_id: int, field: Field, visible: bool) -> TagFieldState:
        x = self.kf.x
        vx, vy = float(x[3]), float(x[4])
        speed = float(np.hypot(vx, vy))
        direction = float(np.degrees(np.arctan2(vy, vx))) if speed > SPEED_EPS else 0.0
        return TagFieldState(
            tag_id=tag_id,
            x=float(x[0]),
            y=float(x[1]),
            theta_deg=float(np.degrees(x[2])),
            vx=vx,
            vy=vy,
            speed=speed,
            direction_deg=direction,
            omega_deg=float(np.degrees(x[5])),
            inside=field.contains(float(x[0]), float(x[1])),
            visible=visible,
            age=float(self.age),
            off_plane_m=self.off_plane_m,
        )


class TagTracker:
    """Poses in, states out, one filter per tag id. Hand it a *measured* dt."""

    def __init__(
        self,
        field: Field,
        sigma_a: float = 1.0,
        sigma_alpha_deg: float = 180.0,
        sigma_xy: float = 0.01,
        sigma_theta_deg: float = 2.0,
        gate: float = 9.0,
        max_coast_s: float = 0.5,
    ) -> None:
        self.field = field
        #: Process noise: how hard a robot may accelerate (m/s^2) and turn
        #: (deg/s^2) between frames without the filter treating it as noise.
        self.sigma_a = float(sigma_a)
        self.sigma_alpha = float(np.radians(sigma_alpha_deg))
        #: Measurement noise of one pose: position (m) and heading (rad).
        self.sigma_xy = float(sigma_xy)
        self.sigma_theta = float(np.radians(sigma_theta_deg))
        self.gate = float(gate)
        self.max_coast_s = float(max_coast_s)
        self.filters: dict[int, TagFilter] = {}

    def reset(self) -> None:
        self.filters.clear()

    def _new(self, pose: TagFieldPose) -> TagFilter:
        return TagFilter(pose, self.sigma_a, self.sigma_alpha, self.gate, self.max_coast_s)

    def update(self, poses: list[TagFieldPose], dt: float) -> list[TagFieldState]:
        for f in self.filters.values():
            f.predict(dt)

        seen: dict[int, bool] = {}
        for pose in poses:
            f = self.filters.get(pose.tag_id)
            if f is None:
                self.filters[pose.tag_id] = self._new(pose)
                seen[pose.tag_id] = True
                continue
            ok = f.update(pose, self.sigma_xy, self.sigma_theta)
            if not ok and f.lost():
                # Gated out for longer than a tag could plausibly be hidden:
                # the track is stale (the robot was picked up and moved), so
                # restart on what we see rather than coast forever.
                self.filters[pose.tag_id] = self._new(pose)
                ok = True
            seen[pose.tag_id] = ok

        for tid in [t for t, f in self.filters.items() if f.lost() and t not in seen]:
            del self.filters[tid]

        return [
            f.state(tid, self.field, seen.get(tid, False))
            for tid, f in sorted(self.filters.items())
        ]
