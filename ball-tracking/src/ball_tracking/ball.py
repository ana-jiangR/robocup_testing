"""Finding the ball, putting it in field metres, and filtering it into a velocity.

    mask      color likelihood -> candidates -> the best one   (ball_color.py)
    geometry  pixel + radius    -> a point in field metres      (here)
    filter    a sequence of those -> position and velocity      (here)

A ball is not a tag: one point and one scalar, and no orientation at all. So
direction means direction of *travel*, from the filter, not heading.

Two things here are load-bearing and easy to get wrong.

**The plane is at z = radius, not z = 0.** The image gives you the ball's centre,
which floats one radius above the surface. Intersecting at z = 0 overshoots by
radius / tan(elevation of the view ray): zero directly under an overhead camera,
~10 mm at the far corner of a 1.2 x 0.8 m field from 1.5 m up, ~3 cm across the
whole field for a camera tilted to 35 degrees. A systematic bias in a fixed
direction, so unlike noise it never averages out.

**An airborne ball still yields a plausible ground position** -- the ray simply
carries on to the floor. Nothing looks wrong; it is just wrong, and sliding fast.
The cross-check is free, because apparent radius is a second, independent ranger.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from vision_core.field import CameraFieldTransform, Field
from vision_core.kalman import KalmanFilter, cv_process_noise, cv_transition

from .ball_color import BallColor

#: Below this speed the direction of travel is noise, not a direction.
SPEED_EPS = 0.05


# --------------------------------------------------------------------------
# Output contract. Downstream code reads these fields; keep them stable.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BallFieldState:
    """Where the ball is and where it is going, in field coordinates.

    Deliberately shaped like tag_tracking.TagFieldPose. The difference: a tag's
    theta_deg is a heading read off the image, while direction_deg here is the
    direction of travel, estimated by the filter. A stationary ball has none.
    """

    x: float  # metres along field +X
    y: float  # metres along field +Y
    z: float  # metres along the field normal; == radius when grounded
    vx: float  # metres/second along field +X
    vy: float  # metres/second along field +Y
    speed: float  # in-plane speed, m/s
    direction_deg: float  # direction of travel about the field normal, -180..180
    grounded: bool  # touching the field, so x/y are the precise ray/plane answer
    visible: bool  # seen in this frame; False means the filter is coasting
    age: float  # seconds since the ball was last actually seen

    def inside(self, field: Field) -> bool:
        return field.contains(self.x, self.y)


# --------------------------------------------------------------------------
# Blob detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BallBlob:
    """One candidate, in pixels."""

    u: float
    v: float
    radius_px: float
    area_px: float
    circularity: float  # 4*pi*A / P^2, 1.0 for a perfect circle
    fill: float  # contour area / min-enclosing-circle area
    score: float


class BallDetector:
    """Color mask -> the single best ball candidate in a frame.

    Scored, not filtered: a chain of hard rejects can throw away the only
    candidate in a bad frame. A partly occluded ball scores worse but still
    scores, and the filter's gate makes the final call.
    """

    def __init__(
        self,
        color: BallColor,
        min_area_px: float = 30.0,
        max_area_frac: float = 0.25,
        open_px: int = 3,
        close_px: int = 7,
    ) -> None:
        self.color = color
        self.min_area_px = min_area_px
        self.max_area_frac = max_area_frac
        # OPEN first, then CLOSE. Order matters: OPEN removes the speckle that
        # CLOSE would otherwise weld into blobs, and CLOSE then fills the holes
        # left by highlights and the ball's own shading.
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        m = self.color.mask(frame_bgr)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._close)
        return m

    def candidates(self, mask: np.ndarray) -> list[BallBlob]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = self.max_area_frac * mask.size
        out: list[BallBlob] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area_px or area > max_area:
                continue
            perim = float(cv2.arcLength(c, True))
            if perim <= 0.0:
                continue
            circularity = float(4.0 * np.pi * area / (perim * perim))
            # The min-enclosing circle, not the moment centroid. Occlusion eats
            # the near edge of a blob and drags the centroid towards whatever is
            # left; the enclosing circle is pinned by the extremes that survive,
            # so it stays put. Same reason it is the right radius to measure.
            (u, v), r = cv2.minEnclosingCircle(c)
            if r <= 0.0:
                continue
            fill = float(area / (np.pi * r * r))
            out.append(
                BallBlob(float(u), float(v), float(r), area, circularity, fill, 0.0)
            )
        return out

    def best(
        self,
        mask: np.ndarray,
        predicted_px: tuple[float, float] | None = None,
        predicted_radius_px: float | None = None,
    ) -> BallBlob | None:
        """The highest-scoring candidate, or None if nothing plausible is there.

        `predicted_px` comes from the filter. Being near the prediction is
        evidence, but only ever a *bonus* -- never a gate -- because the
        prediction is wrong exactly when the ball does something unexpected,
        which is the moment you least want the detector to go blind.
        """
        cands = self.candidates(mask)
        if not cands:
            return None

        scored: list[BallBlob] = []
        for c in cands:
            # A ball is round: circularity and fill both say so, from different
            # directions (perimeter raggedness vs. area deficit).
            shape = 0.6 * min(c.circularity, 1.0) + 0.4 * min(c.fill, 1.0)
            score = shape
            # Mild preference for bigger blobs, to break ties against specks.
            score += 0.15 * float(np.tanh(c.area_px / 400.0))
            if predicted_px is not None:
                d = float(np.hypot(c.u - predicted_px[0], c.v - predicted_px[1]))
                # Scale by the ball's own size: "two radii away" means the same
                # thing at 1 m and at 3 m, where "40 pixels away" does not.
                span = max(predicted_radius_px or c.radius_px, 4.0)
                score += 0.5 * float(np.exp(-0.5 * (d / (3.0 * span)) ** 2))
            scored.append(
                BallBlob(c.u, c.v, c.radius_px, c.area_px, c.circularity, c.fill, score)
            )
        return max(scored, key=lambda b: b.score)


# --------------------------------------------------------------------------
# Pixel -> field metres
# --------------------------------------------------------------------------


def ray_direction(
    u: float,
    v: float,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """The camera-frame direction of the ray through pixel (u, v).

    Normalised so z == 1, which makes the ray parameter equal to depth in metres
    -- directly comparable with the radius estimate, no conversion between them.
    Distortion is undone first when there are real coefficients for it.
    """
    if dist is not None and np.any(dist):
        pt = cv2.undistortPoints(
            np.array([[[float(u), float(v)]]], np.float64), K, np.asarray(dist), P=K
        )
        u, v = float(pt[0, 0, 0]), float(pt[0, 0, 1])
    d = np.linalg.inv(K) @ np.array([float(u), float(v), 1.0])
    return d / d[2]  # z == 1, so s == depth


def plane_intersection(
    d_cam: np.ndarray,
    transform: CameraFieldTransform,
    plane_z: float,
) -> tuple[np.ndarray, float] | None:
    """Where the view ray meets the field plane at height `plane_z`.

    Returns (point_in_field, depth_m), or None if the ray runs parallel to the
    plane or meets it behind the camera.

    `plane_z` is the ball's radius, not zero: the image gives the ball's centre,
    and the centre of a resting ball sits one radius above the surface.
    """
    a = transform.direction_to_field(d_cam)[0]
    b = transform.camera_origin_in_field()
    if abs(a[2]) < 1e-9:
        return None  # ray parallel to the plane; no intersection
    s = float((plane_z - b[2]) / a[2])
    if s <= 0.0:
        return None  # the plane is behind the camera along this ray
    return b + s * a, s


def depth_from_radius(radius_px: float, ball_radius_m: float, fx: float) -> float:
    """Range implied by apparent size: Z = fx * R / r_px.

    Independent of the ray/plane answer, which is the point: two estimates from
    different physics disagree when an assumption breaks, and the assumption here
    is that the ball is on the ground. Weak, though -- the error grows as Z^2.
    Enough to notice the ball has left the ground; not enough to say where it
    will land.
    """
    if radius_px <= 0.0:
        return float("inf")
    return float(fx * ball_radius_m / radius_px)


def point_at_depth(
    d_cam: np.ndarray, depth_m: float, transform: CameraFieldTransform
) -> np.ndarray:
    """The 3D field point `depth_m` along the ray. For an airborne ball."""
    return transform.camera_to_field(depth_m * d_cam)[0]


def ray_covariance(
    d_cam: np.ndarray,
    transform: CameraFieldTransform,
    sigma_perp: float,
    sigma_depth: float,
) -> np.ndarray:
    """Covariance of a point located along a view ray, in field axes.

    The error on a radius-ranged point is a long thin cigar pointing down the
    ray: millimetres across, tens of centimetres long. So build the ellipsoid in
    camera axes and rotate it, rather than handing the filter a field-axis
    diagonal -- that would claim millimetre precision in field x and y, which is
    a claim about the floor rather than the ray. The filter would then over-trust
    each airborne measurement, collapse its covariance, and gate out the next one
    for disagreeing with the over-confident estimate it just formed.
    """
    d = np.asarray(d_cam, np.float64).reshape(3)
    d_hat = d / np.linalg.norm(d)
    cov_cam = sigma_perp**2 * np.eye(3) + (
        sigma_depth**2 - sigma_perp**2
    ) * np.outer(d_hat, d_hat)
    # p_field = R.T @ p_camera + const, so Cov_field = R.T @ Cov_camera @ R.
    return transform.R.T @ cov_cam @ transform.R


def depth_sigma(depth_m: float, radius_px: float, sigma_radius_px: float = 1.5) -> float:
    """How badly the radius-based range is known, in metres.

    Z = fx*R/r, so dZ/dr = -Z/r, and a pixel of radius error is worth (Z/r)
    metres. At the defaults -- a 40 mm ball on 720p at 60 deg HFOV, so fx ~1109
    px -- the ball is 15 px in radius at 1.5 m and 11 px at 2 m, giving about
    +/-0.15 m and +/-0.27 m. Since r itself falls off as 1/Z, the error grows
    as Z^2.
    """
    if radius_px <= 0.0:
        return float("inf")
    return float(depth_m / radius_px * sigma_radius_px)


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------


class BallFilter:
    """A 6-state Kalman filter: [x, y, z, vx, vy, vz] in field metres.

    The matrix algebra is vision_core.kalman; this class is the ball physics on
    top of it -- swapping between two measurement models, which is the whole
    trick here:

    GROUNDED   the ray/plane (x, y). Real geometry against a known plane, so
               small R. z is pinned to the radius and vz to zero.
    AIRBORNE   the radius-derived 3D point, with a much larger R and a much
               larger process noise, and no gravity.

    No gravity is deliberate. A parabola is the right model for a ball in
    flight and the wrong one for a ball in a hand, and the hand is how this
    gets tested -- the README says to lift the ball, and wall mode exists for
    holding it up. With g in the prediction and a near ball's range known to
    a centimetre, the gate rejected the "it did not fall" measurement within
    two frames, the state fell away at 9.8 m/s^2 unopposed, and the track
    timed out and restarted, over and over. Nor does the model buy anything:
    the radius ranger is too weak to see a parabola over the handful of frames
    a bounce lasts. So airborne acceleration is process noise, sized to cover
    gravity, and the answer stays "airborne yes/no", which is all the sensor
    can give.

    Beyond smoothing this buys velocity worth having (differencing raw positions
    gives ~0.1 m/s of pure noise at 2 m, as fast as the ball rolls), coasting
    through motion-blur dropouts, and a Mahalanobis gate that rejects a false
    positive across the room for being impossible rather than unlikely.
    """

    def __init__(
        self,
        sigma_a: float = 3.0,
        sigma_a_air: float = 10.0,
        gate: float = 9.0,
        max_coast_s: float = 0.35,
    ) -> None:
        #: Process noise as an acceleration, m/s^2. 2-4 suits a ball rolling on
        #: carpet, where the "acceleration" being modelled is mostly the surface
        #: pushing back unevenly. Raise it if the ball gets kicked a lot.
        self.sigma_a = float(sigma_a)
        #: The same, while airborne. Must cover gravity, a bounce, or a hand
        #: changing its mind, since none of those is in the prediction.
        self.sigma_a_air = float(sigma_a_air)
        self.max_coast_s = float(max_coast_s)

        #: Mahalanobis gate. ~9 is the 99% point for 2 degrees of freedom.
        self.kf = KalmanFilter(np.zeros(6), np.eye(6) * 1e3, gate=gate)
        self.initialised = False
        self.grounded = True
        self.age = 0.0

    # -- prediction --------------------------------------------------------

    def predict(self, dt: float) -> None:
        if not self.initialised or dt <= 0.0:
            return
        sigma = self.sigma_a if self.grounded else self.sigma_a_air
        self.kf.predict(cv_transition(dt, 3), cv_process_noise(dt, [sigma] * 3))
        self.age += dt

    # -- update ------------------------------------------------------------

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> bool:
        """One Kalman update, refused if the measurement fails the gate."""
        ok = self.kf.update(z, H, R)
        if ok:
            self.age = 0.0
        return ok

    def start(self, point: np.ndarray, grounded: bool) -> None:
        """Seed the filter from one measurement, with no velocity assumed."""
        self.kf.x = np.concatenate([np.asarray(point, np.float64).reshape(3), np.zeros(3)])
        self.kf.P = np.diag([0.05, 0.05, 0.05, 1.0, 1.0, 1.0]) ** 2
        self.initialised = True
        self.grounded = grounded
        self.age = 0.0

    def update_grounded(
        self, xy: np.ndarray, sigma_xy: float, plane_z: float, jump: float = 0.0
    ) -> bool:
        """Measure the precise in-plane position; pin the out-of-plane state.

        `jump` is how far this measurement is from the current estimate, and
        only matters when the regime is changing -- see release().
        """
        if not self.grounded:
            # Landing is a regime change too, and the estimate we are landing
            # with came from the weak radius ranger. Loosen before believing the
            # precise measurement, for the same reason as the other direction.
            self.release(max(float(jump), 0.10))
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * sigma_xy**2
        ok = self._update(np.asarray(xy, np.float64).reshape(2), H, R)
        self.grounded = True
        # A ball on the field is at exactly one height and is not rising. Say so
        # rather than letting z drift on process noise nobody is correcting.
        x, P = self.kf.x, self.kf.P
        x[2] = plane_z
        x[5] = 0.0
        P[2, :] = P[:, 2] = 0.0
        P[5, :] = P[:, 5] = 0.0
        P[2, 2] = 1e-6
        P[5, 5] = 1e-6
        return ok

    def release(self, sigma: float) -> None:
        """Loosen the state after the regime changed under us.

        `sigma` must be the size of the jump the new regime implies -- the gap
        between where the filter is and where the first measurement of the new
        regime says the ball is -- not the noise on that measurement. Loosened
        by the noise (millimetres for a near ball), the gate rejects a jump of
        a metre as impossible and the filter coasts until the track times out;
        loosened by the jump, the first measurement lands and the filter is
        simply re-seeded where the ball is, velocity kept.

        Applies to x and y as much as to z, which is the subtle part. While
        grounded, x and y came from the ray/plane intersection and were precise,
        so their variance shrank to nearly nothing -- and the moment the ball
        lifts, those same x and y are wrong by tens of centimetres. Skip this and
        the gate correctly rejects the correct new measurement, leaving the
        filter coasting on a confident, stale, sinking estimate.
        """
        var = max(float(sigma) ** 2, 1e-4)
        P = self.kf.P
        for i in range(3):
            P[i, i] = max(P[i, i], var)
        for i in range(3, 6):
            P[i, i] = max(P[i, i], 1.0)

    def update_airborne(
        self, point: np.ndarray, R: np.ndarray, jump: float = 0.0
    ) -> bool:
        """Measure the weak radius-derived 3D point, with honest covariance.

        `R` is the full 3x3 from ray_covariance, not a diagonal: the uncertainty
        is a cigar along the view ray, and saying so is what keeps the filter
        from over-trusting a measurement that is only precise sideways. `jump`
        is the distance from the current estimate, for release().
        """
        R = np.asarray(R, np.float64).reshape(3, 3)
        if self.grounded:
            self.grounded = False
            self.release(max(float(jump), float(np.sqrt(np.max(np.diag(R))))))
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return self._update(np.asarray(point, np.float64).reshape(3), H, R)

    # -- readout -----------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:3].copy()

    def lost(self) -> bool:
        return self.age > self.max_coast_s

    def state(self, visible: bool) -> BallFieldState:
        x = self.kf.x
        vx, vy = float(x[3]), float(x[4])
        speed = float(np.hypot(vx, vy))
        direction = float(np.degrees(np.arctan2(vy, vx))) if speed > SPEED_EPS else 0.0
        return BallFieldState(
            x=float(x[0]),
            y=float(x[1]),
            z=float(x[2]),
            vx=vx,
            vy=vy,
            speed=speed,
            direction_deg=direction,
            grounded=self.grounded,
            visible=visible,
            age=float(self.age),
        )


# --------------------------------------------------------------------------
# The whole pipeline
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BallObservation:
    """What one frame said, before the filter had an opinion about it."""

    blob: BallBlob
    ground_point: np.ndarray | None  # ray/plane answer, field metres
    ground_depth: float  # depth to the plane, metres
    radius_depth: float  # depth from apparent size, metres
    air_point: np.ndarray | None  # radius-based 3D answer, field metres
    airborne: bool
    sigma_xy: float  # in-plane sigma for the grounded measurement, metres
    sigma_z: float  # depth sigma of the radius ranger, metres
    air_cov: np.ndarray | None  # 3x3 field-frame covariance of air_point


class BallTracker:
    """Frame in, BallFieldState out. The only thing a caller must get right is
    handing over a measured dt."""

    def __init__(
        self,
        color: BallColor,
        transform: CameraFieldTransform,
        K: np.ndarray,
        dist: np.ndarray | None = None,
        sigma_px: float = 1.5,
        sigma_a: float = 3.0,
        sigma_a_air: float = 10.0,
        air_sigmas: float = 1.5,
        air_min_margin: float = 0.04,
        air_hold: int = 2,
    ) -> None:
        self.color = color
        self.detector = BallDetector(color)
        self.transform = transform
        self.K = np.asarray(K, np.float64)
        self.dist = None if dist is None else np.asarray(dist, np.float64)
        self.fx = float(self.K[0, 0])
        self.radius_m = float(color.radius_m)
        self.sigma_px = float(sigma_px)
        self.filter = BallFilter(sigma_a=sigma_a, sigma_a_air=sigma_a_air)
        #: Sigmas of depth disagreement needed to *enter* the airborne state,
        #: with a floor under it so a very close ball is not called airborne on
        #: millimetres. Leaving again needs only 0.4 of this -- see observe().
        self.air_sigmas = float(air_sigmas)
        self.air_min_margin = float(air_min_margin)
        self.air_hold = int(air_hold)
        self._air_votes = 0
        self._airborne = False

        self.last_mask: np.ndarray | None = None
        self.last_blob: BallBlob | None = None
        self.last_obs: BallObservation | None = None

    # -- geometry ----------------------------------------------------------

    def observe(self, blob: BallBlob) -> BallObservation:
        """Turn one blob into both range estimates, and decide which to believe."""
        d_cam = ray_direction(blob.u, blob.v, self.K, self.dist)

        hit = plane_intersection(d_cam, self.transform, self.radius_m)
        ground_point, ground_depth = (None, float("inf")) if hit is None else hit

        radius_depth = depth_from_radius(blob.radius_px, self.radius_m, self.fx)
        sigma_z = depth_sigma(radius_depth, blob.radius_px)

        # The cross-check. An airborne ball is nearer than the point where its
        # view ray reaches the floor, so the radius says "closer" while the plane
        # says "further".
        #
        # A Schmitt trigger, not one threshold. A single threshold cannot work
        # here: set it high enough that a shimmering radius cannot fake a bounce
        # and it is also high enough to miss a ball genuinely 20 cm off the
        # ground; set it low and the regime flaps frame to frame, which is worse
        # than either answer because it keeps swapping the measurement model out
        # from under the filter. So: a high bar to *become* airborne, a low bar
        # to stop being airborne. Once the ball has demonstrably left the ground,
        # believing it is still up there is the cheaper mistake.
        gap = ground_depth - radius_depth
        enter = max(self.air_sigmas * sigma_z, self.air_min_margin)
        exit_ = 0.4 * enter
        if np.isfinite(ground_depth) and np.isfinite(radius_depth):
            disagrees = gap > (exit_ if self._airborne else enter)
        else:
            disagrees = False
        self._air_votes = (
            min(self._air_votes + 1, self.air_hold)
            if disagrees
            else max(self._air_votes - 1, 0)
        )
        # Rising edge needs the full hold; falling edge only needs the votes to
        # run out, so landing is noticed promptly.
        if self._air_votes >= self.air_hold:
            self._airborne = True
        elif self._air_votes == 0:
            self._airborne = False
        airborne = self._airborne

        air_point = (
            point_at_depth(d_cam, radius_depth, self.transform)
            if np.isfinite(radius_depth)
            else None
        )
        depth = radius_depth if airborne else ground_depth
        # A pixel is worth more metres the further away the ball is.
        sigma_xy = self.sigma_px * float(depth) / self.fx if np.isfinite(depth) else 1.0

        air_cov = (
            ray_covariance(
                d_cam,
                self.transform,
                sigma_perp=self.sigma_px * float(radius_depth) / self.fx,
                sigma_depth=sigma_z,
            )
            if air_point is not None and np.isfinite(sigma_z)
            else None
        )

        return BallObservation(
            blob=blob,
            ground_point=ground_point,
            ground_depth=float(ground_depth),
            radius_depth=float(radius_depth),
            air_point=air_point,
            airborne=bool(airborne),
            sigma_xy=float(sigma_xy),
            sigma_z=float(sigma_z),
            air_cov=air_cov,
        )

    # -- prediction, in pixels, for the detector ---------------------------

    def predicted_pixel(self) -> tuple[tuple[float, float], float] | None:
        """Where the filter thinks the ball will appear, and how big. Or None."""
        if not self.filter.initialised:
            return None
        p_cam = self.transform.field_to_camera(self.filter.position)[0]
        if p_cam[2] <= 1e-3:
            return None  # behind the camera
        uv = self.K @ (p_cam / p_cam[2])
        r = self.fx * self.radius_m / float(p_cam[2])
        return (float(uv[0]), float(uv[1])), float(r)

    # -- one frame ---------------------------------------------------------

    def update(self, frame_bgr: np.ndarray, dt: float) -> BallFieldState | None:
        """Process one frame. Returns None only before the ball is ever seen.

        `dt` must be *measured*, not assumed: sample perf_counter right after
        cap.read() and pass the difference. A filter fed a nominal 1/30 while the
        camera actually delivers a jittery 24 will mis-scale every velocity it
        reports, and the error looks like drift rather than like a bug.
        """
        self.filter.predict(dt)

        pred = self.predicted_pixel()
        mask = self.detector.mask(frame_bgr)
        blob = self.detector.best(
            mask,
            predicted_px=pred[0] if pred else None,
            predicted_radius_px=pred[1] if pred else None,
        )
        self.last_mask = mask
        self.last_blob = blob

        if blob is None:
            self.last_obs = None
            if not self.filter.initialised:
                return None
            # Coast on the prediction, and say so via visible=False.
            return self.filter.state(visible=False)

        obs = self.observe(blob)
        self.last_obs = obs

        use_air = obs.airborne and obs.air_point is not None and obs.air_cov is not None
        point = obs.air_point if use_air else obs.ground_point
        if point is None:
            return self.filter.state(visible=False) if self.filter.initialised else None

        if not self.filter.initialised:
            self.filter.start(point, grounded=not use_air)
            return self.filter.state(visible=True)

        # A regime change moves the estimate by the whole ground/air gap, not
        # by a measurement's noise. Tell the filter how far, so the gate lets
        # the first measurement of the new regime in.
        jump = float(np.linalg.norm(np.asarray(point, np.float64) - self.filter.position))
        if use_air:
            accepted = self.filter.update_airborne(point, obs.air_cov, jump=jump)
        else:
            accepted = self.filter.update_grounded(
                np.asarray(point)[:2], obs.sigma_xy, self.radius_m, jump=jump
            )

        if not accepted and self.filter.lost():
            # Gated out for longer than the ball could plausibly be hidden. The
            # track is stale rather than occluded, so restart it on what we see
            # instead of coasting forever on a stale prediction.
            self.filter.start(point, grounded=not use_air)

        return self.filter.state(visible=accepted)
