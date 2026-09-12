"""Field geometry and the camera->field transform.

THIS FILE IS THE SWAP POINT.

Everything downstream (tracking, drawing, the printed x/y/theta) depends only on
the small contract in this file:

    CameraFieldTransform  -- a rigid pose: field coordinates -> camera coordinates
    TagFieldPose          -- the output record: tag_id, x, y, theta_deg, ...

Today the pose is invented (SyntheticFieldTransform): we simply declare that a
1.2 x 0.8 m rectangle floats some distance in front of the lens. Nothing
physical marks it.

For RoboCup the pose will be *measured* from four reference AprilTags at known
field positions. That is a new subclass (see ReferenceTagFieldTransform below)
which fills in the same R and t. No other file changes, and the output format
does not change.

Coordinate conventions
----------------------
Camera frame (OpenCV): +X right, +Y down, +Z forward out of the lens. Metres.

Field frame: origin at one corner of the rectangle, +X along `width`,
+Y along `height`, +Z along the field's surface normal. Right-handed.
A tag at (0, 0) sits over the origin corner; (width, height) is the far corner.

theta is the tag's rotation about the field normal +Z, in degrees, measured from
field +X, counter-clockwise positive, and 0 when the tag is upright.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Output contract. Downstream code reads these fields; keep them stable.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TagFieldPose:
    """Where one tag is, in field coordinates."""

    tag_id: int
    x: float  # metres from the origin corner, along field +X
    y: float  # metres from the origin corner, along field +Y
    theta_deg: float  # rotation about the field normal, -180..180
    off_plane_m: float  # signed distance off the field plane (along field +Z)
    inside: bool  # within the rectangle in both x and y
    decision_margin: float  # detector confidence, higher is better

    def as_tuple(self) -> tuple[int, float, float, float]:
        """The frozen public format: (id, x, y, theta_deg)."""
        return (self.tag_id, self.x, self.y, self.theta_deg)


@dataclass(frozen=True)
class Field:
    """A rectangle, in metres."""

    width: float = 1.2  # extent along field +X
    height: float = 0.8  # extent along field +Y

    def corners(self) -> np.ndarray:
        """The 4 corners in field coords, (4,3), starting at the origin corner."""
        w, h = self.width, self.height
        return np.array(
            [[0.0, 0.0, 0.0], [w, 0.0, 0.0], [w, h, 0.0], [0.0, h, 0.0]],
            dtype=np.float64,
        )

    def grid(self, step: float = 0.2) -> list[np.ndarray]:
        """Interior grid lines as a list of (2,3) segments, for drawing."""
        segs: list[np.ndarray] = []
        for i in range(1, int(round(self.width / step))):
            gx = i * step
            if gx < self.width:
                segs.append(np.array([[gx, 0, 0], [gx, self.height, 0]], np.float64))
        for i in range(1, int(round(self.height / step))):
            gy = i * step
            if gy < self.height:
                segs.append(np.array([[0, gy, 0], [self.width, gy, 0]], np.float64))
        return segs

    def contains(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height


# --------------------------------------------------------------------------
# The transform contract.
# --------------------------------------------------------------------------


class CameraFieldTransform:
    """A rigid transform: field coordinates -> camera coordinates.

    Subclasses only supply R and t. Everything else is derived here, so a new
    way of *obtaining* the pose costs one small subclass and nothing else.

        p_camera = R @ p_field + t
    """

    #: rotation, field -> camera. Columns are the field axes in camera coords.
    R: np.ndarray
    #: translation: the field origin corner, expressed in camera coords.
    t: np.ndarray
    #: human-readable note about where this pose came from, shown on screen.
    source: str

    def __init__(self, R: np.ndarray, t: np.ndarray, source: str = "unspecified") -> None:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        # Cheap sanity check: R must actually be a rotation.
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("R is not orthonormal; not a rotation matrix")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError("R has determinant != 1; left-handed or scaled")
        self.R = R
        self.t = t
        self.source = source

    # -- points ------------------------------------------------------------

    def field_to_camera(self, pts_field: np.ndarray) -> np.ndarray:
        """(N,3) field points -> (N,3) camera points."""
        p = np.atleast_2d(np.asarray(pts_field, dtype=np.float64))
        return p @ self.R.T + self.t

    def camera_to_field(self, pts_cam: np.ndarray) -> np.ndarray:
        """(N,3) camera points -> (N,3) field points."""
        p = np.atleast_2d(np.asarray(pts_cam, dtype=np.float64))
        return (p - self.t) @ self.R

    # -- directions (rotation only, no translation) ------------------------

    def direction_to_field(self, vecs_cam: np.ndarray) -> np.ndarray:
        """(N,3) camera direction vectors -> (N,3) field direction vectors."""
        v = np.atleast_2d(np.asarray(vecs_cam, dtype=np.float64))
        return v @ self.R

    # -- for drawing -------------------------------------------------------

    def rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        """Pose as (rvec, tvec) for cv2.projectPoints of field-frame points."""
        import cv2

        rvec, _ = cv2.Rodrigues(self.R)
        return rvec, self.t.reshape(3, 1)

    def camera_origin_in_field(self) -> np.ndarray:
        """Where the camera sits, in field coords. For the top-down view."""
        return (-self.R.T @ self.t).reshape(3)


# --------------------------------------------------------------------------
# Today's transform: entirely made up.
# --------------------------------------------------------------------------


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


class SyntheticFieldTransform(CameraFieldTransform):
    """A field pose we invented, not measured. Placeholder for the real thing.

    Two flavours, both parameterised so the field lands in view:

    wall  -- the rectangle stands up facing the camera, `distance` metres away,
             like a poster on the far wall. Field +Y points up. This is the one
             to use with a hand-held phone: moving the phone left/right and
             up/down maps straight onto field x/y.

    floor -- the rectangle lies flat, camera `height` metres above the surface
             looking down at `pitch` degrees. Geometrically the RoboCup case,
             and awkward to demo by hand, but it proves nothing downstream
             assumed a fronto-parallel plane.
    """

    @classmethod
    def wall(
        cls,
        field: Field,
        distance: float = 1.5,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> SyntheticFieldTransform:
        # Field axes in camera coords when the field squarely faces the lens:
        #   field +X -> camera +X (right)
        #   field +Y -> camera -Y (up, because camera +Y points down)
        #   field +Z -> camera -Z (normal points back at the camera)
        base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
        R = _ry(np.radians(yaw_deg)) @ _rx(np.radians(pitch_deg)) @ base
        # Put the field *centre* straight ahead at `distance`, then back out the
        # translation of the origin corner.
        centre_cam = np.array([0.0, 0.0, float(distance)])
        centre_field = np.array([field.width / 2.0, field.height / 2.0, 0.0])
        t = centre_cam - R @ centre_field
        src = f"synthetic wall, {distance:.2f} m ahead"
        if yaw_deg or pitch_deg:
            src += f", yaw {yaw_deg:+.0f} pitch {pitch_deg:+.0f}"
        return cls(R, t, src)

    @classmethod
    def floor(
        cls,
        field: Field,
        height: float = 1.0,
        pitch_deg: float = 35.0,
    ) -> SyntheticFieldTransform:
        phi = np.radians(max(abs(pitch_deg), 1e-3))
        # Camera axes expressed in field coords, for a camera tilted down by phi:
        # +X right, +Z forward and downward, +Y down-ish.
        R_cam_to_field = np.array(
            [
                [1, 0, 0],
                [0, -np.sin(phi), np.cos(phi)],
                [0, -np.cos(phi), -np.sin(phi)],
            ],
            np.float64,
        )
        R = R_cam_to_field.T  # field -> camera
        # Sit the camera so its optical axis lands on the field centre.
        cam_in_field = np.array(
            [field.width / 2.0, field.height / 2.0 - height / np.tan(phi), float(height)]
        )
        t = -R @ cam_in_field
        return cls(
            R, t, f"synthetic floor, cam {height:.2f} m up, {pitch_deg:.0f} deg down"
        )


# --------------------------------------------------------------------------
# Tomorrow's transform. Same R and t, obtained by measurement.
# --------------------------------------------------------------------------


class ReferenceTagFieldTransform(CameraFieldTransform):
    """Field pose solved from reference tags at known field positions.

    This is the RoboCup path. Fix four tags to the real field, record where
    their centres are in field coordinates, and this recovers R and t from a
    single camera frame. Construction is the only difference from the synthetic
    case; every method inherited above, and the x/y/theta that comes out, is
    unchanged.

    `layout` maps tag_id -> (x, y) in metres, e.g. the four field corners:

        {0: (0.0, 0.0), 1: (1.2, 0.0), 2: (1.2, 0.8), 3: (0.0, 0.8)}

    Unlike the synthetic transform, this one needs real camera intrinsics to be
    worth anything -- the errors no longer cancel. See the README.
    """

    MIN_TAGS = 4

    @classmethod
    def from_detections(
        cls,
        detections,
        layout: dict[int, tuple[float, float]],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> ReferenceTagFieldTransform:
        import cv2

        obj, img = [], []
        for det in detections:
            if det.tag_id in layout:
                fx, fy = layout[det.tag_id]
                obj.append([fx, fy, 0.0])
                img.append(list(det.center))
        if len(obj) < cls.MIN_TAGS:
            raise LookupError(
                f"saw {len(obj)} reference tags, need {cls.MIN_TAGS} "
                f"(looking for ids {sorted(layout)})"
            )
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5, np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            np.array(obj, np.float64),
            np.array(img, np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed on the reference tags")
        R, _ = cv2.Rodrigues(rvec)
        return cls(R, tvec.reshape(3), f"measured from {len(obj)} reference tags")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"R": self.R.tolist(), "t": self.t.tolist(), "source": self.source},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> ReferenceTagFieldTransform:
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["R"]), np.array(d["t"]), d.get("source", str(path)))


# --------------------------------------------------------------------------
# Detection -> field pose. Downstream of the swap point: this must not care
# which CameraFieldTransform it was handed.
# --------------------------------------------------------------------------


#: Which axis of the detector's tag frame points to the tag's visual right --
#: the direction that runs left-to-right across an upright tag as you look at it.
#:
#: pupil_apriltags reports pose_R with the tag's +X axis pointing to the tag's
#: visual LEFT, +Y up and +Z into the tag (away from the viewer). That is a 180
#: degree rotation about the tag normal from the convention you might expect, so
#: visual-right is -X. Established empirically; selfcheck.py fails loudly if it
#: ever changes.
TAG_VISUAL_RIGHT = np.array([-1.0, 0.0, 0.0])


def tag_field_pose(
    detection,
    transform: CameraFieldTransform,
    field: Field,
    heading_offset_deg: float = 0.0,
) -> TagFieldPose:
    """Convert one pose-estimated AprilTag detection into field coordinates.

    `detection` must come from Detector.detect(..., estimate_tag_pose=True), so
    that it carries pose_R (tag->camera rotation) and pose_t (the tag centre in
    camera coords, metres).

    theta is 0 when the tag is upright, and increases counter-clockwise as seen
    from in front of the field. `heading_offset_deg` is added to it, for when the
    tag is mounted on a robot that does not face the same way as the tag's right
    edge -- a robot with the tag rotated 90 degrees on its shell wants -90 here.
    """
    centre_cam = np.asarray(detection.pose_t, dtype=np.float64).reshape(3)
    x, y, z = transform.camera_to_field(centre_cam)[0]

    # Heading: take the direction that points right across the upright tag,
    # express it in the field frame, and measure its angle about the field normal.
    pose_R = np.asarray(detection.pose_R, dtype=np.float64).reshape(3, 3)
    forward_cam = pose_R @ TAG_VISUAL_RIGHT
    vx, vy, _ = transform.direction_to_field(forward_cam)[0]
    theta = float(np.degrees(np.arctan2(vy, vx)) + heading_offset_deg)
    theta = (theta + 180.0) % 360.0 - 180.0  # wrap to (-180, 180]

    return TagFieldPose(
        tag_id=int(detection.tag_id),
        x=float(x),
        y=float(y),
        theta_deg=theta,
        off_plane_m=float(z),
        inside=field.contains(float(x), float(y)),
        decision_margin=float(detection.decision_margin),
    )
