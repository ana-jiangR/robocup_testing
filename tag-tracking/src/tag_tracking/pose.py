"""AprilTag detections -> field coordinates.

This is the tag-shaped half of what used to be field.py. It lives in the skill,
not in vision-core, because it is about reading AprilTags: the detector's frame
conventions, the tag's heading, the record a tag tracker emits. The geometry it
stands on -- where the field is at all -- is vision-core's problem.

Downstream of the swap point: nothing here may care which CameraFieldTransform
it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vision_core.field import CameraFieldTransform, Field

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
