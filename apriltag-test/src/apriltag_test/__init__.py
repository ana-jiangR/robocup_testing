"""Virtual-field AprilTag tracking.

The public surface is the field geometry and the camera->field transform; see
field.py, which is the one place that decides where the field is.
"""

from .field import (
    CameraFieldTransform,
    Field,
    ReferenceTagFieldTransform,
    SyntheticFieldTransform,
    TagFieldPose,
    tag_field_pose,
)

__all__ = [
    "CameraFieldTransform",
    "Field",
    "ReferenceTagFieldTransform",
    "SyntheticFieldTransform",
    "TagFieldPose",
    "tag_field_pose",
]
