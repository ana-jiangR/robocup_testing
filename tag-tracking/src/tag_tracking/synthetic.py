"""A digital field with reference AprilTags on it, for exercising field-pose
calibration and `track --synthetic-camera` with no hardware attached.

This is the AprilTag-shaped counterpart to vision_core.charuco's
SyntheticChArucoCamera: it needs pupil_apriltags to render and matters only to
this skill, so it lives here rather than in vision-core, same reasoning as
pose.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from vision_core.field import CameraFieldTransform, Field

MODULES = 8  # tag36h11: 6 data bits + a 1-module black border ring, edge to edge


@lru_cache(maxsize=64)
def _tag_bitmap(tag_id: int, px_per_module: int = 24) -> np.ndarray:
    """Cached: an animated camera re-renders the same handful of tag ids every
    frame, and regenerating each bitmap from the dictionary is not free.
    """
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    return cv2.aruco.generateImageMarker(d, tag_id, MODULES * px_per_module, 1)


def _warp_tag(
    frame: np.ndarray, tag_id: int, x: float, y: float, theta_deg: float, size: float,
    rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray,
) -> None:
    """Paint one AprilTag into `frame` at a given field pose, quiet zone
    included. Same trick selfcheck.py uses to fabricate detections.
    """
    th = np.radians(theta_deg)
    ex = np.array([np.cos(th), np.sin(th), 0.0])  # tag +X in field coords
    ey = np.array([np.sin(th), -np.cos(th), 0.0])  # tag +Y in field coords
    centre = np.array([x, y, 0.0])
    h = size / 2.0
    local = [(-h, -h), (h, -h), (h, h), (-h, h)]
    obj = np.array([centre + u * ex + v * ey for u, v in local])

    src_img = _tag_bitmap(tag_id)
    sh, sw = src_img.shape
    src_quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    dst_quad = img_pts.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(
        src_img, M, (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
    )
    mask = cv2.warpPerspective(
        np.full_like(src_img, 255), M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST
    )
    margin = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    frame[margin > 0] = 255
    frame[mask > 0] = warped[mask > 0]


@dataclass
class SyntheticFieldCamera:
    """A fixed camera looking at reference tags glued to a digital field.

    `layout` is tag_id -> (x, y) in field metres -- exactly what
    ReferenceTagFieldTransform.from_detections expects. `extra_tags` are
    additional, static (tag_id, x, y, theta_deg) tags to render, e.g. a
    'moving' tag to check the calibrated pose reads back the position it was
    placed at. `trajectory`, if set, adds one more tag whose pose is
    recomputed from the wall clock on every read() -- this is what makes
    `track --synthetic-camera` show something moving with no camera plugged
    in; see orbit_trajectory() below for a ready-made path.
    """

    layout: dict[int, tuple[float, float]]
    truth_transform: CameraFieldTransform
    tag_size: float = 0.08
    shape: tuple[int, int] = (720, 1280)
    K: np.ndarray | None = None
    extra_tags: tuple[tuple[int, float, float, float], ...] = ()
    noise_std: float = 1.0
    trajectory: Callable[[float], tuple[int, float, float, float]] | None = None

    def __post_init__(self) -> None:
        if self.K is None:
            from vision_core.intrinsics import from_fov

            h, w = self.shape
            self.K = from_fov(w, h)
        self._t0 = time.perf_counter()

    def read(self) -> np.ndarray:
        frame = np.full(self.shape, 210, np.uint8)
        rvec, tvec = self.truth_transform.rvec_tvec()
        for tid, (x, y) in self.layout.items():
            _warp_tag(frame, tid, x, y, 0.0, self.tag_size, rvec, tvec, self.K)
        for tid, x, y, th in self.extra_tags:
            _warp_tag(frame, tid, x, y, th, self.tag_size, rvec, tvec, self.K)
        if self.trajectory is not None:
            tid, x, y, th = self.trajectory(time.perf_counter() - self._t0)
            _warp_tag(frame, tid, x, y, th, self.tag_size, rvec, tvec, self.K)
        if self.noise_std:
            noise = np.random.normal(0, self.noise_std, frame.shape)
            frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return frame

    def release(self) -> None:
        pass


def orbit_trajectory(
    field: Field, tag_id: int = 9, period_s: float = 12.0, spin_period_s: float = 5.0,
) -> Callable[[float], tuple[int, float, float, float]]:
    """A smooth elliptical path for a 'moving' tag, for --synthetic-camera.

    The ellipse is deliberately a bit larger than the field, so the path dips
    outside near its flattest points -- the same IN/OUT boundary behaviour a
    real tag carried past the edge of the field would trigger, instead of a
    demo that only ever shows the happy path.
    """
    cx, cy = field.width / 2.0, field.height / 2.0
    rx, ry = field.width * 0.62, field.height * 0.62

    def trajectory(t: float) -> tuple[int, float, float, float]:
        w = 2 * np.pi / period_s
        x = cx + rx * np.cos(w * t)
        y = cy + ry * np.sin(w * t)
        theta = (360.0 * t / spin_period_s) % 360.0
        theta = (theta + 180.0) % 360.0 - 180.0
        return tag_id, float(x), float(y), float(theta)

    return trajectory
