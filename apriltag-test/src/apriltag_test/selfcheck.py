"""Prove the field maths without a camera.

Renders synthetic images of a tag at known field positions, runs them through
the real detector and the real field.py code, and checks the numbers come back.
If this passes, any error you see later is in the camera, the intrinsics, or the
tag on the phone -- not in the geometry.

    uv run selfcheck
"""

from __future__ import annotations

import cv2
import numpy as np
import pupil_apriltags as pa

from .field import (
    Field,
    ReferenceTagFieldTransform,
    SyntheticFieldTransform,
    tag_field_pose,
)

TAG_FAMILY = "tag36h11"
MODULES = 8  # tag36h11 is 8x8 modules edge-to-edge of the black square

# Tolerances. These are not arbitrary: an 80 mm tag seen at 1.5 m by a 1280-wide
# 60-degree camera is only ~60 px across, and corner localisation good to ~0.3 px
# still leaves roughly 1% range error. In-plane x/y is the accurate part; the
# off-plane (depth) axis is always the worst for a small planar target, which is
# worth remembering when you read off_plane_m on screen later.
TOL_XY_M = 0.015
TOL_OFF_PLANE_M = 0.030
TOL_THETA_DEG = 2.0


def intrinsics(width: int, height: int, hfov_deg: float = 60.0) -> np.ndarray:
    """A plausible pinhole camera. Same helper the tracker uses."""
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array(
        [[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1]], np.float64
    )


def tag_image(tag_id: int, px_per_module: int = 24) -> np.ndarray:
    """The 8x8-module black square, no quiet zone."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    return cv2.aruco.generateImageMarker(d, tag_id, MODULES * px_per_module, 1)


def tag_corners_in_field(
    x: float, y: float, theta_deg: float, size: float
) -> np.ndarray:
    """The tag's 4 corners in field coords, in image order (TL, TR, BR, BL).

    The tag lies in the field plane, centred at (x, y), rotated by theta about
    the field normal. Tag +X is its right edge, tag +Y points down the image, so
    tag +Y maps to -theta-rotated field -Y.
    """
    th = np.radians(theta_deg)
    ex = np.array([np.cos(th), np.sin(th), 0.0])  # tag +X in field coords
    ey = np.array([np.sin(th), -np.cos(th), 0.0])  # tag +Y in field coords
    centre = np.array([x, y, 0.0])
    h = size / 2.0
    local = [(-h, -h), (h, -h), (h, h), (-h, h)]
    return np.array([centre + u * ex + v * ey for u, v in local])


def render(
    tag_id: int,
    poses: list[tuple[float, float, float]],
    transform,
    tag_size: float,
    K: np.ndarray,
    shape: tuple[int, int],
    into: np.ndarray | None = None,
) -> np.ndarray:
    """Paint tags at the given (x, y, theta) field poses into a grey frame.

    Pass `into` to add more tags to an existing frame; compositing separate
    frames afterwards would swallow the white quiet zones the detector needs.
    """
    h, w = shape
    frame = np.full((h, w), 210, np.uint8) if into is None else into
    src = tag_image(tag_id)
    sh, sw = src.shape
    src_quad = np.array(
        [[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32
    )
    rvec, tvec = transform.rvec_tvec()
    for x, y, theta in poses:
        pts_field = tag_corners_in_field(x, y, theta, tag_size)
        img_pts, _ = cv2.projectPoints(pts_field, rvec, tvec, K, np.zeros(5))
        dst_quad = img_pts.reshape(4, 2).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_quad, dst_quad)
        # Warp the tag and its white quiet zone in one go.
        padded = cv2.copyMakeBorder(src, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
        warped = cv2.warpPerspective(
            padded, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )
        mask = cv2.warpPerspective(
            np.full_like(src, 255), M, (w, h), flags=cv2.INTER_NEAREST
        )
        # A white margin around the quad, so the tag is not touching grey.
        margin = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        frame[margin > 0] = 255
        frame[mask > 0] = warped[mask > 0]
    return frame


def check_round_trip() -> list[str]:
    """Put a tag at known field poses, see if we get those poses back."""
    failures = []
    field = Field(1.2, 0.8)
    shape = (720, 1280)
    K = intrinsics(shape[1], shape[0], 60.0)
    tag_size = 0.08  # 80 mm black square, a realistic phone-screen tag
    det = pa.Detector(families=TAG_FAMILY, nthreads=2, quad_decimate=1.0)
    cam = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    cases = [
        ("wall", SyntheticFieldTransform.wall(field, distance=1.5)),
        ("wall yawed 20deg", SyntheticFieldTransform.wall(field, 1.5, yaw_deg=20.0)),
        ("floor", SyntheticFieldTransform.floor(field, height=1.0, pitch_deg=35.0)),
    ]
    truth = [
        (0.20, 0.20, 0.0),
        (0.60, 0.40, 45.0),
        (1.00, 0.60, -90.0),
    ]

    for name, transform in cases:
        for want in truth:
            frame = render(7, [want], transform, tag_size, K, shape)
            found = det.detect(
                frame, estimate_tag_pose=True, camera_params=cam, tag_size=tag_size
            )
            if not found:
                failures.append(f"{name}: no detection for pose {want}")
                continue
            got = tag_field_pose(found[0], transform, field)
            dx, dy = abs(got.x - want[0]), abs(got.y - want[1])
            dth = abs((got.theta_deg - want[2] + 180) % 360 - 180)
            ok = (
                dx < TOL_XY_M
                and dy < TOL_XY_M
                and dth < TOL_THETA_DEG
                and abs(got.off_plane_m) < TOL_OFF_PLANE_M
            )
            flag = "ok " if ok else "FAIL"
            print(
                f"  [{flag}] {name:18s} want x={want[0]:.2f} y={want[1]:.2f} "
                f"th={want[2]:+6.1f}  got x={got.x:.3f} y={got.y:.3f} "
                f"th={got.theta_deg:+6.1f} off={got.off_plane_m:+.3f}  "
                f"err {max(dx, dy) * 1000:4.1f} mm / {dth:.1f} deg"
            )
            if not ok:
                failures.append(
                    f"{name}: pose {want} -> "
                    f"({got.x:.3f}, {got.y:.3f}, {got.theta_deg:.1f})"
                )
    return failures


def check_bounds() -> list[str]:
    """A tag outside the rectangle must report inside=False."""
    failures = []
    field = Field(1.2, 0.8)
    transform = SyntheticFieldTransform.wall(field, distance=1.5)
    for x, y, expect in [(0.6, 0.4, True), (1.35, 0.4, False), (0.6, -0.12, False)]:
        if field.contains(x, y) is not expect:
            failures.append(f"contains({x}, {y}) should be {expect}")
    # And the geometry agrees with Field.contains via a real detection.
    shape = (720, 1280)
    K = intrinsics(shape[1], shape[0], 60.0)
    det = pa.Detector(families=TAG_FAMILY, nthreads=2, quad_decimate=1.0)
    cam = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    frame = render(7, [(1.35, 0.40, 0.0)], transform, 0.08, K, shape)
    found = det.detect(frame, estimate_tag_pose=True, camera_params=cam, tag_size=0.08)
    if not found:
        failures.append("out-of-bounds tag was not detected at all")
    else:
        got = tag_field_pose(found[0], transform, field)
        print(
            f"  [{'ok ' if not got.inside else 'FAIL'}] out-of-bounds     "
            f"x={got.x:.3f} y={got.y:.3f} inside={got.inside} (want inside=False)"
        )
        if got.inside:
            failures.append(f"tag at x=1.35 reported inside (x={got.x:.3f})")
    return failures


def check_transform_swap() -> list[str]:
    """The point of the exercise: swap the invented pose for a measured one.

    Builds a ground-truth floor pose, renders four reference tags at known field
    corners, recovers the pose with ReferenceTagFieldTransform, and checks that a
    fifth tag lands in the same place under both transforms.
    """
    failures = []
    field = Field(1.2, 0.8)
    shape = (720, 1280)
    K = intrinsics(shape[1], shape[0], 60.0)
    det = pa.Detector(families=TAG_FAMILY, nthreads=2, quad_decimate=1.0)
    cam = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    truth_transform = SyntheticFieldTransform.floor(field, height=1.1, pitch_deg=40.0)
    layout = {0: (0.0, 0.0), 1: (1.2, 0.0), 2: (1.2, 0.8), 3: (0.0, 0.8)}

    # Render the four reference tags plus a moving tag, all into one frame. Every
    # tag here is the same physical size, because detect() is told a single
    # tag_size for the whole frame -- mixing sizes scales poses along the view ray.
    tag_size = 0.08
    frame = np.full(shape, 210, np.uint8)
    for tid, (x, y) in layout.items():
        render(tid, [(x, y, 0.0)], truth_transform, tag_size, K, shape, into=frame)
    moving_truth = (0.75, 0.55, 30.0)
    render(7, [moving_truth], truth_transform, tag_size, K, shape, into=frame)

    found = det.detect(
        frame, estimate_tag_pose=True, camera_params=cam, tag_size=tag_size
    )
    ids = sorted(d.tag_id for d in found)
    print(f"  detected ids {ids} (want [0, 1, 2, 3, 7])")
    if not set(layout).issubset(ids):
        failures.append(f"reference tags missing, saw {ids}")
        return failures

    measured = ReferenceTagFieldTransform.from_detections(found, layout, K)
    dR = np.degrees(
        np.arccos(np.clip((np.trace(measured.R.T @ truth_transform.R) - 1) / 2, -1, 1))
    )
    dt = float(np.linalg.norm(measured.t - truth_transform.t))
    print(f"  recovered pose vs truth: rotation off {dR:.2f} deg, origin off {dt:.4f} m")
    if dR > 1.0 or dt > 0.02:
        failures.append(f"recovered pose off by {dR:.2f} deg / {dt:.4f} m")

    moving = next(d for d in found if d.tag_id == 7)
    a = tag_field_pose(moving, truth_transform, field)
    b = tag_field_pose(moving, measured, field)
    print(
        f"  same tag, invented transform: x={a.x:.3f} y={a.y:.3f} th={a.theta_deg:+.1f}"
    )
    print(
        f"  same tag, measured transform: x={b.x:.3f} y={b.y:.3f} th={b.theta_deg:+.1f}"
    )
    if abs(a.x - b.x) > 0.02 or abs(a.y - b.y) > 0.02:
        failures.append("measured and invented transforms disagree beyond 2 cm")
    if abs(a.x - moving_truth[0]) > 0.02 or abs(a.y - moving_truth[1]) > 0.02:
        failures.append(f"moving tag off truth {moving_truth}")
    return failures


def main() -> None:
    print("Field maths self-check -- synthetic images, no camera involved.\n")
    failures: list[str] = []

    print("1. tag at known field poses, three different camera/field geometries:")
    failures += check_round_trip()

    print("\n2. out-of-bounds detection:")
    failures += check_bounds()

    print("\n3. swapping the invented transform for one measured from 4 ref tags:")
    failures += check_transform_swap()

    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed. The geometry is sound; on to the camera.")


if __name__ == "__main__":
    main()
