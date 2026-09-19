"""Real camera intrinsic calibration, via a ChArUco board.

`intrinsics.py` only loads/saves/guesses a camera matrix; this is where one
actually gets *measured* -- focal length, principal point, and lens
distortion, from several views of a ChArUco board (a checkerboard with
AprilTag-style markers baked into it, so OpenCV's own detector finds its
corners precisely even at an angle).

Lives in vision-core, not a skill: this has nothing to do with AprilTags or
any one skill's target, only with the camera itself, same as intrinsics.py.

Also holds SyntheticChArucoCamera, a digital stand-in for a webcam waving a
board around -- so this module's own math can be exercised and trusted with
no hardware attached. See tag_tracking.synthetic for the AprilTag-specific
counterpart (a digital field with reference tags), which does need pupil_apriltags
and so cannot live here.

The board's physical size barely matters for calibration itself: plane-based
calibration (Zhang's method, what cv2.calibrateCamera runs) cannot observe
board scale independent of distance, so getting square-mm wrong mostly just
rescales the discarded per-view poses, not the K or distortion this measures.
What actually matters is variety: distances, tilts, and covering the corners
of the frame as well as the centre -- distortion is only constrained by
points away from the image centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Iterable

import cv2
import numpy as np

BOARD_SQUARES = (5, 7)
BOARD_SQUARE_M = 0.035
BOARD_MARKER_M = 0.026
BOARD_DICT = cv2.aruco.DICT_5X5_100

MIN_FRAMES = 12


def make_board(
    squares: tuple[int, int] = BOARD_SQUARES,
    square_m: float = BOARD_SQUARE_M,
    marker_m: float = BOARD_MARKER_M,
):
    """Build the ChArUco board object. Use the same call (same squares/sizes)
    to generate the printable board and to run calibration against it -- the
    two only agree with each other, not with any external standard.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(BOARD_DICT)
    return cv2.aruco.CharucoBoard(squares, square_m, marker_m, dictionary), dictionary


@dataclass
class IntrinsicsResult:
    K: np.ndarray
    dist: np.ndarray
    rms_reproj_px: float
    n_frames: int
    source: str


def calibrate_intrinsics(
    frames: Iterable[np.ndarray],
    board,
    image_size: tuple[int, int],
    min_frames: int = MIN_FRAMES,
    fix_k3: bool = True,
) -> IntrinsicsResult:
    """The actual math. `frames` is grayscale (or BGR) images from anywhere --
    a real webcam or a SyntheticChArucoCamera makes no difference here.

    fix_k3 pins the highest-order radial term to zero by default: it is the
    least-constrained coefficient for a board that does not fill the whole
    frame, and left free it will happily fit large, wrong values that still
    lower reprojection error on the calibration views without describing the
    lens.
    """
    detector = cv2.aruco.CharucoDetector(board)
    all_obj, all_img = [], []
    seen = 0
    for frame in frames:
        seen += 1
        grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _mk_corners, _mk_ids = detector.detectBoard(grey)
        # OpenCV's detectBoard() can return charucoCorners/charucoIds of
        # mismatched length on some views -- an OpenCV-side quirk, not a
        # board or camera problem -- so a length check has to gate this, not
        # just a None/count check, or matchImagePoints chokes on the pair.
        if (
            ch_corners is None or ch_ids is None
            or len(ch_corners) < 6 or len(ch_corners) != len(ch_ids)
        ):
            continue
        obj, img = board.matchImagePoints(ch_corners, ch_ids)
        pts = img.reshape(-1, 2).astype(np.float64)
        # Degenerate for homography fitting if the points are (nearly)
        # collinear in ANY direction, not just axis-aligned -- a grazing
        # partial view often lines its few visible corners up along a
        # diagonal, which an x/y-range check alone does not catch, and
        # cv2.calibrateCamera throws a hard C++ assertion on it rather than
        # failing gracefully. The eigenvalues of the point cloud's covariance
        # are the spread along its two principal axes regardless of
        # orientation; the smaller one collapsing towards zero is exactly
        # "collinear", whichever way the line points.
        centered = pts - pts.mean(axis=0)
        spread = np.sqrt(np.linalg.eigvalsh((centered.T @ centered) / len(pts))[0])
        if spread < 6.0:
            continue  # corners nearly collinear in some direction
        all_obj.append(obj)
        all_img.append(img)
    if len(all_obj) < min_frames:
        raise RuntimeError(
            f"only {len(all_obj)}/{seen} frames had a usable board view, "
            f"need >= {min_frames}. Cover more distances/tilts, and the corners "
            f"of the frame, not just the centre."
        )
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None, flags=flags
    )
    return IntrinsicsResult(
        K=K,
        dist=dist.ravel(),
        rms_reproj_px=float(rms),
        n_frames=len(all_obj),
        source=f"charuco, {len(all_obj)} views, rms {rms:.2f}px",
    )


# --------------------------------------------------------------------------
# A digital board, for exercising the calibration math with no hardware.
# --------------------------------------------------------------------------


def default_board_poses(board_w: float, board_h: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """A spread of distances, tilts, and off-centre shifts standing in for
    'wave the board around, covering the corners of the frame too' -- the
    same advice a real calibration session needs, scripted instead of waved.
    """

    def pose(
        dist_m: float, tilt_deg: float = 0.0, yaw_deg: float = 0.0,
        shift_x: float = 0.0, shift_y: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        phi, psi = np.radians(tilt_deg), np.radians(yaw_deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        Ry = np.array([[np.cos(psi), 0, np.sin(psi)], [0, 1, 0], [-np.sin(psi), 0, np.cos(psi)]])
        R = Rx @ Ry
        centre = np.array([board_w / 2.0, board_h / 2.0, 0.0])
        t = np.array([shift_x, shift_y, float(dist_m)]) - R @ centre
        rvec, _ = cv2.Rodrigues(R)
        return rvec, t

    return [
        pose(0.6, 0), pose(0.6, 25), pose(0.6, -25), pose(0.6, 0, 25), pose(0.6, 0, -25),
        pose(0.8, 20), pose(0.8, -20), pose(0.8, 0, 20), pose(0.9, 25, 15),
        pose(0.5, 15, -15), pose(0.7, -30, 20), pose(0.7, 30, -20), pose(1.1, 0, 0),
        pose(0.6, 35, 0), pose(0.6, -35, 0),
        pose(0.6, 10, 10, shift_x=0.25, shift_y=0.15),
        pose(0.6, -10, -10, shift_x=-0.25, shift_y=0.15),
        pose(0.6, 10, -10, shift_x=0.25, shift_y=-0.15),
        pose(0.6, -10, 10, shift_x=-0.25, shift_y=-0.15),
        pose(0.6, 0, 0, shift_x=0.15, shift_y=0.0),
    ]


@dataclass
class SyntheticChArucoCamera:
    """Renders a real cv2.aruco.CharucoBoard at a scripted sequence of poses.

    The board is rasterised once at high resolution, perspective-warped into
    each frame as a pinhole camera would see it, then pushed through the lens
    distortion as a per-pixel remap. The two steps must stay separate: a
    homography can only place the board's four corners, it cannot bend the
    lines between them, so warping straight to the *distorted* corners renders
    a board that is still straight-lined and calibrates to ~zero distortion no
    matter what dist_truth says. (Filling each square as its own polygon was
    tried before that and fabricated phantom distortion from rasterisation
    bias instead.)
    """

    board: "cv2.aruco.CharucoBoard"
    K_truth: np.ndarray
    dist_truth: np.ndarray | None = None
    shape: tuple[int, int] = (720, 1280)
    noise_std: float = 1.5  # sensor-noise stand-in, so repeated reads are not bit-identical
    poses: list[tuple[np.ndarray, np.ndarray]] = _field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dist_truth is None:
            self.dist_truth = np.zeros(5)
        nx, ny = self.board.getChessboardSize()
        sq = self.board.getSquareLength()
        self.board_size_m = (nx * sq, ny * sq)
        if not self.poses:
            self.poses = default_board_poses(*self.board_size_m)
        px_per_m = 4000  # high-res source raster; the warp below is what actually lands in-frame
        w_px = max(2, int(round(self.board_size_m[0] * px_per_m)))
        h_px = max(2, int(round(self.board_size_m[1] * px_per_m)))
        self._pattern = self.board.generateImage((w_px, h_px), marginSize=0)
        self._i = 0
        # Distorted pixel -> where it samples the pinhole image. Built once.
        self._distort_map = None
        if np.any(self.dist_truth):
            h, w = self.shape
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            grid = np.stack([xs.ravel(), ys.ravel()], axis=1)[:, None, :]
            und = cv2.undistortPoints(grid, self.K_truth, self.dist_truth, P=self.K_truth)
            self._distort_map = und.reshape(h, w, 2).astype(np.float32)

    def _render(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        h, w = self.shape
        ph, pw = self._pattern.shape[:2]
        src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
        bw, bh = self.board_size_m
        obj = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float64)
        pix, _ = cv2.projectPoints(obj, rvec, tvec, self.K_truth, None)  # pinhole
        dst = pix.reshape(4, 2).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        frame = cv2.warpPerspective(
            self._pattern, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=235
        )
        if self._distort_map is not None:
            frame = cv2.remap(frame, self._distort_map, None, cv2.INTER_LINEAR, borderValue=235)
        if self.noise_std:
            noise = np.random.normal(0, self.noise_std, frame.shape)
            frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return frame

    def frames(self):
        """One frame per scripted pose -- the finite sequence a self-test consumes."""
        for rvec, tvec in self.poses:
            yield self._render(rvec, tvec)

    def read(self) -> np.ndarray:
        """cycles through the scripted poses on repeated calls."""
        rvec, tvec = self.poses[self._i % len(self.poses)]
        self._i += 1
        return self._render(rvec, tvec)
