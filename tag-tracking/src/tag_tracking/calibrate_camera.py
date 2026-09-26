"""One-time camera intrinsic calibration: focal length, principal point, and
lens distortion, via a ChArUco board.

    uv run calibrate-camera                  # live webcam, wave a board around
    uv run calibrate-camera --synthetic      # self-test against a digital camera, no hardware

Saves to calib/intrinsics.json at the repo root -- the actual calibration math
(vision_core.charuco) lives in vision-core, since it is camera-only and has
nothing to do with AprilTags; this is just the CLI wrapper: argparse, the live
capture window, and wiring the result into vision_core.intrinsics.save() so
track.py (and any future skill) picks it up automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from vision_core import intrinsics as intr
from vision_core.camera import open_camera, read_frame, read_key
from vision_core.charuco import (
    MIN_FRAMES,
    SyntheticChArucoCamera,
    calibrate_intrinsics,
    make_board,
)

MAX_LIVE_FRAMES = 60
#: A view only counts if its corners sit at least this far (median, px) from
#: where the same corners were in every view already captured. Without it a
#: 30 fps camera fills all MAX_LIVE_FRAMES from one pose in two seconds, and
#: the calibration never sees the tilts and frame corners it needs.
MIN_VIEW_CHANGE_PX = 25.0


def _is_new_view(view: dict[int, np.ndarray], captured: list[dict[int, np.ndarray]]) -> bool:
    for prev in captured:
        shared = view.keys() & prev.keys()
        if len(shared) < 4:
            continue
        moved = np.median([np.linalg.norm(view[i] - prev[i]) for i in shared])
        if moved < MIN_VIEW_CHANGE_PX:
            return False
    return True


def _run_synthetic(args):
    board, _dictionary = make_board(
        tuple(args.squares), args.square_mm / 1000.0, args.marker_mm / 1000.0
    )
    shape = (args.height, args.width)
    K_truth = intr.from_fov(args.width, args.height, 60.0)
    # k3 is 0 because calibrate_intrinsics pins it (fix_k3); a non-zero truth
    # there would just leak into the recovered k2 and muddy the comparison.
    dist_truth = np.array([0.05, -0.03, 0.001, -0.0005, 0.0])
    cam = SyntheticChArucoCamera(board, K_truth, dist_truth, shape=shape)

    print("Synthetic camera -- ground truth:")
    print(f"  K =\n{K_truth}")
    print(f"  distortion = {dist_truth}\n")

    result = calibrate_intrinsics(
        cam.frames(), board, (args.width, args.height), min_frames=args.min_frames
    )
    print(f"recovered K =\n{result.K}")
    print(f"recovered distortion = {result.dist}")
    print(f"rms reprojection error: {result.rms_reproj_px:.3f} px over {result.n_frames} views")
    fx_err_pct = 100 * abs(result.K[0, 0] - K_truth[0, 0]) / K_truth[0, 0]
    cx_err_px = abs(result.K[0, 2] - K_truth[0, 2])
    k1_err = abs(result.dist[0] - dist_truth[0])
    print(f"fx error vs ground truth: {fx_err_pct:.2f}%   cx error: {cx_err_px:.1f} px   "
          f"k1 error: {k1_err:.3f}")
    ok = fx_err_pct < 2.0 and k1_err < 0.02
    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    return result, args.out


def _run_live(args):
    board, _dictionary = make_board(
        tuple(args.squares), args.square_mm / 1000.0, args.marker_mm / 1000.0
    )
    n_markers = len(board.getIds())
    detector = cv2.aruco.CharucoDetector(board)

    print("Move the board around: different distances, tilts, and the corners")
    print("of the frame, not just the centre.")
    print(f"Capturing until {args.min_frames}+ good views are seen.")
    print("'q' to finish once enough are captured, Esc to abort.\n")

    cap = open_camera(args.camera, args.width, args.height)
    frames = []
    views: list[dict[int, np.ndarray]] = []
    coverage: list[np.ndarray] = []  # every captured corner, drawn so gaps show
    size = None
    win = "calibrate-camera"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            ok, bgr = read_frame(cap)
            if not ok:
                raise RuntimeError("camera stopped returning frames")
            if size is None:
                # Calibrate, and save, at the size the camera actually delivers --
                # it may not honour --width/--height, and a K keyed to the wrong
                # resolution is silently wrong everywhere it loads.
                size = (bgr.shape[1], bgr.shape[0])
                if size != (args.width, args.height):
                    print(f"note: asked for {args.width}x{args.height}, camera gives "
                          f"{size[0]}x{size[1]} -- calibrating at {size[0]}x{size[1]}; "
                          f"pass that --width/--height to every other command")
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids, _mk_corners, mk_ids = detector.detectBoard(grey)
            disp = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            n_seen = 0 if mk_ids is None else len(mk_ids)
            # OpenCV's detectBoard() can return charucoCorners/charucoIds that
            # disagree even when they look consistent from Python (matching
            # len(), no Nones) -- drawDetectedCornersCharuco has a known
            # shape-handling bug on some builds that still asserts on that.
            # So: never call it. Draw the corners ourselves with plain
            # cv2.circle, which only ever looks at a flattened point list and
            # cannot hit that bug.
            good = (
                ch_corners is not None and ch_ids is not None
                and len(ch_corners) >= 6 and len(ch_corners) == len(ch_ids)
            )
            for pt in coverage:
                cv2.circle(disp, (int(round(pt[0])), int(round(pt[1]))), 2,
                           (255, 160, 0), -1, cv2.LINE_AA)
            new_view = False
            if good:
                pts = np.asarray(ch_corners, dtype=np.float64).reshape(-1, 2)
                for pt in pts:
                    cv2.circle(disp, (int(round(pt[0])), int(round(pt[1]))), 5,
                               (0, 255, 0), -1, cv2.LINE_AA)
                view = dict(zip(np.asarray(ch_ids).ravel().tolist(), pts))
                new_view = _is_new_view(view, views)
                if new_view and len(frames) < MAX_LIVE_FRAMES:
                    frames.append(grey)
                    views.append(view)
                    coverage.extend(pts)
            cv2.putText(
                disp,
                f"captured {len(frames)}  (need {args.min_frames}+)   "
                f"markers seen {n_seen}/{n_markers}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if good else (0, 0, 255), 2, cv2.LINE_AA,
            )
            if good and not new_view and len(frames) < MAX_LIVE_FRAMES:
                cv2.putText(disp, "same pose as a captured view -- move, tilt, or go to "
                            "an uncovered part of the frame", (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)
            if not good:
                hint = (
                    "no markers visible -- move the board into frame" if n_seen == 0 else
                    "partial view -- hold it flatter-on, closer, or in better light"
                )
                cv2.putText(disp, hint, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 165, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)
            key = read_key()
            if key == 27:
                print("aborted")
                raise SystemExit(0)
            if key == ord("q") and len(frames) >= args.min_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    args.width, args.height = size  # main() saves under these
    result = calibrate_intrinsics(frames, board, size, min_frames=args.min_frames)
    print(f"\nK =\n{result.K}")
    print(f"distortion = {result.dist}")
    print(f"rms reprojection error: {result.rms_reproj_px:.3f} px over {result.n_frames} views")
    print(f"horizontal field of view: {intr.hfov_of(result.K, size[0]):.1f} deg")
    if result.rms_reproj_px > 1.0:
        print("warning: rms above 1 px -- blurry views, a board that is not flat, or the "
              "wrong board. Consider re-running before trusting this.")
    return result, (args.out or True)  # live mode always saves; True means "default path"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--synthetic", action="store_true",
        help="self-test against a digital ChArUco board, no camera needed",
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument(
        "--squares", type=int, nargs=2, metavar=("X", "Y"), default=[5, 7],
        help="ChArUco grid size in squares (default 5 7)",
    )
    ap.add_argument("--square-mm", type=float, default=35.0)
    ap.add_argument("--marker-mm", type=float, default=26.0)
    ap.add_argument(
        "--out", type=str, default=None,
        help="where to save (default calib/intrinsics.json at the repo root; "
        "--synthetic saves nowhere unless you pass this)",
    )
    ap.add_argument(
        "--save-board-png", metavar="PATH", help="write the board as a PNG, to print it"
    )
    args = ap.parse_args()

    if args.save_board_png:
        board, _dictionary = make_board(
            tuple(args.squares), args.square_mm / 1000.0, args.marker_mm / 1000.0
        )
        sx, sy = args.squares
        img = board.generateImage((1600, int(round(1600 * sy / sx))), marginSize=40)
        cv2.imwrite(args.save_board_png, img)
        print(f"wrote {args.save_board_png}")
        return

    result, out = _run_synthetic(args) if args.synthetic else _run_live(args)

    if not out:
        print("\n(--synthetic run, nothing saved -- pass --out to save anyway)")
        return

    path = intr.save(
        result.K, args.width, args.height, result.source, dist=result.dist,
        path=Path(out) if isinstance(out, str) else None,
    )
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
