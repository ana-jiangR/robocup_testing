"""Named ball color profiles, and the tool that measures them.

    uv run calibrate-ball --profile match --radius-mm 40

The ball is not decided -- a stand-in now, a dedicated one later -- so nothing is
hardcoded. Colors are named profiles in calib/ball_color.json, picked with
--ball-profile. Adding a ball is running this tool again with a new name.

A profile is a 2D hue/saturation histogram measured off the real ball, used with
cv2.calcBackProject rather than a hard cv2.inRange. inRange is a cliff: one
degree of hue drift and the ball vanishes. Back-projection returns a likelihood,
so the ball fades instead and the blob scorer can still find it.

Brightness is deliberately left out of the histogram and applied as a wide-open
gate. V is what shadow does to a ball; H and S are what identify it.

Buy a **matte** ball, and lean magenta (OpenCV H 150-172) -- skin, wood, carpet
and cardboard all cluster at H 0-30. Matte matters more than the hue: a specular
highlight punches a moving hole in the mask and destabilises the apparent radius,
which is what the height estimate depends on. Press `h` in the tool to see which
hues your actual room already uses, rather than taking any of this on faith.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from vision_core.camera import list_cameras, lock_camera, unlock_camera, open_camera, read_key
from vision_core.paths import calib_path

#: Shared across skills -- it describes the camera's view of a ball, and lives
#: beside intrinsics.json for the same reason.
COLOR_FILE = "ball_color.json"

#: Histogram resolution. 30 hue bins is 6 degrees each: fine enough to separate
#: magenta from red, coarse enough that a handful of sampled pixels still fills
#: it densely rather than turning into a comb of spikes.
H_BINS, S_BINS = 30, 32
H_RANGE, S_RANGE = 180, 256

#: How far the stored histogram reaches beyond the pixels it was measured from,
#: in bins, before back-projection. A profile measured under one lamp meets the
#: same ball under another: shade and a bluish ambient both pull saturation
#: down, a warmer bulb pushes it up, and a re-balanced camera nudges the hue.
#: The measured histogram is a tight island in H/S, so a few bins of drift lands
#: the ball on zero likelihood and it vanishes -- the very cliff back-projection
#: was chosen to avoid. Spreading it along S (widely: 4 bins is 32 levels) and
#: along H (a little: 1 bin is 6 degrees) lets the ball fade instead. Both are
#: per profile and tunable from calibrate-ball.
H_SPREAD_BINS = 1.0
S_SPREAD_BINS = 4.0

WIN = "ball color calibration"


# --------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------


@dataclass
class BallColor:
    """One named ball: how to find it, and how big it is.

    The radius lives here, not on the command line, because it is a property of
    this ball and every geometric step needs it. Selecting a profile should
    select the whole ball, not just its hue.
    """

    name: str
    radius_m: float = 0.020
    note: str = ""
    #: (H_BINS, S_BINS) float32, normalised to 0..255. The measured color.
    hist: np.ndarray = dc_field(default_factory=lambda: np.zeros((H_BINS, S_BINS), np.float32))
    #: Floor on saturation: washed-out pixels carry no reliable hue at all.
    s_min: int = 80
    #: Value gate, deliberately wide. Excludes only true black and blown-out white.
    v_min: int = 40
    v_max: int = 255
    #: Back-projection likelihood above which a pixel counts as ball, 0..255.
    threshold: int = 40
    #: Reach beyond the measured histogram, in bins -- see H_SPREAD_BINS.
    h_spread: float = H_SPREAD_BINS
    s_spread: float = S_SPREAD_BINS
    created: str = ""

    # -- masking -----------------------------------------------------------

    def working_hist(self) -> np.ndarray:
        """The histogram back-projection actually runs against: the measured
        one, spread by (h_spread, s_spread) bins and re-normalised to 255 so
        `threshold` keeps meaning the same thing.

        The measured histogram is what gets saved; the spread is applied on use,
        so a profile can be re-tuned without re-sampling the ball, and an old
        profile picks up the default spread the moment it is loaded.
        """
        hist = np.asarray(self.hist, np.float32)
        if self.h_spread <= 0.0 and self.s_spread <= 0.0:
            return np.ascontiguousarray(hist)
        # Hue wraps -- red sits at both ends of the axis -- so pad the H axis
        # with itself rather than letting the blur clamp at the border.
        pad = int(np.ceil(3.0 * max(self.h_spread, 0.0))) + 1
        padded = np.concatenate([hist[-pad:], hist, hist[:pad]], axis=0)
        # Rows are H, columns are S: sigmaY spreads hue, sigmaX saturation.
        blurred = cv2.GaussianBlur(
            padded, (0, 0),
            sigmaX=max(self.s_spread, 1e-3), sigmaY=max(self.h_spread, 1e-3),
            borderType=cv2.BORDER_REPLICATE,
        )
        out = np.ascontiguousarray(blurred[pad:pad + H_BINS])
        if out.max() > 0.0:
            cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
        return out

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """A uint8 0/255 mask of where this ball probably is."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject(
            [hsv], [0, 1], self.working_hist(), [0, H_RANGE, 0, S_RANGE], scale=1.0
        )
        gate = cv2.inRange(
            hsv,
            np.array([0, self.s_min, self.v_min], np.uint8),
            np.array([179, 255, self.v_max], np.uint8),
        )
        prob = cv2.bitwise_and(prob, gate)
        # A small blur before thresholding fills single-pixel pinholes that
        # thresholding would otherwise turn into ragged blob edges.
        prob = cv2.GaussianBlur(prob, (5, 5), 0)
        _, m = cv2.threshold(prob, self.threshold, 255, cv2.THRESH_BINARY)
        return m

    def likelihood(self, frame_bgr: np.ndarray) -> np.ndarray:
        """The raw 0..255 back-projection, for the preview."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.calcBackProject(
            [hsv], [0, 1], self.working_hist(), [0, H_RANGE, 0, S_RANGE], scale=1.0
        )

    def hue_peak(self) -> int:
        """The dominant hue, in OpenCV units (0..179). For printing."""
        if not self.hist.any():
            return -1
        per_hue = self.hist.sum(axis=1)
        return int(round(int(np.argmax(per_hue)) * (H_RANGE / H_BINS)))

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "radius_m": float(self.radius_m),
            "note": self.note,
            "h_bins": H_BINS,
            "s_bins": S_BINS,
            "hist": np.asarray(self.hist, np.float32).round(2).tolist(),
            "s_min": int(self.s_min),
            "v_min": int(self.v_min),
            "v_max": int(self.v_max),
            "threshold": int(self.threshold),
            "h_spread": float(self.h_spread),
            "s_spread": float(self.s_spread),
            "created": self.created or date.today().isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BallColor:
        hist = np.array(d["hist"], np.float32)
        if hist.shape != (H_BINS, S_BINS):
            raise ValueError(
                f"profile {d.get('name')!r} has a {hist.shape} histogram but this "
                f"build uses {(H_BINS, S_BINS)} -- re-run calibrate-ball"
            )
        return cls(
            name=d["name"],
            radius_m=float(d.get("radius_m", 0.020)),
            note=d.get("note", ""),
            hist=hist,
            s_min=int(d.get("s_min", 80)),
            v_min=int(d.get("v_min", 40)),
            v_max=int(d.get("v_max", 255)),
            threshold=int(d.get("threshold", 40)),
            h_spread=float(d.get("h_spread", H_SPREAD_BINS)),
            s_spread=float(d.get("s_spread", S_SPREAD_BINS)),
            created=d.get("created", ""),
        )


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------


def color_file() -> Path:
    return calib_path(COLOR_FILE)


def load_all(path: Path | None = None) -> dict[str, BallColor]:
    path = path or color_file()
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {k: BallColor.from_dict(v) for k, v in d.get("profiles", {}).items()}


def load_profile(name: str, path: Path | None = None) -> BallColor:
    """One named profile, with an error that says what you could have asked for."""
    path = path or color_file()
    profiles = load_all(path)
    if not profiles:
        raise SystemExit(
            f"no ball color profiles yet ({path} is missing or empty).\n"
            f"Run:  uv run calibrate-ball --profile {name}"
        )
    if name not in profiles:
        raise SystemExit(
            f"no ball profile {name!r} in {path}. Have: {', '.join(sorted(profiles))}.\n"
            f"Run:  uv run calibrate-ball --profile {name}"
        )
    return profiles[name]


def save_profile(profile: BallColor, path: Path | None = None) -> Path:
    """Add or replace one profile, leaving the others alone."""
    path = path or color_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {"profiles": {}}
    doc.setdefault("profiles", {})[profile.name] = profile.to_dict()
    doc["default"] = doc.get("default") or profile.name
    path.write_text(json.dumps(doc, indent=2))
    return path


# --------------------------------------------------------------------------
# Measuring a profile
# --------------------------------------------------------------------------


def histogram_from_samples(samples_hsv: np.ndarray, s_min: int) -> np.ndarray:
    """Build the normalised H/S histogram from sampled (N, 3) HSV pixels.

    Low-saturation pixels go first: a grey pixel's hue is whatever rounding says,
    and leaving them in smears the histogram across the whole spectrum.
    """
    s = samples_hsv.reshape(-1, 3)
    s = s[s[:, 1] >= s_min]
    if len(s) < 50:
        raise ValueError(
            f"only {len(s)} usable pixels after the S>={s_min} cut. Sample a bigger "
            f"patch of the ball, add light, or lower --s-min."
        )
    hist = cv2.calcHist(
        [s.reshape(-1, 1, 3)], [0, 1], None, [H_BINS, S_BINS], [0, H_RANGE, 0, S_RANGE]
    )
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def scene_hue_histogram(frame_bgr: np.ndarray, s_min: int = 60) -> np.ndarray:
    """Hue counts across the whole frame.

    For empirical color choice: a ball hue is only good if the room is not
    already full of it, and you cannot reason about the room -- you have to look.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    px = hsv.reshape(-1, 3)
    px = px[(px[:, 1] >= s_min) & (px[:, 2] >= 40)]
    if len(px) == 0:
        return np.zeros(H_BINS, np.float64)
    counts, _ = np.histogram(px[:, 0], bins=H_BINS, range=(0, H_RANGE))
    return counts.astype(np.float64) / max(counts.sum(), 1)


#: Rough names for the OpenCV hue wheel, to make the histogram readable.
_HUE_NAMES = [
    (0, "red"), (10, "orange"), (20, "yellow"), (35, "green"), (75, "cyan"),
    (100, "blue"), (130, "violet"), (150, "magenta"), (170, "red"),
]


def _hue_name(h: float) -> str:
    name = _HUE_NAMES[0][1]
    for lo, n in _HUE_NAMES:
        if h >= lo:
            name = n
    return name


def print_scene_histogram(counts: np.ndarray, mark: int | None = None) -> None:
    """Print the scene's hue distribution as a text bar chart.

    A map of which hues are already taken. You want the ball in a near-empty bin;
    every occupied one is a false positive the scorer has to work to reject.
    """
    print("\nScene hue histogram (saturated pixels only, this frame):")
    print("  OpenCV H   approx       share")
    top = max(counts.max(), 1e-9)
    width = H_RANGE / H_BINS
    for i, c in enumerate(counts):
        lo = int(i * width)
        bar = "#" * int(round(40 * c / top))
        flag = ""
        if mark is not None and lo <= mark < lo + width:
            flag = "  <-- your ball"
        print(f"  {lo:3d}-{int(lo + width - 1):3d}  {_hue_name(lo):>8s}  "
              f"{c * 100:5.1f}%  {bar}{flag}")
    quiet = sorted(range(len(counts)), key=lambda i: counts[i])[:4]
    names = ", ".join(
        f"H~{int(i * width)}-{int(i * width + width - 1)} ({_hue_name(i * width)})"
        for i in sorted(quiet)
    )
    print(f"\n  Emptiest bins in this scene: {names}")
    print("  A ball sitting in an empty bin is a ball with no competition.\n")


# --------------------------------------------------------------------------
# The interactive tool
# --------------------------------------------------------------------------


class _Sampler:
    """Click-drag a rectangle over the ball to sample it."""

    def __init__(self) -> None:
        self.start: tuple[int, int] | None = None
        self.box: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1
        self.dragging = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.box = None
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.start:
            self.box = (*self.start, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            self.box = (*self.start, x, y)
            self.dragging = False

    def region(self) -> tuple[int, int, int, int] | None:
        """The box as (x0, y0, x1, y1), ordered, or None if it is too small."""
        if not self.box:
            return None
        x0, y0, x1, y1 = self.box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if (x1 - x0) < 6 or (y1 - y0) < 6:
            return None
        return x0, y0, x1, y1


def _clip_box(
    box: tuple[int, int, int, int], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Clamp a drag box to the video frame's bounds."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return (
        max(0, min(x0, w)), max(0, min(y0, h)),
        max(0, min(x1, w)), max(0, min(y1, h)),
    )


def _overlay(frame: np.ndarray, profile: BallColor, sampler: _Sampler,
             have_sample: bool) -> np.ndarray:
    """Video on the left, live mask on the right."""
    shown = frame.copy()
    box = sampler.region()
    if box:
        cv2.rectangle(shown, box[:2], box[2:], (60, 240, 60), 2)

    if have_sample:
        prob = profile.likelihood(frame)
        mask = profile.mask(frame)
        # Likelihood as a heatmap, with the thresholded mask outlined on top, so
        # you can see both what the histogram thinks and where the cut lands.
        right = cv2.applyColorMap(prob, cv2.COLORMAP_INFERNO)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, contours, -1, (90, 255, 90), 2)
        n_on = int(cv2.countNonZero(mask))
        pct = 100.0 * n_on / mask.size
        label = f"mask {pct:.2f}% of frame, {len(contours)} blobs"
        color = (90, 255, 90) if 0.02 < pct < 5.0 else (90, 140, 255)
    else:
        right = np.zeros_like(frame)
        label = "drag a box over the ball to sample it"
        color = (200, 200, 200)

    cv2.putText(right, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv2.LINE_AA)
    head = (
        f"profile '{profile.name}'  r={profile.radius_m * 1000:.0f}mm  "
        f"thresh {profile.threshold}  s_min {profile.s_min}  "
        f"spread H{profile.h_spread:g} S{profile.s_spread:g}"
        + (f"  H~{profile.hue_peak()}" if have_sample else "")
    )
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(shown, head, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)
    return np.hstack([shown, right])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Keys: drag = sample the ball, h = scene hue histogram, "
            "- / + = threshold, [ / ] = s_min, < / > = S spread, ; / ' = H spread, "
            "a = add to sample, s = save, q = quit"
        ),
    )
    ap.add_argument("--profile", default="test",
                    help="profile name to create or overwrite (default 'test'). "
                         "Use 'test' for the stand-in ball and 'match' for the real one")
    ap.add_argument("--radius-mm", type=float, default=20.0,
                    help="ball RADIUS in millimetres (default 20, i.e. a 40 mm ball)")
    ap.add_argument("--note", default="", help="free text stored with the profile")
    ap.add_argument("--s-min", type=int, default=80,
                    help="saturation floor, 0-255 (default 80)")
    ap.add_argument("--threshold", type=int, default=40,
                    help="back-projection cut, 0-255 (default 40)")
    ap.add_argument("--h-spread", type=float, default=H_SPREAD_BINS,
                    help="how far the profile reaches beyond the sampled hues, in "
                         f"bins of {H_RANGE / H_BINS:g} degrees (default {H_SPREAD_BINS:g})")
    ap.add_argument("--s-spread", type=float, default=S_SPREAD_BINS,
                    help="how far it reaches along saturation, in bins of "
                         f"{S_RANGE / S_BINS:g} levels (default {S_SPREAD_BINS:g}). Raise "
                         "it if the ball drops out of the mask when it rolls into "
                         "shade or under a different lamp")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the auto-exposure / auto-WB lock (not recommended: "
                         "you would be measuring a color the camera is still changing)")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--list-profiles", action="store_true",
                    help="print the profiles already stored, and exit")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    if args.list_profiles:
        profiles = load_all()
        if not profiles:
            print(f"no profiles yet in {color_file()}")
            return
        print(f"{color_file()}:")
        for name, p in sorted(profiles.items()):
            print(f"  {name:10s} r={p.radius_m * 1000:.0f}mm  H~{p.hue_peak():3d}  "
                  f"thresh {p.threshold:3d}  s_min {p.s_min:3d}  "
                  f"spread H{p.h_spread:g} S{p.s_spread:g}  "
                  f"{p.created}  {p.note}")
        return

    cap = open_camera(args.camera, args.width, args.height)
    locked = False
    if not args.no_lock:
        # Let the camera settle on the scene before freezing it, otherwise we
        # lock in whatever exposure it happened to open with.
        for _ in range(20):
            cap.read()
        print("camera lock: " + ", ".join(lock_camera(cap)))
        locked = True

    try:
        profile = BallColor(
            name=args.profile,
            radius_m=args.radius_mm / 1000.0,
            note=args.note,
            s_min=args.s_min,
            threshold=args.threshold,
            h_spread=args.h_spread,
            s_spread=args.s_spread,
        )
        sampler = _Sampler()
        have_sample = False
        samples: list[np.ndarray] = []

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, sampler.on_mouse)

        print(f"\nCalibrating profile '{profile.name}' "
              f"(ball radius {profile.radius_m * 1000:.0f} mm).")
        print("  drag a box over the ball   sample it (release to apply)")
        print("  a                          add another drag to the same sample")
        print("  h                          print a hue histogram of the scene")
        print("  - / +                      back-projection threshold down / up")
        print("  [ / ]                      saturation floor down / up")
        print("  < / >                      saturation spread down / up")
        print("  ; / '                      hue spread down / up")
        print("  s                          save     q / Esc  quit\n")
        print("Aim for a mask that covers the ball and almost nothing else.")
        print("Then roll the ball into shadow and under other lights and check it")
        print("survives; if it drops out, widen the spread with '>' before saving.\n")

        def resample() -> None:
            nonlocal have_sample
            if not samples:
                return
            try:
                profile.hist = histogram_from_samples(np.vstack(samples), profile.s_min)
                have_sample = True
                print(f"sampled {sum(len(s) for s in samples)} px, "
                      f"peak hue H~{profile.hue_peak()} ({_hue_name(profile.hue_peak())})")
            except ValueError as e:
                print(f"sample rejected: {e}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped returning frames")
                break

            box = sampler.region()
            if box and not sampler.dragging:
                # The window is video | mask, so a drag that strays onto the
                # right half must be clipped to the video, not silently
                # sample nothing.
                box = _clip_box(box, frame.shape)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                patch = hsv[box[1]:box[3], box[0]:box[2]].reshape(-1, 3)
                samples = [patch]  # a fresh drag replaces; 'a' accumulates
                sampler.box = None
                resample()

            cv2.imshow(WIN, _overlay(frame, profile, sampler, have_sample))

            key = read_key()
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                print_scene_histogram(
                    scene_hue_histogram(frame),
                    profile.hue_peak() if have_sample else None,
                )
            elif key in (ord("-"), ord("_")):
                profile.threshold = max(1, profile.threshold - 5)
                print(f"threshold {profile.threshold}")
            elif key in (ord("+"), ord("=")):
                profile.threshold = min(254, profile.threshold + 5)
                print(f"threshold {profile.threshold}")
            elif key == ord("["):
                profile.s_min = max(0, profile.s_min - 5)
                resample()
                print(f"s_min {profile.s_min}")
            elif key == ord("]"):
                profile.s_min = min(254, profile.s_min + 5)
                resample()
                print(f"s_min {profile.s_min}")
            elif key in (ord(","), ord("<")):
                profile.s_spread = max(0.0, profile.s_spread - 1.0)
                print(f"s_spread {profile.s_spread:g} bins")
            elif key in (ord("."), ord(">")):
                profile.s_spread = min(float(S_BINS), profile.s_spread + 1.0)
                print(f"s_spread {profile.s_spread:g} bins")
            elif key in (ord(";"), ord(":")):
                profile.h_spread = max(0.0, profile.h_spread - 0.5)
                print(f"h_spread {profile.h_spread:g} bins ({profile.h_spread * H_RANGE / H_BINS:g} deg)")
            elif key in (ord("'"), ord('"')):
                profile.h_spread = min(H_BINS / 2.0, profile.h_spread + 0.5)
                print(f"h_spread {profile.h_spread:g} bins ({profile.h_spread * H_RANGE / H_BINS:g} deg)")
            elif key == ord("a"):
                b = sampler.region()
                if b:
                    b = _clip_box(b, frame.shape)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    samples.append(hsv[b[1]:b[3], b[0]:b[2]].reshape(-1, 3))
                    resample()
                else:
                    print("drag a box first, then press 'a' to add it")
            elif key == ord("s"):
                if not have_sample:
                    print("nothing sampled yet -- drag a box over the ball first")
                    continue
                path = save_profile(profile)
                print(f"saved profile '{profile.name}' to {path}")
                print(f"  use it with:  uv run track-ball --ball-profile {profile.name}")

    finally:
        # The lock lives in the driver and outlasts this process. Leaving
        # it set hands the next run -- and every other app -- a frozen
        # picture, which on some cameras is a dark green one.
        if locked:
            print("camera unlock: " + ", ".join(unlock_camera(cap)))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
