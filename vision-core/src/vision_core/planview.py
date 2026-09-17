"""Drawing the field: as an overlay on the video, and as a top-down panel.

`project` and `draw_field` put the field on the camera image; `PlanView` is the
panel beside it, owning the scaling and the furniture while each skill draws its
own marks through `to_px`. Shared for the same reason field.py is -- both skills
draw the identical rectangle, and two copies would drift apart.

Axes: +X right, +Y up, origin at the rectangle's bottom-left. In floor mode that
really is a top-down view; in wall mode it is head-on. Same axes either way.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .field import CameraFieldTransform, Field

FIELD_COLOR = (90, 230, 90)
OUT_COLOR = (90, 110, 255)
GRID_COLOR = (48, 48, 48)
BG_COLOR = (24, 24, 24)

#: BGR. Distinct hues so several objects stay readable at once.
PALETTE = [
    (80, 200, 255),
    (120, 255, 140),
    (255, 160, 90),
    (200, 130, 255),
    (90, 230, 230),
    (255, 210, 120),
]

DEFAULT_WIDTH = 440
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(index: int) -> tuple[int, int, int]:
    return PALETTE[int(index) % len(PALETTE)]


class PlanView:
    """A fixed-size panel with a field rectangle scaled to fit inside it."""

    def __init__(
        self,
        field: Field,
        height: int,
        width: int = DEFAULT_WIDTH,
        mode: str = "wall",
        title: str | None = None,
    ) -> None:
        self.field = field
        self.width = int(width)
        self.height = int(height)
        self.mode = mode
        self.title = title or (
            "TOP-DOWN (field coords)" if mode == "floor" else "FIELD PLANE (head-on)"
        )

        pad, top = 34, 64
        #: Left/right margin in pixels; skills use it to line text up.
        self.pad = pad
        usable_w, usable_h = self.width - 2 * pad, self.height - top - pad
        # Leave a margin outside the rectangle so something that strays out of
        # bounds is still drawn somewhere visible instead of clipped off the panel.
        self.scale = 0.86 * min(usable_w / field.width, usable_h / field.height)
        fw, fh = field.width * self.scale, field.height * self.scale
        self._ox = pad + (usable_w - fw) / 2.0  # origin pixel: bottom-left of rect
        self._oy = top + (usable_h - fh) / 2.0 + fh

    # -- coordinates -------------------------------------------------------

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        """Field metres -> panel pixels."""
        return (
            int(round(self._ox + x * self.scale)),
            int(round(self._oy - y * self.scale)),
        )

    def on_panel(self, px: tuple[int, int]) -> bool:
        return 0 <= px[0] < self.width and 0 <= px[1] < self.height

    def metres(self, m: float) -> int:
        """A length in metres as a length in panel pixels."""
        return int(round(m * self.scale))

    # -- drawing -----------------------------------------------------------

    def base(self, subtitle: str | None = None) -> np.ndarray:
        """A fresh panel: background, title, grid, rectangle, origin and axes."""
        panel = np.full((self.height, self.width, 3), BG_COLOR[0], np.uint8)
        x0 = self.pad - 12

        cv2.putText(panel, self.title, (x0, 26), _FONT, 0.48, (200, 200, 200), 1,
                    cv2.LINE_AA)
        sub = subtitle or f"{self.field.width:g} x {self.field.height:g} m"
        cv2.putText(panel, sub, (x0, 46), _FONT, 0.44, (140, 140, 140), 1, cv2.LINE_AA)

        for seg in self.field.grid(0.2):
            cv2.line(panel, self.to_px(*seg[0][:2]), self.to_px(*seg[1][:2]),
                     GRID_COLOR, 1)
        cv2.rectangle(panel, self.to_px(0, 0),
                      self.to_px(self.field.width, self.field.height),
                      FIELD_COLOR, 1, cv2.LINE_AA)

        o = self.to_px(0, 0)
        cv2.circle(panel, o, 4, FIELD_COLOR, -1, cv2.LINE_AA)
        cv2.putText(panel, "0,0", (o[0] - 26, o[1] + 16), _FONT, 0.4, FIELD_COLOR, 1,
                    cv2.LINE_AA)
        cv2.arrowedLine(panel, o, self.to_px(0.18, 0), (70, 70, 255), 2, cv2.LINE_AA,
                        tipLength=0.3)
        cv2.arrowedLine(panel, o, self.to_px(0, 0.18), (70, 255, 70), 2, cv2.LINE_AA,
                        tipLength=0.3)
        return panel

    def draw_camera(self, panel: np.ndarray, transform: CameraFieldTransform) -> None:
        """Where the camera sits.

        The panel shows only the two in-plane axes, so the out-of-plane distance
        is spelled out rather than implied.
        """
        cam = transform.camera_origin_in_field()
        cpx = self.to_px(float(cam[0]), float(cam[1]))
        if not self.on_panel(cpx):
            return
        cv2.drawMarker(panel, cpx, (170, 170, 170), cv2.MARKER_TRIANGLE_UP, 11, 2)
        cv2.putText(panel, f"cam ({cam[2]:+.2f} m out)", (cpx[0] + 9, cpx[1] + 4),
                    _FONT, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

    def draw_trail(
        self,
        panel: np.ndarray,
        trail: Iterable[tuple[float, float]],
        color: tuple[int, int, int],
    ) -> None:
        """A path in field coords, fading out towards the oldest point."""
        pts = [self.to_px(x, y) for x, y in trail]
        for i in range(1, len(pts)):
            fade = i / len(pts)
            cv2.line(panel, pts[i - 1], pts[i],
                     tuple(int(c * fade * 0.7) for c in color), 1, cv2.LINE_AA)

    def draw_arrow(
        self,
        panel: np.ndarray,
        x: float,
        y: float,
        angle_deg: float,
        length_m: float,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """An arrow from a field point, given a direction and a length in metres."""
        th = np.radians(angle_deg)
        tail = self.to_px(x, y)
        tip = self.to_px(x + length_m * np.cos(th), y + length_m * np.sin(th))
        cv2.arrowedLine(panel, tail, tip, color, thickness, cv2.LINE_AA, tipLength=0.35)


# --------------------------------------------------------------------------
# The same field, overlaid on the camera image.
# --------------------------------------------------------------------------


def project(
    pts_field: np.ndarray, transform: CameraFieldTransform, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Field points -> pixels, plus a flag for whether they are in front of us.

    cv2.projectPoints happily returns coordinates for points behind the camera,
    which then draw as wild lines across the frame. So check the depth ourselves.
    """
    cam = transform.field_to_camera(pts_field)
    in_front = cam[:, 2] > 1e-3
    rvec, tvec = transform.rvec_tvec()
    px, _ = cv2.projectPoints(
        np.asarray(pts_field, np.float64).reshape(-1, 1, 3), rvec, tvec, K, np.zeros(5)
    )
    return px.reshape(-1, 2), in_front


def draw_field(
    frame: np.ndarray,
    field: Field,
    transform: CameraFieldTransform,
    K: np.ndarray,
    show_grid: bool = True,
) -> None:
    """The virtual field outline, grid, origin and axes, overlaid on the video."""
    if show_grid:
        for seg in field.grid(0.2):
            px, ok = project(seg, transform, K)
            if ok.all():
                cv2.line(frame, tuple(px[0].astype(int)), tuple(px[1].astype(int)),
                         (60, 120, 60), 1, cv2.LINE_AA)

    corners, ok = project(field.corners(), transform, K)
    if ok.all():
        cv2.polylines(frame, [corners.astype(np.int32)], True, FIELD_COLOR, 2,
                      cv2.LINE_AA)
        for i, name in enumerate(["(0,0)", "(W,0)", "(W,H)", "(0,H)"]):
            p = corners[i].astype(int)
            cv2.circle(frame, tuple(p), 4, FIELD_COLOR, -1, cv2.LINE_AA)
            cv2.putText(frame, name, (p[0] + 6, p[1] - 6), _FONT, 0.45, FIELD_COLOR,
                        1, cv2.LINE_AA)
    elif ok.any():
        cv2.putText(frame, "field partly behind camera", (12, 52), _FONT, 0.6,
                    OUT_COLOR, 2, cv2.LINE_AA)

    # Field axes at the origin corner: X red, Y green (0.2 m each).
    axes = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]], np.float64)
    px, ok = project(axes, transform, K)
    if ok.all():
        o = tuple(px[0].astype(int))
        cv2.arrowedLine(frame, o, tuple(px[1].astype(int)), (70, 70, 255), 2,
                        cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(frame, o, tuple(px[2].astype(int)), (70, 255, 70), 2,
                        cv2.LINE_AA, tipLength=0.25)
