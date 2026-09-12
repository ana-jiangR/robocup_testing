"""Serve an AprilTag to your phone at a known physical size.

Run this on the laptop, open the printed URL on the phone, and the page will
size the tag in real millimetres. There is no printer and no ruler in the loop:
the phone is calibrated against a bank card, which is manufactured to
ISO/IEC 7810 ID-1 -- 85.60 x 53.98 mm, the same in every wallet in the world.

    uv run serve-tag
    uv run serve-tag --tag-id 7 --size-mm 80

The size the page reports is the edge-to-edge width of the BLACK SQUARE, which
is exactly what the detector means by tag size. Pass it to the tracker.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import socket
import socketserver

import cv2
import numpy as np

MODULES = 8  # tag36h11: 6 data bits + a 1-module black border ring


def tag_png_base64(tag_id: int, px_per_module: int = 64) -> str:
    """The 8x8-module black square as a base64 PNG, no quiet zone baked in."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    img = cv2.aruco.generateImageMarker(d, tag_id, MODULES * px_per_module, 1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode tag PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def lan_ip() -> str:
    """Best guess at this machine's address on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


PAGE = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>AprilTag __TAGID__</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; -webkit-user-select: none; user-select: none; }
  body { margin: 0; background: #fff; color: #111;
         font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .wrap { padding: 16px; max-width: 560px; margin: 0 auto; }
  h1 { font-size: 17px; margin: 0 0 4px; }
  p.sub { margin: 0 0 16px; color: #555; font-size: 13px; }
  .step { border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 14px; }
  .step h2 { font-size: 14px; margin: 0 0 10px; text-transform: uppercase;
             letter-spacing: .04em; color: #666; }
  #card { height: 53.98mm; border: 2px dashed #c00; border-radius: 3.5mm;
          background: repeating-linear-gradient(45deg,#fff,#fff 8px,#fafafa 8px,#fafafa 16px);
          position: relative; }
  #card span { position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%);
               font-size: 11px; color: #c00; white-space: nowrap; }
  input[type=range] { width: 100%; margin: 14px 0 4px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .row label { font-size: 13px; color: #555; }
  input[type=number] { width: 5.5em; padding: 6px; font-size: 15px;
                       border: 1px solid #ccc; border-radius: 6px; }
  .readout { font-variant-numeric: tabular-nums; font-size: 13px; color: #555; }
  code { background: #f2f2f2; padding: 2px 6px; border-radius: 4px;
         font-size: 13px; user-select: all; -webkit-user-select: all; }
  button { font-size: 15px; padding: 10px 14px; border-radius: 8px;
           border: 1px solid #bbb; background: #f7f7f7; }
  button.primary { background: #111; color: #fff; border-color: #111; }
  /* The tag itself: white quiet zone around a hard-edged black square. */
  #tagbox { background: #fff; display: inline-block; line-height: 0; }
  #tagbox img { image-rendering: pixelated; image-rendering: crisp-edges; display: block; }
  #stage { text-align: center; padding: 10px 0; }
  #ruler { height: 8mm; border-left: 2px solid #06c; border-right: 2px solid #06c;
           border-bottom: 2px solid #06c; margin-top: 10px; }
  /* Presentation mode: nothing but the tag on white. */
  body.present .wrap > *:not(#stage) { display: none; }
  body.present { background: #fff; }
  body.present .wrap { padding: 0; max-width: none; }
  body.present #stage { position: fixed; inset: 0; display: flex;
                        align-items: center; justify-content: center; }
  #exit { position: fixed; top: 0; right: 0; width: 64px; height: 64px;
          opacity: 0; display: none; }
  body.present #exit { display: block; }
</style>
</head><body>
<div class="wrap">

  <h1>AprilTag <span id="hid">__TAGID__</span> &middot; tag36h11</h1>
  <p class="sub">Turn screen brightness up, and turn auto-rotate off.</p>

  <div class="step">
    <h2>1 &middot; Calibrate this screen</h2>
    <p style="margin:0 0 6px;font-size:13px">Hold any bank card against the box and
      drag until the box matches the card exactly.</p>
    <div id="card"><span>85.60 &times; 53.98 mm</span></div>
    <input type="range" id="slider" min="120" max="900" value="323" step="0.5">
    <div class="readout" id="calread"></div>
  </div>

  <div class="step">
    <h2>2 &middot; Tag size</h2>
    <div class="row">
      <label for="mm">Black square</label>
      <input type="number" id="mm" value="__SIZEMM__" min="20" max="400" step="1">
      <span>mm</span>
    </div>
    <div class="readout" style="margin-top:10px">
      Pass to the tracker: <code id="cli">--tag-size 0.080</code>
    </div>
    <p style="margin:10px 0 0;font-size:13px;color:#555">Optional sanity check with
      a ruler &mdash; this bar should measure exactly <b>50 mm</b>:</p>
    <div id="ruler"></div>
  </div>

  <div class="step">
    <h2>3 &middot; Show it</h2>
    <div class="row">
      <button class="primary" id="go">Full screen tag</button>
      <span class="readout">Tap the top-right corner to come back.</span>
    </div>
  </div>

  <div id="stage">
    <div id="tagbox"><img id="tag" src="data:image/png;base64,__PNG__" alt=""></div>
  </div>
  <div id="exit"></div>
</div>
<script>
const MODULES = __MODULES__, CARD_MM = 85.60;
const card = document.getElementById('card'), slider = document.getElementById('slider');
const mmIn = document.getElementById('mm'), tag = document.getElementById('tag');
const box = document.getElementById('tagbox'), calread = document.getElementById('calread');
const cli = document.getElementById('cli'), ruler = document.getElementById('ruler');

function load(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : +v; }
                      catch (e) { return d; } }
function save(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

let cardPx = load('cardPx', 323);
let sizeMm = load('sizeMm', +mmIn.value);
slider.value = cardPx; mmIn.value = sizeMm;

function render() {
  const pxPerMm = cardPx / CARD_MM;
  card.style.width = cardPx + 'px';
  card.style.height = (53.98 * pxPerMm) + 'px';
  // The image is the black square; the quiet zone is padding outside it.
  const side = sizeMm * pxPerMm;
  tag.style.width = side + 'px';
  tag.style.height = side + 'px';
  box.style.padding = (2 * sizeMm / MODULES * pxPerMm) + 'px';  // 2 modules of white
  ruler.style.width = (50 * pxPerMm) + 'px';
  calread.textContent = pxPerMm.toFixed(2) + ' css px per mm'
    + '  ·  ' + (sizeMm / MODULES).toFixed(2) + ' mm per module';
  cli.textContent = '--tag-size ' + (sizeMm / 1000).toFixed(4);
}
slider.addEventListener('input', () => { cardPx = +slider.value; save('cardPx', cardPx); render(); });
mmIn.addEventListener('input', () => {
  const v = +mmIn.value; if (v >= 20 && v <= 400) { sizeMm = v; save('sizeMm', v); render(); }
});
document.getElementById('go').addEventListener('click', async () => {
  document.body.classList.add('present');
  try { await document.documentElement.requestFullscreen(); } catch (e) {}
  try { navigator.wakeLock && await navigator.wakeLock.request('screen'); } catch (e) {}
});
document.getElementById('exit').addEventListener('click', async () => {
  document.body.classList.remove('present');
  try { document.fullscreenElement && await document.exitFullscreen(); } catch (e) {}
});
render();
</script>
</body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    page: bytes = b""
    info: dict = {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/info"):
            body = json.dumps(self.info).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = self.page, "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass  # quiet; the console is for the instructions


def build_page(tag_id: int, size_mm: float) -> bytes:
    return (
        PAGE.replace("__TAGID__", str(tag_id))
        .replace("__SIZEMM__", f"{size_mm:g}")
        .replace("__MODULES__", str(MODULES))
        .replace("__PNG__", tag_png_base64(tag_id))
    ).encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag-id", type=int, default=0, help="tag36h11 id (default 0)")
    ap.add_argument(
        "--size-mm",
        type=float,
        default=80.0,
        help="initial black-square width in mm (default 80)",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--save-png",
        metavar="PATH",
        help="also write the tag as a PNG, if you would rather print it",
    )
    args = ap.parse_args()

    if args.save_png:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        img = cv2.aruco.generateImageMarker(d, args.tag_id, MODULES * 64, 1)
        pad = 2 * 64  # 2 modules of quiet zone
        img = cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
        )
        cv2.imwrite(args.save_png, img)
        print(f"wrote {args.save_png}  (black square is 8/12 of the image width)")

    Handler.page = build_page(args.tag_id, args.size_mm)
    Handler.info = {"tag_id": args.tag_id, "family": "tag36h11", "modules": MODULES}

    ip = lan_ip()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"\nTag {args.tag_id} (tag36h11) is being served.\n")
        print(f"  On your phone, open:   http://{ip}:{args.port}/")
        print(f"  (phone and laptop must be on the same wifi)\n")
        print("Then: match the box to a bank card, set the size, tap 'Full screen tag'.")
        print("The page prints the --tag-size value to hand to the tracker.\n")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
