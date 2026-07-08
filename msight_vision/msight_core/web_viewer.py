import threading
import cv2

from msight_core.nodes import SinkNode
from msight_core.data import DetectionResultsData

try:
    from flask import Flask, Response
except ImportError:
    raise ImportError("flask is required for WebDetectionViewerNode: pip install flask")

_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MSight Detection Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d0d0d;
    color: #d0d0d0;
    font-family: ui-monospace, monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 1.5rem;
    gap: 1rem;
  }
  header { font-size: 0.78rem; letter-spacing: 0.12em; color: #666; text-transform: uppercase; }
  #frame { max-width: 100%; border: 1px solid #2a2a2a; display: block; }
  #status { font-size: 0.72rem; color: #444; }
</style>
</head>
<body>
  <header>MSight &mdash; Detection Viewer</header>
  <img id="frame" src="/stream" alt="stream"
       onload="document.getElementById('status').textContent='connected'"
       onerror="document.getElementById('status').textContent='waiting for stream…'" />
  <span id="status">connecting…</span>
</body>
</html>"""


class _FrameBuffer:
    """Condition-variable frame store: multiple MJPEG clients each wait for the next frame."""

    def __init__(self):
        self._frame = None
        self._cond = threading.Condition()

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._cond.notify_all()

    def get(self, timeout=1.0):
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._frame


class WebDetectionViewerNode(SinkNode):
    """Serves an MJPEG stream of annotated detection frames on an HTTP port.

    Open http://<host>:<port>/ in a browser — no X11 / display required.
    """

    def __init__(self, configs, port: int = 9010):
        super().__init__(configs)
        self.port = port
        self._buf = _FrameBuffer()
        self._start_server()

    def _start_server(self):
        app = Flask(__name__)
        app.logger.disabled = True

        @app.route("/")
        def index():
            return _HTML

        @app.route("/stream")
        def stream():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        thread = threading.Thread(
            target=app.run,
            kwargs={"host": "0.0.0.0", "port": self.port, "threaded": True},
            daemon=True,
        )
        thread.start()
        self.logger.info(f"Web viewer available at http://0.0.0.0:{self.port}")

    def _mjpeg_generator(self):
        while True:
            frame = self._buf.get(timeout=1.0)
            if frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

    def on_message(self, data: DetectionResultsData):
        self.logger.info(f"Received detection results from sensor: {data.sensor_name}")
        frame = data.raw_sensor_data.to_ndarray().copy()
        for obj in data.detection_result.object_list:
            x1, y1, x2, y2 = map(int, obj.box)
            px, py = map(int, obj.pixel_bottom_center)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            label = f"{obj.class_id} {obj.score:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        self._buf.put(frame)
