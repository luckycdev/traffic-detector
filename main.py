import cv2
import os
import time
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

source = "https://5fca316e7c40f.streamlock.net/live-secure/customInstance/M070WBIPC-14-LQ.stream/playlist.m3u8?wowzatokenendtime=1774664606&wowzatokenstarttime=1774661006&wowzatokenhash=zBe77AkVpduJwWx7HAJJzpyjPEeBMvZWXaJzFhiPqoM="

# Use webcam by setting environment variable VIDEO_SOURCE=0.
VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    source,
)

if VIDEO_SOURCE.isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)
model = YOLO("yolo26n.pt")


def frame_generator():
    cam = cv2.VideoCapture(VIDEO_SOURCE)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    fps = int(cam.get(cv2.CAP_PROP_FPS))

    if not cam.isOpened():
        while True:
            fallback = 255 * (cv2.UMat(480, 640, cv2.CV_8UC3).get() * 0)
            cv2.putText(
                fallback,
                "Unable to open video source",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            ok, buffer = cv2.imencode(".jpg", fallback)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.5)

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue

        results = model.predict(frame, classes=[2, 3, 5, 7], verbose=False)

        coverage = 0
        for result in results:
            boxes_area = 0
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
                boxes_area += (x2 - x1) * (y2 - y1)
            coverage = (boxes_area / frame_area) * 100
            print(boxes_area)
            print(frame_area)
            print(round(coverage, 2), "\n")
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Traffic Detector Live</title>
        <style>
          body { font-family: Arial, sans-serif; background: #111; color: #eee; text-align: center; }
          h1 { margin-top: 20px; }
          img { max-width: 95vw; max-height: 80vh; border: 3px solid #2ecc71; border-radius: 8px; }
        </style>
      </head>
      <body>
        <h1>Traffic Detector Output</h1>
        <img src="/video_feed" alt="Live traffic stream" />
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)