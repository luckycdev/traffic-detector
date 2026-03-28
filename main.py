import cv2
import os
import time
import re
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request
import numpy as np
from threading import Lock
from get_cams import fetch_cameras

app = Flask(__name__)

# default to I-70 WB At Prospect Ave
source = "https://traveler.modot.org/tisvc/api/Tms/CameraStream/M070WBC-09-LQ"
DEFAULT_CAMERA_NAME = "I-70 WB At Prospect Ave"

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", source)

if VIDEO_SOURCE.isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)

model = YOLO("yolo26n.pt")


def normalize_video_source(source_value):
    if isinstance(source_value, int):
        return source_value
    if source_value is None:
        return None
    source_text = str(source_value).strip()
    if source_text.isdigit():
        return int(source_text)
    return source_text


def extract_stream_from_html(html_value):
    if not html_value:
        return None
    text = str(html_value)
    stream_patterns = [
        r"https?://[^\s\"'<>]*CameraStream/[^\s\"'<>]+",
        r"https?://[^\s\"'<>]+\.m3u8[^\s\"'<>]*",
        r"src=[\"']([^\"']+)[\"']",
    ]

    for pattern in stream_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1) if match.lastindex else match.group(0)
        if candidate.startswith("//"):
            candidate = f"https:{candidate}"
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate

    return None


def load_camera_sources():
    camera_map = {}
    try:
        cameras = fetch_cameras()
        for location, data in cameras.items():
            stream_url = extract_stream_from_html(data.get("html"))
            if stream_url:
                camera_map[location] = normalize_video_source(stream_url)
    except Exception as exc:
        print("Failed to load camera list:", exc)

    if DEFAULT_CAMERA_NAME not in camera_map:
        camera_map[DEFAULT_CAMERA_NAME] = normalize_video_source(source)

    return camera_map


camera_sources = load_camera_sources()
source_lock = Lock()
active_camera = DEFAULT_CAMERA_NAME
default_source = normalize_video_source(VIDEO_SOURCE)
if active_camera not in camera_sources:
    for camera_name, camera_source in camera_sources.items():
        if camera_source == default_source:
            active_camera = camera_name
            break

if active_camera not in camera_sources and camera_sources:
    active_camera = next(iter(camera_sources))

active_video_source = camera_sources.get(active_camera, default_source)

stats_lock = Lock()
live_stats = {
    "vehicle_count": 0,
    "coverage": 0.0,
    "raw_coverage": 0.0,
    "road_learned_percent": 0.0,
    "boxes_area": 0,
    "road_area": 0,
    "frame_area": 0,
    "road_learning_ready": False,
    "class_counts": {},
    "last_updated": "",
    "selected_camera": active_camera,
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def frame_generator():
    cam = None
    current_source = None
    frame_width = 0
    frame_height = 0
    frame_area = 0
    road_mask = None

    smoothed_coverage = 0

    mask_fade_start_time = time.time()
    first_mask_seen_time = None

    while True:
        with source_lock:
            desired_source = active_video_source
            desired_camera = active_camera

        if desired_source != current_source:
            if cam is not None:
                cam.release()
            cam = cv2.VideoCapture(desired_source)
            current_source = desired_source
            frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_area = frame_width * frame_height
            road_mask = None
            smoothed_coverage = 0
            mask_fade_start_time = time.time()
            first_mask_seen_time = None
            with stats_lock:
                live_stats["frame_area"] = int(frame_area)
                live_stats["selected_camera"] = desired_camera

        if cam is None or not cam.isOpened():
            fallback = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                fallback,
                f"Unable to open source: {desired_camera}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                fallback,
                str(desired_source)[:80],
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
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
            continue

        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        with stats_lock:
            live_stats["frame_area"] = int(frame_area)
            live_stats["selected_camera"] = desired_camera

        results = model.predict(frame, classes=[2, 3, 5, 7], verbose=False)

        if road_mask is None or road_mask.shape != (frame_height, frame_width):
            road_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            mask_fade_start_time = time.time()
            first_mask_seen_time = None

        # fade old mask after 5 seconds -- TODO: could show that traffic is heavy if for example only one lane has cars in it
        elapsed_time = time.time() - mask_fade_start_time
        if elapsed_time > 5:
            road_mask = np.clip(road_mask * 0.95, 0, 255).astype(np.uint8)
            road_mask[road_mask < 5] = 0

        boxes_area = 0
        vehicle_count = 0
        class_counts = {}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # ignore upper part (reduces sky noise)
                if y2 < frame_height * 0.3:
                    continue

                vehicle_count += 1
                class_name = model.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                label = f"{class_name} {conf:.2f}"

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

                area = (x2 - x1) * (y2 - y1)
                boxes_area += area

                # update road mask dynamically
                road_mask[y1:y2, x1:x2] = np.maximum(road_mask[y1:y2, x1:x2], 200)

        road_area = np.count_nonzero(road_mask)
        now = time.time()
        if road_area > 0 and first_mask_seen_time is None:
            first_mask_seen_time = now

        coverage_warmup_done = (
            first_mask_seen_time is not None and (now - first_mask_seen_time) >= 0.5
        )
        road_learned_percent = (road_area / frame_area) * 100 if frame_area > 0 else 0
        road_learning_ready = road_area > frame_area * 0.05
        raw_coverage = (boxes_area / road_area) * 100 if road_area > 0 and coverage_warmup_done else 0
        effective_coverage = raw_coverage if road_learning_ready else 0
        smoothed_coverage = smoothed_coverage * 0.8 + effective_coverage * 0.2
        coverage = smoothed_coverage

        with stats_lock:
            live_stats.update(
                {
                    "vehicle_count": vehicle_count,
                    "coverage": round(coverage, 2),
                    "raw_coverage": round(raw_coverage, 2),
                    "road_learned_percent": round(road_learned_percent, 2),
                    "boxes_area": int(boxes_area),
                    "road_area": int(road_area),
                    "frame_area": int(frame_area),
                    "road_learning_ready": bool(road_learning_ready),
                    "class_counts": class_counts,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # display road mask
        mask_colored = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.4, 0)

        #print("boxes_area:", boxes_area)
        #print("road_area:", road_area)
        #print("coverage:", round(coverage, 2), "\n")

        # overlay coverage on screen
        cv2.putText(
            frame,
            f"Coverage: {coverage:.2f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

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
                    body { font-family: Arial, sans-serif; background: #111; color: #eee; text-align: center; margin: 0; padding: 0 12px 24px; }
          h1 { margin-top: 20px; }
          img { max-width: 95vw; max-height: 80vh; border: 3px solid #2ecc71; border-radius: 8px; }
                    .stats-wrap { max-width: 1000px; margin: 16px auto 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
                    .stat-card { background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 10px; }
                    .stat-title { font-size: 0.85rem; color: #9aa0a6; margin-bottom: 6px; }
                    .stat-value { font-size: 1.2rem; font-weight: 700; }
                    .detail-panel { max-width: 1000px; margin: 12px auto 0; background: #1b1b1b; border: 1px solid #333; border-radius: 8px; padding: 10px; text-align: left; }
                    .class-list { margin: 8px 0 0; padding-left: 20px; }
                    .muted { color: #9aa0a6; }
                    .controls { max-width: 1000px; margin: 10px auto 0; display: flex; gap: 8px; justify-content: center; align-items: center; flex-wrap: wrap; }
                    .controls select, .controls button { background: #1b1b1b; color: #eee; border: 1px solid #333; border-radius: 6px; padding: 8px 10px; }
                    .controls button { cursor: pointer; }
        </style>
      </head>
      <body>
        <h1>Traffic Detector Output</h1>
                <div class="controls">
                    <label for="camera_select">Camera:</label>
                    <select id="camera_select"></select>
                    <button id="camera_apply" type="button">Switch Camera</button>
                    <span id="camera_status" class="muted"></span>
                </div>
        <img src="/video_feed" alt="Live traffic stream" />
                <div class="stats-wrap">
                    <div class="stat-card"><div class="stat-title">Vehicles (Current Frame)</div><div class="stat-value" id="vehicle_count">0</div></div>
                    <div class="stat-card"><div class="stat-title">Smoothed Coverage</div><div class="stat-value" id="coverage">0.00%</div></div>
                    <div class="stat-card"><div class="stat-title">Raw Coverage</div><div class="stat-value" id="raw_coverage">0.00%</div></div>
                </div>
                <div class="detail-panel">
                    <div><strong>Road Learning:</strong> <span id="road_learning_ready">Not ready</span></div>
                    <div><strong>Total Road Learned:</strong> <span id="road_learned_percent">0.00%</span></div>
                    <div><strong>Last Updated:</strong> <span id="last_updated" class="muted">-</span></div>
                    <div style="margin-top: 6px;"><strong>Detected Vehicle Types</strong></div>
                    <ul id="class_counts" class="class-list"></ul>
                </div>
                <script>
                    async function loadCameras() {
                        try {
                            const response = await fetch('/cameras', { cache: 'no-store' });
                            if (!response.ok) return;
                            const payload = await response.json();
                            const cameraSelect = document.getElementById('camera_select');
                            cameraSelect.innerHTML = '';

                            for (const cameraName of payload.cameras || []) {
                                const option = document.createElement('option');
                                option.value = cameraName;
                                option.textContent = cameraName;
                                cameraSelect.appendChild(option);
                            }

                            if (payload.selected_camera) {
                                cameraSelect.value = payload.selected_camera;
                                document.getElementById('camera_status').textContent = `Current: ${payload.selected_camera}`;
                            }
                        } catch (error) {
                            console.error('Failed to load cameras:', error);
                        }
                    }

                    async function switchCamera() {
                        const cameraSelect = document.getElementById('camera_select');
                        const selected = cameraSelect.value;
                        if (!selected) return;

                        try {
                            const response = await fetch('/select_camera', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ camera: selected }),
                                cache: 'no-store'
                            });

                            const payload = await response.json();
                            if (!response.ok) {
                                document.getElementById('camera_status').textContent = payload.error || 'Failed to switch camera';
                                return;
                            }

                            document.getElementById('camera_status').textContent = `Current: ${payload.selected_camera}`;
                        } catch (error) {
                            console.error('Failed to switch camera:', error);
                        }
                    }

                    async function refreshStats() {
                        try {
                            const response = await fetch(`/stats?t=${Date.now()}`, { cache: 'no-store' });
                            if (!response.ok) return;
                            const data = await response.json();

                            document.getElementById('vehicle_count').textContent = data.vehicle_count;
                            document.getElementById('coverage').textContent = `${data.coverage.toFixed(2)}%`;
                            document.getElementById('raw_coverage').textContent = `${data.raw_coverage.toFixed(2)}%`;
                            document.getElementById('road_learning_ready').textContent = data.road_learning_ready ? 'Ready' : 'Not ready';
                            document.getElementById('road_learned_percent').textContent = `${data.road_learned_percent.toFixed(2)}%`;
                            document.getElementById('last_updated').textContent = data.last_updated || '-';

                            const classList = document.getElementById('class_counts');
                            classList.innerHTML = '';
                            const entries = Object.entries(data.class_counts || {});
                            if (!entries.length) {
                                const li = document.createElement('li');
                                li.className = 'muted';
                                li.textContent = 'No vehicles detected in current frame';
                                classList.appendChild(li);
                            } else {
                                entries.sort((a, b) => b[1] - a[1]);
                                for (const [name, count] of entries) {
                                    const li = document.createElement('li');
                                    li.textContent = `${name}: ${count}`;
                                    classList.appendChild(li);
                                }
                            }
                        } catch (error) {
                            console.error('Failed to fetch stats:', error);
                        }
                    }

                                        document.getElementById('camera_apply').addEventListener('click', switchCamera);
                                        loadCameras();
                    setInterval(refreshStats, 200);
                    refreshStats();
                </script>
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cameras")
def cameras():
    with source_lock:
        selected_camera = active_camera
    return jsonify(
        {
            "cameras": list(camera_sources.keys()),
            "selected_camera": selected_camera,
        }
    )


@app.route("/select_camera", methods=["POST"])
def select_camera():
    data = request.get_json(silent=True) or {}
    selected_camera = data.get("camera")

    if selected_camera not in camera_sources:
        return jsonify({"error": "Unknown camera"}), 400

    global active_camera, active_video_source
    with source_lock:
        active_camera = selected_camera
        active_video_source = camera_sources[selected_camera]

    return jsonify({"ok": True, "selected_camera": active_camera})


@app.route("/stats")
def stats():
    with stats_lock:
        snapshot = dict(live_stats)
    response = jsonify(to_jsonable(snapshot))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)