import cv2
import math
import os
import time
import re
import logging
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, send_from_directory
import numpy as np
from threading import Lock, Thread
from dotenv import load_dotenv
from get_cams import fetch_cameras
from maps import load_camera_points

load_dotenv()

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = os.getenv("SERVER_PORT", "5050")
DEFAULT_STREAM_SOURCE = os.getenv("DEFAULT_STREAM_SOURCE", "0")
DEFAULT_CAMERA_NAME = os.getenv("DEFAULT_CAMERA_NAME", "Default Camera")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")

app = Flask(__name__)

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", DEFAULT_STREAM_SOURCE)

if VIDEO_SOURCE.isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)

# YOLO 26 nano
model = YOLO(YOLO_MODEL)


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
        camera_map[DEFAULT_CAMERA_NAME] = normalize_video_source(DEFAULT_STREAM_SOURCE)

    return camera_map


camera_sources = load_camera_sources()
default_source = normalize_video_source(VIDEO_SOURCE)


def resolve_default_camera_name():
    if DEFAULT_CAMERA_NAME in camera_sources:
        return DEFAULT_CAMERA_NAME

    for camera_name, camera_source in camera_sources.items():
        if camera_source == default_source:
            return camera_name

    if camera_sources:
        return next(iter(camera_sources))

    return DEFAULT_CAMERA_NAME


default_camera_name = resolve_default_camera_name()
model_lock = Lock()
workers_lock = Lock()
camera_workers = {}
WORKER_IDLE_TIMEOUT_SECONDS = 10


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

def traffic_rating(class_counts, coverage):
    """Calculates numerical (0-10) and text based traffic rating using formula based on vehicle coverage
    on road, vehicle type, and number of vehicles"""
    baseline_coverage = 6.5

    adjusted_coverage = max(coverage - baseline_coverage, 0)
    adjusted_max = 100 - baseline_coverage

    car_wt = 1.0
    motorcycle_wt = 0.5
    bus_wt = 2.5
    truck_wt = 3.0

    car_count = class_counts.get("car", 0)
    motorcycle_count = class_counts.get("motorcycle", 0)
    bus_count = class_counts.get("bus", 0)
    truck_count = class_counts.get("truck", 0)
    total_count = car_count + motorcycle_count + bus_count + truck_count

    weighted_count = (car_count * car_wt) + (motorcycle_count * motorcycle_wt) + (bus_count * bus_wt) + (truck_count * truck_wt)
    weight_factor = weighted_count / max(total_count, 1.0)

    num_traffic_score = (adjusted_coverage / adjusted_max) * weight_factor

    # scale to 0-10 range
    num_traffic_score_0_to_10 = min(num_traffic_score * 10, 10)

    if num_traffic_score_0_to_10 <= 2.5:
        text_traffic_score = "Light Traffic"
    elif num_traffic_score_0_to_10 <= 5.0:
        text_traffic_score = "Moderate Traffic"
    elif num_traffic_score_0_to_10 <= 7.5:
        text_traffic_score = "Heavy Traffic"
    else:
        text_traffic_score = "Very Heavy Traffic"

    return num_traffic_score_0_to_10, text_traffic_score


def vehicle_movement_rating(current_positions, previous_positions,
                            stopped_threshold=0.5, slow_threshold=4.0,
                            max_match_distance=100.0):
    """Returns count of vehicles classified as stopped, slow, or fast based on
    pixel displacement between frames. Uses nearest-neighbor matching so it
    works when ByteTrack cannot assign stable track IDs."""

    # Stores count of vehicles in each movement category
    movement_counts = {"stopped": 0, "slow": 0, "fast": 0}
    if not previous_positions:
        return movement_counts

    # Copy so matched positions can be removed without modifying original
    available = list(previous_positions)

    for (cx, cy) in current_positions:
        best_dist = float("inf")
        best_idx = -1

        # Find the closest previous position to this vehicle
        for i, (px, py) in enumerate(available):
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = i

        # Skip if no match found or nearest match is too far away
        if best_idx < 0 or best_dist > max_match_distance:
            continue

        # Remove matched position so it can't be matched again
        available.pop(best_idx)

        # Classify displacement
        if best_dist < stopped_threshold:
            movement_counts["stopped"] += 1
        elif best_dist < slow_threshold:
            movement_counts["slow"] += 1
        else:
            movement_counts["fast"] += 1

    return movement_counts


def get_empty_stats(camera_name):
    return {
        "vehicle_count": 0,
        "coverage": 0.0,
        "raw_coverage": 0.0,
        "fps": 0.0,
        "resolution": None,
        "road_mask_percent": 0.0,
        "boxes_area": 0,
        "road_area": 0,
        "frame_area": 0,
        "road_learning_ready": False,
        "class_counts": {},
        "movement_counts": {"stopped": 0, "slow": 0, "fast": 0},
        "last_updated": "",
        "selected_camera": camera_name,
    }


class CameraWorker:
    def __init__(self, camera_name, camera_source):
        self.camera_name = camera_name
        self.camera_source = camera_source
        self.lock = Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_stats = get_empty_stats(camera_name)
        self.active_viewers = 0
        self.last_accessed = time.monotonic()
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def touch(self):
        with self.lock:
            self.last_accessed = time.monotonic()

    def add_viewer(self):
        with self.lock:
            self.active_viewers += 1
            self.last_accessed = time.monotonic()

    def remove_viewer(self):
        with self.lock:
            self.active_viewers = max(0, self.active_viewers - 1)
            self.last_accessed = time.monotonic()

    def should_stop(self):
        with self.lock:
            idle_seconds = time.monotonic() - self.last_accessed
            return self.active_viewers == 0 and idle_seconds > WORKER_IDLE_TIMEOUT_SECONDS

    def run(self):
        cam = cv2.VideoCapture(self.camera_source)
        road_mask = None
        road_mask_last_seen = None
        smoothed_coverage = 0
        smoothed_fps = 0.0
        previous_frame_time = None
        first_mask_seen_time = None
        previous_positions = []
        generator_start_time = time.monotonic()

        while True:
            if self.should_stop():
                break

            if cam is None or not cam.isOpened():
                fallback = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    fallback,
                    f"Unable to open source: {self.camera_name}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    fallback,
                    str(self.camera_source)[:80],
                    (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                ok, buffer = cv2.imencode(".jpg", fallback)
                if ok:
                    with self.lock:
                        self.latest_frame = buffer.tobytes()
                        self.latest_frame_id += 1
                        self.latest_stats = get_empty_stats(self.camera_name)
                        self.latest_stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                time.sleep(0.5)
                cam = cv2.VideoCapture(self.camera_source)
                road_mask = None
                road_mask_last_seen = None
                smoothed_coverage = 0
                first_mask_seen_time = None
                generator_start_time = time.monotonic()
                continue

            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height
            frame_time = time.monotonic()
            if previous_frame_time is not None:
                delta_seconds = frame_time - previous_frame_time
                if delta_seconds > 0:
                    instant_fps = 1.0 / delta_seconds
                    smoothed_fps = smoothed_fps * 0.8 + instant_fps * 0.2
            previous_frame_time = frame_time

            with model_lock:
                results = model.predict(
                    frame,
                    conf = 0.15,
                    classes = [2, 3, 5, 7]
                )

            if road_mask is None or road_mask.shape != (frame_height, frame_width):
                road_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                road_mask_last_seen = np.zeros((frame_height, frame_width), dtype=np.float64)
                first_mask_seen_time = None

            now = time.monotonic() - generator_start_time

            stale_mask = (road_mask > 0) & ((now - road_mask_last_seen) > 5.0)
            if np.any(stale_mask):
                faded_values = (road_mask[stale_mask].astype(np.float32) * 0.95).astype(np.uint8)
                road_mask[stale_mask] = faded_values
                road_mask[road_mask < 5] = 0
                road_mask_last_seen[road_mask == 0] = 0.0

            boxes_area = 0
            vehicle_count = 0
            class_counts = {}
            current_positions = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if y2 < frame_height * 0.3:
                        continue

                    vehicle_count += 1
                    class_name = model.names[cls]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_positions.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, min(frame_height - 8, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

                    boxes_area += (x2 - x1) * (y2 - y1)
                    road_mask[y1:y2, x1:x2] = np.maximum(road_mask[y1:y2, x1:x2], 200)
                    road_mask_last_seen[y1:y2, x1:x2] = now

            movement_counts = vehicle_movement_rating(current_positions, previous_positions)
            previous_positions = current_positions

            road_area = np.count_nonzero(road_mask)
            if road_area > 0 and first_mask_seen_time is None:
                first_mask_seen_time = now
            elif road_area == 0:
                first_mask_seen_time = None

            coverage_warmup_done = (
                first_mask_seen_time is not None and (now - first_mask_seen_time) >= 0.5
            )
            road_mask_percent = (road_area / frame_area) * 100 if frame_area > 0 else 0
            road_learning_ready = road_area > frame_area * 0.05
            raw_coverage = (boxes_area / road_area) * 100 if road_area > 0 and coverage_warmup_done else 0
            effective_coverage = raw_coverage if road_learning_ready else 0
            smoothed_coverage = smoothed_coverage * 0.8 + effective_coverage * 0.2
            coverage = smoothed_coverage

            traffic_score, traffic_label = traffic_rating(class_counts, coverage)

            mask_colored = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.4, 0)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            stats_snapshot = {
                "vehicle_count": vehicle_count,
                "coverage": round(coverage, 2),
                "raw_coverage": round(raw_coverage, 2),
                "traffic_score": round(traffic_score, 2),
                "traffic_label": traffic_label,
                "fps": round(smoothed_fps, 2),
                "resolution": f"{frame_width}x{frame_height}",
                "road_mask_percent": round(road_mask_percent, 2),
                "boxes_area": int(boxes_area),
                "road_area": int(road_area),
                "frame_area": int(frame_area),
                "road_learning_ready": bool(road_learning_ready),
                "class_counts": class_counts,
                "movement_counts": movement_counts,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "selected_camera": self.camera_name,
            }

            with self.lock:
                self.latest_frame = buffer.tobytes()
                self.latest_frame_id += 1
                self.latest_stats = stats_snapshot

        if cam is not None:
            cam.release()

        with workers_lock:
            if camera_workers.get(self.camera_name) is self:
                camera_workers.pop(self.camera_name, None)


def get_requested_camera_name():
    camera_name = request.args.get("camera", default_camera_name)
    if camera_name not in camera_sources:
        return default_camera_name
    return camera_name


def get_or_create_worker(camera_name):
    with workers_lock:
        worker = camera_workers.get(camera_name)
        if worker is None or not worker.thread.is_alive():
            worker = CameraWorker(camera_name, camera_sources[camera_name])
            camera_workers[camera_name] = worker
    worker.touch()
    return worker


def get_existing_worker(camera_name):
    with workers_lock:
        worker = camera_workers.get(camera_name)
        if worker is None:
            return None
        if not worker.thread.is_alive():
            camera_workers.pop(camera_name, None)
            return None
        return worker


def stream_worker_frames(worker):
    last_sent_frame_id = -1
    worker.add_viewer()
    try:
        while True:
            worker.touch()
            with worker.lock:
                frame_bytes = worker.latest_frame
                frame_id = worker.latest_frame_id
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            if frame_id == last_sent_frame_id:
                time.sleep(0.01)
                continue
            last_sent_frame_id = frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        worker.remove_viewer()


@app.route("/")
def index():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root_dir, "index.html")


@app.route("/index.css")
def index_css():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root_dir, "index.css")


@app.route("/index.js")
def index_js():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root_dir, "index.js")


@app.route("/video_feed")
def video_feed():
    camera_name = get_requested_camera_name()
    worker = get_or_create_worker(camera_name)
    return Response(stream_worker_frames(worker), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cameras")
def cameras():
    return jsonify(
        {
            "cameras": list(camera_sources.keys()),
            "selected_camera": default_camera_name,
        }
    )


@app.route("/map_cameras")
def map_cameras():
    try:
        points = load_camera_points(allowed_locations=camera_sources.keys())
    except Exception as exc:
        return jsonify({"error": f"Failed to load map cameras: {exc}"}), 500

    return jsonify(
        {
            "cameras": points,
            "selected_camera": default_camera_name,
        }
    )


@app.route("/select_camera", methods=["POST"])
def select_camera():
    data = request.get_json(silent=True) or {}
    selected_camera = data.get("camera")

    if selected_camera not in camera_sources:
        return jsonify({"error": "Unknown camera"}), 400

    return jsonify({"ok": True, "selected_camera": selected_camera})


@app.route("/stats")
def stats():
    camera_name = get_requested_camera_name()
    worker = get_or_create_worker(camera_name)
    with worker.lock:
        snapshot = dict(worker.latest_stats)

    if "traffic_score" not in snapshot:
        snapshot["traffic_score"] = 0.0
    if "traffic_label" not in snapshot:
        snapshot["traffic_label"] = "Light Traffic"

    response = jsonify(to_jsonable(snapshot))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"Server running at http://{SERVER_HOST}:{SERVER_PORT}", flush=True)
    app.run(host=SERVER_HOST, port=int(SERVER_PORT), debug=False)