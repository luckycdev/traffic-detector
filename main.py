import cv2
import os
import time
import re
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, send_from_directory
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


def frame_generator():
    cam = None
    current_source = None
    frame_width = 0
    frame_height = 0
    frame_area = 0
    road_mask = None
    road_mask_last_seen = None

    smoothed_coverage = 0
    generator_start_time = time.monotonic()

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
            road_mask_last_seen = None
            smoothed_coverage = 0
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
            road_mask_last_seen = np.zeros((frame_height, frame_width), dtype=np.float64)
            first_mask_seen_time = None

        now = time.monotonic() - generator_start_time

        # fade only pixels that have not been observed for more than 5 seconds
        if road_mask_last_seen is not None:
            stale_mask = (road_mask > 0) & ((now - road_mask_last_seen) > 5.0)
            if np.any(stale_mask):
                faded_values = (road_mask[stale_mask].astype(np.float32) * 0.95).astype(np.uint8)
                road_mask[stale_mask] = faded_values
                road_mask[road_mask < 5] = 0
                road_mask_last_seen[road_mask == 0] = 0.0

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

                # keep active detections fresh; unseen regions age out and fade
                road_mask[y1:y2, x1:x2] = np.maximum(road_mask[y1:y2, x1:x2], 200)
                road_mask_last_seen[y1:y2, x1:x2] = now

        road_area = np.count_nonzero(road_mask)
        if road_area > 0 and first_mask_seen_time is None:
            first_mask_seen_time = now
        elif road_area == 0:
            first_mask_seen_time = None

        coverage_warmup_done = (
            first_mask_seen_time is not None and (now - first_mask_seen_time) >= 0.5
        )
        road_learned_percent = (road_area / frame_area) * 100 if frame_area > 0 else 0
        road_learning_ready = road_area > frame_area * 0.05
        raw_coverage = (boxes_area / road_area) * 100 if road_area > 0 and coverage_warmup_done else 0
        effective_coverage = raw_coverage if road_learning_ready else 0
        smoothed_coverage = smoothed_coverage * 0.8 + effective_coverage * 0.2
        coverage = smoothed_coverage

        traffic_score, traffic_label = traffic_rating(class_counts, coverage)

        with stats_lock:
            live_stats.update(
                {
                    "vehicle_count": vehicle_count,
                    "coverage": round(coverage, 2),
                    "traffic_score": round(traffic_score, 2),
                    "traffic_label": traffic_label,
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
