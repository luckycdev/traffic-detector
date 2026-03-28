from get_cams import fetch_cameras


def load_camera_points(allowed_locations=None):
    allowed = set(allowed_locations) if allowed_locations else None
    camera_points = []

    cameras = fetch_cameras()
    for location, data in cameras.items():
        if allowed is not None and location not in allowed:
            continue

        x = data.get("x")
        y = data.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            camera_points.append({"location": location, "x": x, "y": y})

    return camera_points
