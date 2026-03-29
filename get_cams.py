import requests
import os


def load_env_file(env_filename=".env"):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_filename)
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

CAMERAS_JSON_URL = os.getenv(
    "CAMERAS_JSON_URL",
    "https://traveler.modot.org/timconfig/feed/desktop/StreamingCams2.json",
)

def fetch_cameras():
    response = requests.get(CAMERAS_JSON_URL)
    response.raise_for_status()  # will raise error if request fails
    cams = response.json()
    
    cam_dict = {}
    for cam in cams:
        location = cam.get("location")
        cam_dict[location] = {
            "html": cam.get("html"),
            "x": cam.get("x"),
            "y": cam.get("y")
        }
    
    return cam_dict

if __name__ == "__main__":
    cameras = fetch_cameras()
    for loc, data in cameras.items():
        print(f"{loc}: {data}")