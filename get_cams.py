import requests
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

CAMERAS_JSON_URL = os.getenv(
    "CAMERAS_JSON_URL",
    "https://traveler.modot.org/timconfig/feed/desktop/StreamingCams2.json",
)

# Returns a dictionary of cameras with their name, location, and stream url
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

# Ran if running this file directly, prints cameras
if __name__ == "__main__":
    cameras = fetch_cameras()
    for loc, data in cameras.items():
        print(f"{loc}: {data}")