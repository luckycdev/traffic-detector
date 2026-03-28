import requests

URL = "https://traveler.modot.org/timconfig/feed/desktop/StreamingCams2.json"

def fetch_cameras():
    response = requests.get(URL)
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