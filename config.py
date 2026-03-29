import os

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = os.getenv("SERVER_PORT", "5050")
DEFAULT_STREAM_SOURCE = os.getenv("DEFAULT_STREAM_SOURCE", "https://traveler.modot.org/tisvc/api/Tms/CameraStream/K070EBIPC-14-LQ")
DEFAULT_CAMERA_NAME = os.getenv("DEFAULT_CAMERA_NAME", "I-70 EB At 18th St Expressway")
