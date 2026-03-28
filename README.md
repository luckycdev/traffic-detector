# Eagle Eye Traffic Detector

## Installation

You need Docker, docker-compose and git setup on your machine (Refer to [Docker documentation](https://docs.docker.com/) if you need any help with this)

If you wish to change any settings such as server ip, server port, default camera, or cameras json, edit `config.py`

After that, run the following commands:

```
git clone https://github.com/luckycdev/traffic-detector
cd traffic-detector
docker compose up -d --build
```

Once done, you can access it at http://localhost:5050 (may be different if you edited `config.py`)