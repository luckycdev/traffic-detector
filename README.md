# Eagle Eye Traffic Detector

Eagle Eye Traffic Detector is a traffic detector that uses YOLO computer vision model to detect vehicles from live camera feeds. Then it calculates the amount of traffic on the road based on multiple variables.

![Video Feed](https://i.luckyc.dev/eagleeye1.png)
![Stats](https://i.luckyc.dev/eagleeye2.png)
![Map](https://i.luckyc.dev/eagleeye3.png)

## Installation

You need Docker, docker-compose and git setup on your machine (Refer to [Docker documentation](https://docs.docker.com/) if you need any help with this)

If you wish to change any settings such as server ip, server port, default camera, or cameras json, edit `.env`

After that, run the following commands:

```
git clone https://github.com/luckycdev/traffic-detector
cd traffic-detector
docker compose up -d --build
```

Once done, you can access it at http://localhost:5050 (may be different if you edited `.env`)
