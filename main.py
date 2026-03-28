## Live feed with webcam code test
# https://5fca316e7c40f.streamlock.net/live-secure/customInstance/M070WBIPC-11-LQ.stream/playlist.m3u8?wowzatokenendtime=1774660003&wowzatokenstarttime=1774656403&wowzatokenhash=BUeEDuAf-7eYF42gqKZxg2_9_vB8pSX2BY9HvMrXULM=
#import cv2
# Install ultralytics and relevent import statements
!pip install ultralytics
!pip install opencv-python

import cv2
from ultralytics import YOLO
import numpy as np


from IPython.display import display, clear_output
from PIL import Image

# Open the default camera
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("https://5fca316e7c40f.streamlock.net/live-secure/customInstance/M070WBIPC-11-LQ.stream/playlist.m3u8?wowzatokenendtime=1774660003&wowzatokenstarttime=1774656403&wowzatokenhash=BUeEDuAf-7eYF42gqKZxg2_9_vB8pSX2BY9HvMrXULM=")

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (frame_width, frame_height))

# Creating Yolo model object (currently Yolo26n)
model = YOLO("yolo26n.pt")



while True:
    ret, frame = cam.read()

    results = model.predict(frame, classes=[2,3,5,7])

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    #cv2.imshow("Frame", frame)

    clear_output(wait=True)
    display(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()