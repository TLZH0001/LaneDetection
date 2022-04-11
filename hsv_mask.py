import cv2
import numpy as np

# define range of road color in HSV
lower_road = np.array([117, 0, 98])
upper_road = np.array([179, 20, 204])

cap = cv2.VideoCapture("./sample_video.MOV")
while not cap.isOpened():
    cap = cv2.VideoCapture("./sample_video.MOV")
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
#     cap.read()
#     cap.read()
#     cap.read()
#     cap.read()
#     cap.read()
#     cap.read()
    if flag:
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only road colors
        mask = cv2.inRange(hsv_img, lower_road, upper_road)

        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        dilated = cv2.dilate(eroded, np.ones((5, 5), np.uint8))
        
        # The frame is ready and already captured
        cv2.imshow('video', dilated)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break

