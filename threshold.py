import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
Performs HSV thresholding, then tries to removes top of frame above the road
Author: Ashwin Vangipuram
"""

# 0, 180, 22, 255, 0, 207
max_value = 255
low_H = 0
low_S = 22
low_V = 0
high_H = 180
high_S = 255
high_V = 207
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
pauseWhenFound = 0
old_gray = None
p0 = None
heur_thresh = 200

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

def trueFalsePause(val):
    global pauseWhenFound
    pauseWhenFound = val
    cv2.setTrackbarPos('pausing', window_capture_name, pauseWhenFound)
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

def drawRects(frame, contours):
    tempPts = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt['cont'])
        boxpts = cv2.boxPoints(rect)
        box = np.int0(boxpts)
        cv2.drawContours(frame,[box],0,(0,0,255),1)
        cv2.drawContours(frame, [cnt['cont']],0,(0,255,0),1)
        cv2.drawContours(frame, [cv2.convexHull(cnt['cont'])],0,(255,0,0),1)
        tempPts.append(rect[0])
        cv2.putText(frame, str(cnt['heur']), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    if len(tempPts) > 1 and allLarger(heur_thresh):
        global paused
        if pauseWhenFound:
            paused = True
        avgPt = getAvgPt(midPt(tempPts[0], tempPts[1]))
        cv2.circle(frame, (avgPt[0], avgPt[1]), 10, (0,0,255), -1)

def midPt(pt1, pt2):
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

def getAvgPt(pt):
    points.append(pt)
    exes = list(map(lambda x: x[0], points))
    whys = list(map(lambda y: y[1], points))

    if len(points) > 50:
        del points[:10]
    return (int(sum(exes) / len(exes)), int(sum(whys) / len(whys)))

def allLarger(thresh):
    for cnt in likelyGate:
        if cnt['heur'] < thresh:
            return False
    return True

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('camera', help='Camera devide number.', default=0, type=str)
args = parser.parse_args()
cap = cv2.VideoCapture(args.camera)

cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)

cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
cv2.createTrackbar('pausing', window_capture_name, pauseWhenFound, 1, trueFalsePause)

#cv2.createTrackbar('low_canny', 'canny', low_canny, 500, lcanny)
paused = False

points = []
while True:
    if not paused:
        ret, frame = cap.read() #reads the frame
    else:
        frame = untampered

    if ret:
        if not paused:
            frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)#resizes frame so that it fits on screen
            blur = cv2.GaussianBlur(frame, (5, 5), 1)
            frame_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            #frame_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            #canny = cv2.Canny(frame_gray, 200, 3, True)
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V)) #low_S ideal = 98 Sets threshold in hsv
        frame_threshold = cv2.bitwise_not(frame_threshold)
        res = cv2.bitwise_and(frame, frame, mask=frame_threshold)
        # print(frame.shape, res.shape, frame_threshold.shape)

        blacked = np.copy(res)
        for row in range(res.shape[0]):
            if np.count_nonzero(res[row]) < 10:
                blacked[:row] = np.zeros((res.shape[1], 3))
                break

        cv2.imshow(window_capture_name, frame)
        cv2.imshow(window_detection_name, frame_threshold)
        cv2.imshow('Result', blacked)

        untampered = np.copy(frame)

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break
    if key == ord('p'):
        paused = not paused

