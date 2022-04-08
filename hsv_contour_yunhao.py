"""
By Yuhao Cao
See canny_yunhao.ipynb for experimental code
Usgae: place test MOV under this folder and name it test.MOV then run:
python hsv_contour_yunhao.py
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


CANNY_THRESHOLD_1 = 0
CANNY_THRESHOLD_2 = 200
FILENAME = 'test.mov'
GROUND_MASK_LOWER = np.array([0,0,80],dtype='uint8')
GROUND_MASK_UPPER = np.array([255,50,200],dtype='uint8')
GRASS_MASK_LOWER = np.array([43,50,20],dtype='uint8')
GRASS_MASK_UPPER = np.array([128,255,255],dtype='uint8')
NUM_FRAMES_PER_TIME = 20

#https://stackoverflow.com/questions/41138000/fit-quadrilateral-tetragon-to-a-blob
#https://medium.com/analytics-vidhya/building-a-lane-detection-system-f7a727c6694
dilkernel = np.ones((5,5), np.uint8)
erokernel = np.ones((6,6), np.uint8)
erokernel[:2,:] = 0
erokernel[-2:,:] = 0
erokernel[:,:2]=0
erokernel[:,-2:]=0
def groundAndGrassMask(image):
    hsvImg = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    ground = cv.inRange(hsvImg,GROUND_MASK_LOWER,GROUND_MASK_UPPER)
    grass = cv.inRange(hsvImg,GRASS_MASK_LOWER,GRASS_MASK_UPPER)
    
    grass = cv.erode(grass,erokernel,iterations=2)
    ground = cv.erode(ground,erokernel,iterations=2)
    ground = cv.dilate(ground,dilkernel,iterations=10)
    grass = cv.dilate(grass, dilkernel, iterations=10)
    
    combined = cv.bitwise_and(ground,cv.bitwise_not(grass))
    combined = cv.erode(combined,erokernel,iterations=4)
    #combined = cv.dilate(combined,dilkernel,iterations=5)
    
    return ground,grass,combined

def getContour(alpha):
    #https://stackoverflow.com/questions/66753026/opencv-smoother-contour
    contours,hierachy = cv.findContours(alpha,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key=cv.contourArea)
    contour_img = np.zeros_like(alpha)
    cv.drawContours(contour_img, [big_contour], 0, 255, -1)
    # apply dilate to connect the white areas in the alpha channel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40,40))
    dilate = cv.morphologyEx(contour_img, cv.MORPH_DILATE, kernel)
    cannyEdge = cv.Canny(dilate,CANNY_THRESHOLD_1,CANNY_THRESHOLD_2)
    return dilate,cannyEdge

def lanesDetection(frame):
    ground, grass, comb = groundAndGrassMask(frame)
    dilatedContour, edge = getContour(comb)
    return dilatedContour

def progress(capture):
    nextFrameNo = capture.get(cv.CAP_PROP_POS_FRAMES)
    totalFrames = capture.get(cv.CAP_PROP_FRAME_COUNT)
    return nextFrameNo / totalFrames


def videoLanes():
    # cap = cv.VideoCapture('P6010001.MOV')
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     frame = lanesDetection(frame)
    #     cv.imshow('Lanes Detection', frame)
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    cap = cv.VideoCapture(FILENAME)
    cv.namedWindow("ld", cv.WINDOW_KEEPRATIO)
    cv.namedWindow('raw',cv.WINDOW_KEEPRATIO)
    frameCount = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frameCount += NUM_FRAMES_PER_TIME # i.e. at 30 fps, this advances one second
            cap.set(cv.CAP_PROP_POS_FRAMES, frameCount)
        else:
            cap.release()
            break
        
        print(progress(cap))

        dilated = lanesDetection(frame)
        cv.imshow('raw',frame)
        cv.imshow('ld', dilated)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print(cap.isOpened())



if __name__ == "__main__":
    videoLanes()