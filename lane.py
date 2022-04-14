import numpy as np 
import cv2
import time


def floodfilled(frame):
    #img = frame.copy()
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (5,5), 0)

    seed = (1079, 920)

    box = (200, 200)
    x = (seed[0] - box[0] - box[1], seed[0])
    y = (seed[1] - box[1], seed[1] + box[1])


    mean = np.mean(img_hsv[x[0]:x[1], y[0]:y[1], :], axis = (0,1))
    thre = [50, 50, 200]
    img_hsv[seed[0], seed[1]] = mean
    
    mask = np.zeros((img_hsv.shape[0] + 2, img_hsv.shape[1] + 2)).astype(np.uint8)
    print(type(mask))
    cv2.floodFill(img_hsv, mask, seedPoint=seed, newVal=(255, 0, 0), \
        loDiff=tuple(thre), upDiff=tuple(thre), flags= cv2.FLOODFILL_FIXED_RANGE)
    # cv2.circle(img_hsv, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
    mask = mask * 255
    return img_hsv, mask[1:-1, 1:-1]


GROUND_MASK_LOWER = np.array([0,0,80],dtype='uint8')
GROUND_MASK_UPPER = np.array([255,50,200],dtype='uint8')
GRASS_MASK_LOWER = np.array([43,50,20],dtype='uint8')
GRASS_MASK_UPPER = np.array([128,255,255],dtype='uint8')

dilkernel = np.ones((5,5), np.uint8)
erokernel = np.ones((6,6), np.uint8)
erokernel[:2,:] = 0
erokernel[-2:,:] = 0
erokernel[:,:2]=0
erokernel[:,-2:]=0

def groundAndGrassMask(image):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    ground = cv2.inRange(hsvImg,GROUND_MASK_LOWER,GROUND_MASK_UPPER)
    grass = cv2.inRange(hsvImg,GRASS_MASK_LOWER,GRASS_MASK_UPPER)
    
    grass = cv2.erode(grass,erokernel,iterations=2)
    ground = cv2.erode(ground,erokernel,iterations=2)
    ground = cv2.dilate(ground,dilkernel,iterations=10)
    grass = cv2.dilate(grass, dilkernel, iterations=10)
    
    combined = cv2.bitwise_and(ground,cv2.bitwise_not(grass))
    combined = cv2.erode(combined,erokernel,iterations=4)
    #combined = cv.dilate(combined,dilkernel,iterations=5)
    
    return ground,grass,combined

def getContour(alpha):
    #https://stackoverflow.com/questions/66753026/opencv-smoother-contour
    contours,hierachy = cv2.findContours(alpha,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key=cv2.contourArea)
    contour_img = np.zeros_like(alpha)
    cv2.drawContours(contour_img, [big_contour], 0, 255, -1)
    # apply dilate to connect the white areas in the alpha channel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    dilate = cv2.morphologyEx(contour_img, cv2.MORPH_DILATE, kernel)
    return dilate, None

def road_and_grass_detection(frame):
    #img = frame.copy()
    ground, grass, comb = groundAndGrassMask(frame)
    dilatedContour, edge = getContour(comb)
    return dilatedContour

def edge_mask(mask, remove_noise = True):
    gx, gy = np.gradient(mask)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255
    temp_edge = temp_edge.astype(np.uint8)

    if not remove_noise:
        return temp_edge
    else:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(temp_edge, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 1000  #num pixels

        img2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        kernel = np.ones((3, 3))
        img2 = cv2.dilate(img2, kernel)
        return img2.astype(np.uint8)

def find_lane(img):
    #img_copy = img.copy()

    t1 = time.time()
    _, floodfilled_mask = floodfilled(img)
    # return floodfilled_mask
    t2 = time.time()

    rg_detection_mask = road_and_grass_detection(img)
    
    t3 = time.time()
    floodfilled_edge = edge_mask(floodfilled_mask, False)
    #final_mask = np.bitwise_and(floodfilled_mask, rg_detection_mask)
    t4 = time.time()
    #final_edge = edge_mask(final_mask)
    rg_edge = edge_mask(rg_detection_mask, False)
    t5 = time.time()

    print("ft: ", t2-t1, "rg: ", t3-t2, "fe: ", t4-t3, 'rge ', t5 - t4)

    large_kernel = np.ones((20, 20))
    kernel = np.ones((4, 4))
    d = cv2.dilate(floodfilled_edge, large_kernel)
    d2 = cv2.dilate(rg_edge, kernel)
    #final_lane = np.bitwise_and(d, d2)
    final_lane = cv2.dilate(d2, kernel)
    

    return final_lane

def process_video(path, frame_per_sec = 4):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    #print(cap.isOpened())
    while cap.isOpened():
        t1 = time.time()
        ret,frame = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        frame_count += frame_per_sec # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        t2 = time.time()


        #_, lane_mask = floodfilled(frame)
        lane_mask = find_lane(frame)
        
        x, y = np.where(lane_mask == 255)
        frame[x, y] = (255, 0, 0)
        cv2.imshow('lane mask on frame', frame)

        t2 = time.time()
        print("fps: ", 1/(t2-t1), t2-t1)
     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


process_video("/home/roar/Desktop/LaneDetection/training_1000.MOV", 20)