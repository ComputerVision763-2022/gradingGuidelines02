import os
import cv2 as cv
import numpy as np
import time

img1 = cv.imread('../data/1.jpg')
img2 = cv.imread('../data/2.jpg')
img3 = cv.imread('../data/3.jpg')
img4 = cv.imread('../data/arch.jpg')
blank4 = np.ones((360,640))
shape1 = img1.shape[1],img1.shape[0]
shape2 = img2.shape[1],img2.shape[0]
shape3 = img3.shape[1],img3.shape[0]
shape4 = img4.shape[1],img4.shape[0]

pts1 = np.float32([[1511,169],[2955,727],[3004,2046],[1487,2240]])
pts2 = np.float32([[1322,344],[3009,612],[3034,1901],[1307,2006]])
pts3 = np.float32([[929,732],[2815,383],[2850,2230],[899,2085]])
pts4 = np.float32([[0,0],[640,0],[640,360],[0,360]])

homo41 = cv.getPerspectiveTransform(pts4,pts1)
homo42 = cv.getPerspectiveTransform(pts4,pts2)
homo43 = cv.getPerspectiveTransform(pts4,pts3)

def set_homo_and_ret(inp_img, dest_img, homo, dest_shape):
	projected = cv.warpPerspective(inp_img, homo, (dest_shape))
	blank = cv.warpPerspective(blank4, homo, (dest_shape))
	mask = blank != 0
	img_trans = dest_img.copy()
	img_trans[mask] = 0
	img_trans += projected
	return img_trans

cap = cv.VideoCapture('../data/test.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # z = time.time()
    frame = set_homo_and_ret(frame, img1, homo41, shape1)
    scaled = cv.resize(frame,None,fx=.3,fy=.3,interpolation=cv.INTER_LINEAR)
    # print(time.time()-z)
    cv.imshow('frame',scaled)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()