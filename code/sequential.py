import os
import cv2 as cv
import numpy as np
import time

img1 = cv.imread('../data/1.jpg')
# gamma transfers arch to 3
img2 = cv.imread('../results/3_transformed.jpg')
img4 = cv.imread('../data/arch.jpg')
blank4 = img4*0 + 1
shape1 = img1.shape[1],img1.shape[0]
shape2 = img2.shape[1],img2.shape[0]

# gamma gets these (points and homography) from alpha and beta
pts1 = np.float32([[1511,169],[2955,727],[3004,2046],[1487,2240]])
pts2 = np.float32([[1322,344],[3009,612],[3034,1901],[1307,2006]])
pts3 = np.float32([[929,732],[2815,383],[2850,2230],[899,2085]])
pts4 = np.float32([[0,0],[1200,0],[1200,900],[0,900]])
homo32 = cv.getPerspectiveTransform(pts3,pts2)
homo21 = cv.getPerspectiveTransform(pts2,pts1)

def set_homo_and_ret(inp_img, dest_img, homo, dest_shape):
	projected = cv.warpPerspective(inp_img, homo, (dest_shape))
	blank = cv.warpPerspective(blank4, homo, (dest_shape))
	mask = blank != 0
	img_trans = dest_img.copy()
	img_trans[mask] = 0
	img_trans += projected
	return img_trans

p1 = cv.warpPerspective(img2, homo32, shape1)
p2 = cv.warpPerspective(p1, homo21, shape1)
cv.imwrite('../results/sequential.jpg', p2)
