import os
import cv2 as cv
import numpy as np

img1 = cv.imread('../data/1.jpg')
img2 = cv.imread('../data/2.jpg')
img3 = cv.imread('../data/3.jpg')
img4 = cv.imread('../data/arch.jpg')
blank4 = img4*0 + 1
shape1 = img1.shape[1],img1.shape[0]
shape2 = img2.shape[1],img2.shape[0]
shape3 = img3.shape[1],img3.shape[0]
shape4 = img4.shape[1],img4.shape[0]

pts1 = np.float32([[1511,169],[2955,727],[3004,2046],[1487,2240]])
pts2 = np.float32([[1322,344],[3009,612],[3034,1901],[1307,2006]])
pts3 = np.float32([[929,732],[2815,383],[2850,2230],[899,2085]])
pts4 = np.float32([[0,0],[1200,0],[1200,900],[0,900]])

homo41 = cv.getPerspectiveTransform(pts4,pts1)
homo42 = cv.getPerspectiveTransform(pts4,pts2)
homo43 = cv.getPerspectiveTransform(pts4,pts3)

def set_homo_and_save(inp_img, dest_img, homo, dest_shape, name = ''):
	projected = cv.warpPerspective(inp_img, homo, (dest_shape))
	blank = cv.warpPerspective(blank4, homo, (dest_shape))
	mask = blank != 0
	img_trans = dest_img.copy()
	img_trans[mask] = 0
	img_trans += projected
	cv.imwrite(os.path.join('../results',name), img_trans)

set_homo_and_save(img4, img1, homo41, shape1, name = '1_transformed.jpg')
set_homo_and_save(img4, img2, homo42, shape2, name = '2_transformed.jpg')
set_homo_and_save(img4, img3, homo43, shape3, name = '3_transformed.jpg')