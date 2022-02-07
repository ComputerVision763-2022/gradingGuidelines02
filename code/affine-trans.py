import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-mat", type=str, required=True)
args = vars(parser.parse_args())

windowName = 'Distorted'
img = cv.imread('../data/distorted.jpg', 1)
cv.namedWindow(windowName)
cv.imshow(windowName, img)

x1, y1, x4, y4, x3, y3, x2, y2 = 0, 0, 60, 600, 660, 660, 600, 60

(r, c, _) = img.shape

if args['mat'] == 'manual':
    cx, cy = -x4/y4, -y2/x2
    shear1 = np.array([[1, 0, 0], [cy, 1, 0], [0, 0, 1]], dtype='float32')
    shear2 = np.array([[1, cx, 0], [0, 1, 0], [0, 0, 1]], dtype='float32')
    shear_tot = shear2@shear1
    img2 = cv.warpAffine(img, shear_tot[:2, :], (c, r))
    cv.imshow('Original using manual matrix', img2)

if args['mat'] == 'api':
    # Now we use the library function

    init_points = np.array([[x1, y1], [x2, y2], [x3, y3]], 'float32')
    final_points = np.array([[0, 0], [600, 0], [600, 600]], 'float32')
    fun_mat = cv.getAffineTransform(init_points, final_points)
    img3 = cv.warpAffine(img, fun_mat, (c, r))
    cv.imshow('Original using library', img3)

keypress = cv.waitKey(0)
cv.destroyAllWindows()
