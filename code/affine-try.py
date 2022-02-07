import cv2 as cv
import numpy as np


windowName = 'Distorted'
img = cv.imread('../data/chessboard2.jpg', 1)
cv.namedWindow(windowName)
cv.imshow(windowName, img)

# chose 3 corners
x1 = 135
y1 = 219
x4 = 764
y4 = 576
x2 = 656
y2 = 220

(r, c, _) = img.shape

init_points = np.array([[x1, y1], [x2, y2], [x4, y4]], 'float32')
final_points = np.array([[0, 0], [600, 0], [600, 600]], 'float32')
fun_mat = cv.getAffineTransform(init_points, final_points)

img3 = cv.warpAffine(img, fun_mat, (c, r))

cv.imshow('Original using library', img3)

keypress = cv.waitKey(0)
cv.destroyAllWindows()
