import os
import cv2 as cv
import numpy as np


source = cv.imread('../results/sequential.jpg')
target = cv.imread('../results/1_transformed.jpg')

mask = (source > 0)

diff = source-target
mse = np.sum(diff*diff*mask)
rmse = mse/np.sum(target*target)

print("{:.2f}".format(rmse))
