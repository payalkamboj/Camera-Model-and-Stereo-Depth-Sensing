import numpy as np
import cv2
import glob
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image 
import math

#loading camera parameters
R = np.load('parameters/R.npy')

T = np.load('parameters/T.npy')

E = np.load('parameters/E.npy')

F = np.load('parameters/F.npy')

M1 = np.load('parameters/M1.npy')

M2 = np.load('parameters/M2.npy')
d1 = np.load('parameters/d1.npy')
d2 = np.load('parameters/d2.npy')
R1 = np.load('parameters/R1.npy')
R2 = np.load('parameters/R2.npy')
P1 = np.load('parameters/P1.npy')
P2 = np.load('parameters/P2.npy')

mtx_1 = np.load('parameters/mtx_left.npy')

dist_1 = np.load('parameters/dist_left.npy')

mtx_2 = np.load('parameters/mtx_right.npy')

dist_2 = np.load('parameters/dist_right.npy')
Q = np.load('parameters/Q.npy')

img_l = cv2.imread('images/task_3_and_4/left_4.png',0)
img_r = cv2.imread('images/task_3_and_4/right_4.png',0)

img_shape = img_r.shape[::-1]
xl, yl = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_shape, cv2.CV_32FC1)
xr, yr = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_32FC1)

#remapping
img1_rect = cv2.remap(img_l, xl, yl, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img_r, xr, yr, cv2.INTER_LINEAR)
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

# disparity = stereo.compute(img1_rect,img2_rect)
# disparity = cv2.normalize(disparity, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
# plt.imshow(disparity,'gray')
# plt.show()


# points = cv2.reprojectImageTo3D(disparity, Q)
# #depth = Baseline * focal-lens / disparity

# print (points.shape)

################################################################
wsize=31
max_disp = 128
sigma = 1.5
lmbda = 8000.0
left_matcher = cv2.StereoBM_create(max_disp, wsize)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
left_disp = left_matcher.compute(img1_rect, img2_rect)
right_disp = right_matcher.compute(img2_rect,img1_rect)

# Now create DisparityWLSFilter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
filtered_disp = wls_filter.filter(left_disp, img1_rect, disparity_map_right=right_disp)
plt.imshow(filtered_disp,'gray')
plt.show()

cv2.imwrite('task4_result.jpg',filtered_disp)
img_l = Image.open('task4_result.jpg')
left = 80
top = 30
right = 500
bottom = 280
  
im1 = img_l.crop((left, top, right, bottom))
im1.save("task4_cropped.jpg")
