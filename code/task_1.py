import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = []


images_left = glob.glob('images/task_1/left_*.png')
images_right = glob.glob('images/task_1/right_*.png')
images_left.sort()
images_right.sort()
for i, fname in enumerate(images_left):
    img_l = cv2.imread(images_left[i])
    img_r = cv2.imread(images_right[i])
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)
   
    objpoints.append(objp)
    # If found, add object points, image points (after refining them)
    if ret_l is True:
        rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(rt)

        # Draw and display the corners
        ret_l = cv2.drawChessboardCorners(img_l, (9, 6), rt, ret_l)
        cv2.imshow(images_left[i], img_l)
        cv2.waitKey(500)

    if ret_r is True:
        rt_2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(rt_2)

        # Draw and display the corners
        ret_r = cv2.drawChessboardCorners(img_r, (9, 6), corners_r, ret_r)
        cv2.imshow(images_right[i], img_r)
        cv2.waitKey(500)
    img_shape = gray_l.shape[::-1]
cv2.destroyAllWindows()
ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
ret_2, mtx_2, dist_2, rvecs_2, tvecs_2 = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)


img_l = cv2.imread('images/task_1/left_2.png')
img_r = cv2.imread('images/task_1/right_2.png')
h,  w = img_l.shape[:2]

dst_l = cv2.undistort(img_l, mtx_1, dist_1, None, mtx_1)
dst_r = cv2.undistort(img_r, mtx_2, dist_2, None, mtx_2)

cv2.imwrite('ca_l.jpg',dst_l)
cv2.imwrite('ca_r.jpg',dst_r)

#Saving
np.savetxt('mtx_1.out', mtx_1, delimiter=',')
np.savetxt('mtx_2.out', mtx_2, delimiter=',')
np.savetxt('dist_1.out', dist_1, delimiter=',')
np.savetxt('dist_2.out', dist_2, delimiter=',')

PARA_SAVE_PATH = 'parameters/left_camera_intrinsics'
cv_file = cv2.FileStorage(PARA_SAVE_PATH, cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix_l", mtx_1)
cv_file.write("dist_coeff_l", dist_1)
PARA_SAVE_PATH = 'parameters/right_camera_intrinsics'
cv_file = cv2.FileStorage(PARA_SAVE_PATH, cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix_r", mtx_2)
cv_file.write("dist_coeff_r", dist_2)
