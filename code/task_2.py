import numpy as np
import cv2
import glob
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image 


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = []
import numpy as np
import cv2
import glob

#loading the parameters
mtx_1 = np.load('parameters/mtx_left.npy')

dist_1 = np.load('parameters/dist_left.npy')
#print(dist_1)
mtx_2 = np.load('parameters/mtx_right.npy')

dist_2 = np.load('parameters/dist_right.npy')
# print(mtx_left)
# print(dist_left)
# print(mtx_right)
# print(dist_right)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = []


images_left = glob.glob('images/task_2/left_0.png')
#print("images_left",images_left)
images_right = glob.glob('images/task_2/right_0.png')

i = 0
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
        # print("rt", rt.shape)
        # print("rt.datatype", rt.dtype)
        #imgpoints_l.append(rt.reshape(-1, 2))
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


#stereo calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l,
            imgpoints_r, mtx_1, dist_1, mtx_2,
            dist_2, img_shape,
            criteria=stereocalib_criteria, flags=flags)
#print(M1,d1)
# print("R:",R)
# print("T:",T)
# dst_l = cv2.undistort(img_l, M1, d1, None, M1)
# plt.imshow(dst_l)
# plt.show()

# new_image_points = np.concatenate(imgpoints_l).reshape(-1,1, 2)
#print(new_image_points.shape)
# Converting mm to cm for visualization
T = 25.4*T/10

#undistort_points
pts_l = cv2.undistortPoints(np.concatenate(imgpoints_l).reshape(-1,1, 2), M1, np.array(d1), P = M1)
pts_r = cv2.undistortPoints(np.concatenate(imgpoints_r).reshape(-1,1, 2), M2, np.array(d2), P = M2)

#triangulation
t0 = np.array([[0], [0], [0]])
I = np.identity(3)
Proj_mat_l = np.dot(M1, np.concatenate((I, t0), axis=1))
Proj_mat_r = np.dot(M2, np.concatenate((R, T), axis=1))
#print(Proj_mat_l.shape)
#print(Proj_mat_r.shape)
X = cv2.triangulatePoints(Proj_mat_l, Proj_mat_r, pts_l, pts_r)
print(X.shape)
X = X.T
# convert from homogeneous coordinates to 3D
triang_pts3D = X[:, :3]/np.repeat(X[:, 3], 3).reshape(-1, 3)
print(triang_pts3D.shape)
x = triang_pts3D[:,0]
y = triang_pts3D[:,1]
z = triang_pts3D[:,2]
# fig = plt.figure()

#####################################################
#### Plot Camera Pos ###########

def plotCameraPos():
 
   f = 2
   tan_x = 1
   tan_y = 1
 
   R_prime = R
   t_prime = T
 
   cam_center_local = np.asarray([
       [0, 0, 0],      [tan_x, tan_y, 1],
       [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
       [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
       [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
       [tan_x, tan_y, 1],      [0, 0, 0]
       ]).T
 
   cam_center_local *= f
   cam_center = np.matmul(R_prime, cam_center_local) + t_prime
 
   fig = plt.figure()

   ax = fig.add_subplot(1, 1, 1, projection='3d')
 
   ax.plot(cam_center_local[0, :], cam_center_local[1, :], cam_center_local[2, :],
                   color='r', linewidth=2)
   ax.plot(cam_center[0, :], cam_center[1, :], cam_center[2, :],
                   color='g', linewidth=2)

   ax.scatter(x.T, y.T, z.T)

   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')

   ax.set_xlim([-40, 40])
   ax.set_ylim([-40, 40])
   ax.set_zlim([-40, 40])
 
   plt.show()


plotCameraPos()

#stereo rectify STEP 5
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, d1, M2, d2, img_shape, R, T, alpha = -1, newImageSize = None)
#print(validPixROI1)

#CHeking rectification results Step 6
xl, yl = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_shape, cv2.CV_32FC1)
xr, yr = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_shape, cv2.CV_32FC1)

#remapping
img1_rect = cv2.remap(img_l, xl, yl, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img_r, xr, yr, cv2.INTER_LINEAR)
#print(xl, yl)
cv2.imwrite('task2_yol.jpg',img1_rect)
cv2.imwrite('task2_yor.jpg',img2_rect)
plt.imshow(img1_rect)
plt.show()
#cropping the rectified image
#images_right = glob.glob('task2_l.jpg')
img_l = Image.open('task2_l.jpg')
img_r = Image.open('task2_r.jpg')
left = 80
top = 30
right = 500
bottom = 330
  
im1 = img_l.crop((left, top, right, bottom))
im2 = img_r.crop((left, top, right, bottom))
im1.save("task_2_rectified_left.jpg")
im2.save("task_2_rectified_right.jpg")

#Saving the parameters
PARA_SAVE_PATH = 'parameters/stereo_calibration'
cv_file = cv2.FileStorage(PARA_SAVE_PATH, cv2.FILE_STORAGE_WRITE)
cv_file.write("R", R)
cv_file.write("T", T)
cv_file.write("E", E)
cv_file.write("F", F)
cv_file.write("M1", M1)
cv_file.write("M2", M2)
cv_file.write("d1", d1)
cv_file.write("d2", d2)

#saving another file of Paramters
PARA_SAVE_PATH = 'parameters/stereo_rectification'
cv_file = cv2.FileStorage(PARA_SAVE_PATH, cv2.FILE_STORAGE_WRITE)
cv_file.write("R1", R1)
cv_file.write("R2", R2)
cv_file.write("P1", P1)
cv_file.write("P2", P2)
cv_file.write("Q", Q)

np.save('parameters/R', R)

np.save('parameters/T', T)

np.save('parameters/E', E)

np.save('parameters/F', F)
np.save('parameters/M1', M1)
np.save('parameters/M2', M2)
np.save('parameters/d1', d1)
np.save('parameters/d2', d2)
np.save('parameters/R1', R1)
np.save('parameters/R2', R2)
np.save('parameters/P1', P1)
np.save('parameters/P2', P2)
np.save('parameters/Q', Q)