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

mtx_1 = np.load('parameters/mtx_left.npy')

dist_1 = np.load('parameters/dist_left.npy')

mtx_2 = np.load('parameters/mtx_right.npy')

dist_2 = np.load('parameters/dist_right.npy')

img_l = cv2.imread('images/task_3_and_4/left_0.png')
img_r = cv2.imread('images/task_3_and_4/right_0.png')
gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    

h,  w = img_l.shape[:2]

img_l = cv2.undistort(gray_l, mtx_1, dist_1, None, mtx_1)
img_r = cv2.undistort(gray_r, mtx_2, dist_2, None, mtx_2)

orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img_l,None)
# kp2, des2 = orb.detectAndCompute(img_r,None)
kp1 = orb.detect(img_l,None)
print(len(kp1))
kp2 = orb.detect(img_r,None)
#print(len(kp2))
img2 = cv2.drawKeypoints(img_l,kp1, img_l, color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()

def extract_good_feature(radius, keypoint):
    feature_list_kp1 = keypoint
    for feature in keypoint:
        x = feature.pt[0]
        #print(x)
        y = feature.pt[1]
        for feature2 in keypoint:
            x2 = feature2.pt[0]
            y2 = feature2.pt[1]
            distance = math.sqrt( (x - x2)**2 + (y - y2)**2 )
            #print(distance)
            #the one with higher response should be kept
            if((distance < radius) and (feature.response < feature2.response)):
                if(feature in feature_list_kp1):
                    feature_list_kp1.remove(feature)
    #s = list(set(feature_list_kp1)) 
    return feature_list_kp1


#print(s) 
#print(len(feature_list_kp1))
kp1_good_features = extract_good_feature(30,kp1)
print(len(kp1_good_features))
img2 = cv2.drawKeypoints(img_l,kp1_good_features, None, color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()
kp2_good_features = extract_good_feature(30,kp2)
img2 = cv2.drawKeypoints(img_r,kp2_good_features, None, color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()
#print(feature_list_kp1)
############BFMatcher#############
kp_l, des_l = orb.compute(img_l, kp1_good_features)
kp_r, des_r = orb.compute(img_r, kp2_good_features)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

#######3my code##################3
matches = bf.match(des_l,des_r)
matches = sorted(matches, key = lambda x:x.distance)
matches = matches[:40]
#print(matches)
#result = cv2.drawMatches(img_l,kp1,img_r,kp2,matches[:min(n, len(matches))],None,matchColor=[0,255,0], flags=2)
#matched_image = cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, flags=2)

result = cv2.drawMatches(img_l,kp1,img_r,kp2,matches, None, flags=2)

plt.imshow(result, interpolation = 'bicubic')
plt.axis('off')
plt.show()
cv2.imwrite('result_task3.jpg',result)

# #step6
kp_l_pts = cv2.KeyPoint_convert(kp_l)
kp_r_pts = cv2.KeyPoint_convert(kp_r[:len(kp_l)])
#array2 = ( kp2[:len(kp1)] ),
#pts_l = cv2.undistortPoints(kp_l_pts, M1, d1)
pts_r = cv2.undistortPoints(np.concatenate(kp_r_pts).reshape(-1,1, 2), M2, np.array(d2), P = M2)
pts_l = cv2.undistortPoints(np.concatenate(kp_l_pts).reshape(-1,1, 2), M1, np.array(d1), P = M1)
#print(kp_l_pts)
t0 = np.array([[0], [0], [0]])
I = np.identity(3)
Proj_mat_l = np.dot(M1, np.concatenate((I, t0), axis=1))
Proj_mat_r = np.dot(M2, np.concatenate((R, T), axis=1))
X = cv2.triangulatePoints(Proj_mat_l, Proj_mat_r, pts_l, pts_r)
X = X.T
print(X.shape)
# # convert from homogeneous coordinates to 3D
triang_pts3D = X[:, :3]/np.repeat(X[:, 3], 3).reshape(-1, 3)
print(triang_pts3D.shape)

#Graph
# x = triang_pts3D[0]
# y = triang_pts3D[1]
# z = triang_pts3D[2]
# fig = plt.figure()

# ax = fig.add_subplot(1, 1, 1, projection="3d")

# ax.scatter(x, y, z)

# plt.show()
###########################################################
x = triang_pts3D[:,0]
y = triang_pts3D[:,1]
z = triang_pts3D[:,2]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x.T, y.T, z.T)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-250, 250])
ax.set_ylim([-250, 250])
ax.set_zlim([-100, 300])
 
plt.show()
