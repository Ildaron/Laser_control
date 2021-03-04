import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

#img_1 =  cv2.imread("C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/img0.png",0)
#img_2 = cv2.imread("C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/right/img0.png",0)

img_1 = image1 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png')
img_2 = image2 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png')

#Load camera parameters
#ret = np.load('./camera_params/ret.npy')
#K = np.load('./camera_params/K.npy')
#dist = np.load('./camera_params/dist.npy')
#Specify image paths
#img_path1 = './reconstruct_this/left2.jpg'
#img_path2 = './reconstruct_this/right2.jpg'
#Load pictures
#img_1 = cv2.imread(img_path1)
#img_2 = cv2.imread(img_path2)
#Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size 
h,w = img_2.shape[:2]


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


pathR= "C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/"
pathL = "C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/right/"
obj_pts = []
img_ptsL = []
img_ptsR = []



for i in tqdm(range(0,20)):      
 imgR = cv2.imread(pathR+"img%d.png"%i)
 imgL_gray = cv2.imread(pathL+"img%d.png"%i,0)
 imgR_gray = cv2.imread(pathR+"img%d.png"%i,0)
 imgL = cv2.imread(pathL+"img%d.png"%i)
 outputL = imgL.copy()
 outputR = imgR.copy()
 retR, cornersR =  cv2.findChessboardCorners(outputR,(9,6),None)
 retL, cornersL = cv2.findChessboardCorners(outputL,(9,6),None)
 if retR and retL:
  obj_pts.append(objp)
  cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
  cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
  cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
  cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)
  cv2.imshow('cornersR',outputR)
  cv2.imshow('cornersL',outputL)
		#cv2.waitKey(0)
  img_ptsL.append(cornersL)
  img_ptsR.append(cornersR)


# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None) 
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
KR=mtxR
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))
print ("ok",retR, mtxR, distR, rvecsR, tvecsR )



#Get optimal camera matrix for better undistortion 
#new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))


#Undistort images
img_1_undistorted = cv2.undistort(img_1, mtxL, distL, None, new_mtxL)
img_2_undistorted = cv2.undistort(img_2, mtxR, distR, None, new_mtxR)
#Downsample each image 3 times (because they're too big)
img_1_downsampled = img_1_undistorted
img_2_downsampled = img_2_undistorted






#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error. 
win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 5,
 uniquenessRatio = 5,
 speckleWindowSize = 5,
 speckleRange = 5,
 disp12MaxDiff = 1,
 P1 = 8*3*win_size**2,#8*3*win_size**2,
 P2 =32*3*win_size**2) #32*3*win_size**2)
#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()








