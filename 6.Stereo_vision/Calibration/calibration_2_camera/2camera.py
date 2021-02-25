import cv2
import numpy as np
from tqdm import tqdm




# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(1,2)):
	imgL = cv2.imread("C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/image1.png",0)
	imgR = cv2.imread("C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/right/image2.png",0)
	imgL_gray = cv2.imread("C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/image1.png",0)
	imgR_gray = cv2.imread("C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/right/image2.png",0)

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
		#cv2.imshow('cornersR',outputR)
		#cv2.imshow('cornersL',outputL)
		#cv2.waitKey(0)

		img_ptsL.append(cornersL)
		img_ptsR.append(cornersR)


# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))
print ("ok",retR, mtxR, distR, rvecsR, tvecsR )



# Reading the left and right images.



# Setting parameters for StereoSGBM algorithm
minDisparity = 0;
numDisparities = 64;
blockSize = 8;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )

# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disparity",disp)
cv2.waitKey(0)



