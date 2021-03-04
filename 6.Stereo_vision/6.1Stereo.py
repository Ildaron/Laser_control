import cv2
import numpy as np



minDisparity = 0;
numDisparities = 32;
blockSize = 1;
disp12MaxDiff = 2;
uniquenessRatio = 15;
speckleWindowSize = 100;
speckleRange = 5;

stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )



camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)
while 1:
 (grabbed, frame1) = camera1.read()
 (grabbed, frame2) = camera2.read()
# ret, frame1 = cap1.read()
# cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',frame1)
# ret, frame2 = cap2.read()
# cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',frame2)



#imgL = image1 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image3.jpg',0)
#$imgR = image2 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image4.jpg',0)
#imgL = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',0)
#imgR  = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',0)






#Setting parameters for StereoSGBM algorithm


#Creating an object of StereoSGBM algorithm

# Calculating disparith using the StereoSGBM algorithm
 disp = stereo.compute(frame1, frame2).astype(np.float32)
 disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)
#
# Displaying the disparity map


 disp = cv2.erode(disp, None, iterations=1)
 disp = cv2.dilate(disp, None, iterations=1)

 cv2.imshow("disparity", disp)
 cv2.waitKey(1)
