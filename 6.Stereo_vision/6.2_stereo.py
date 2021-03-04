import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
#
camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)

while 1:
 ret, frame1 = camera1.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',frame1)
 ret, frame2 = camera2.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',frame2)
 


 imgL = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',0)
 imgR  = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',0)

 stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
 disparity = stereo.compute(imgL,imgR)

 plt.imshow(disparity,'gray')
 plt.ion()
 plt.pause(.0001)
 plt.show()
 print ("ok")



