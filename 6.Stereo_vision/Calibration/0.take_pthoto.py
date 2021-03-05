  
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

for a in range (0,30,1):
 print (a)
 time.sleep(1)
 cap1 = cv2.VideoCapture(1)     
 cap2 = cv2.VideoCapture(2)         
 ret, frame1 = cap1.read()
 ret, frame2 = cap2.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/right/'+'img'+str(a)+'.png',frame1)
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/'+'img'+str(a)+'.png',frame2)
cv2.destroyAllWindows()


