  
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

for a in range (0,30,1):
 time.sleep(1)
 cap = cv2.VideoCapture(2)         
 ret, frame = cap.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/right/'+'img'+str(a)+'.png',frame)
 cap = cv2.VideoCapture(1)         
 ret, frame = cap.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/'+'img'+str(a)+'.png',frame)



cv2.destroyAllWindows()





