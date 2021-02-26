  
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
cap = cv2.VideoCapture(1)
for a in range (0,30,1):
 time.sleep(1)
 ret, frame = cap.read()
 #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/left/'+str(a)+'.jpg',frame)
cap.release()
cv2.destroyAllWindows()


