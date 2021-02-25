  
import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image_left.png',frame)
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(2)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image_right.png',frame)
cap.release()
cv2.destroyAllWindows()


image = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image_left.png',0)
image2 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image_right.png',0)

img_wide = 350
r = float(img_wide) / image.shape[1]
dim = (img_wide, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
resized2 = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(resized,resized2)
plt.imshow(disparity,'gray')
plt.show()
