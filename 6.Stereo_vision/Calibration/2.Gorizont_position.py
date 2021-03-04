import cv2
import time
x1=000
y1=100
x2=900
y2=100

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

while 1:
 ret, frame1 = cap1.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',frame1)
 ret, frame2 = cap2.read()
 cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',frame2)


 image1 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png')
 line_thickness = 2
 cv2.line(image1, (x1, y1+50), (x2, y2+50), (10, 155, 10), thickness=line_thickness)
 cv2.line(image1, (x1, y1-50), (x2, y2-50), (90, 55, 60), thickness=line_thickness)
 cv2.line(image1, (x1, y1+150), (x2, y2+150), (90, 55, 60), thickness=line_thickness)
 cv2.line(image1, (x1, y1), (x2, y2), (60, 70, 100), thickness=line_thickness)
 
 (h, w) = image1.shape[:2]
 center = (w / 2, h / 2)
 M = cv2.getRotationMatrix2D(center, 180,1)
# image1 = cv2.warpAffine(image1, M, (w, h))


 image2 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png')
 line_thickness = 2
 cv2.line(image2, (x1, y1+50), (x2, y2+50), (10, 155, 10), thickness=line_thickness)
 cv2.line(image2, (x1, y1-50), (x2, y2-50), (90, 55, 60), thickness=line_thickness)
 cv2.line(image2, (x1, y1+150), (x2, y2+150), (90, 55, 60), thickness=line_thickness)
 cv2.line(image2, (x1, y1), (x2, y2), (60, 70, 100), thickness=line_thickness)

 (h, w) = image2.shape[:2]
 center = (w / 2, h / 2)
 M = cv2.getRotationMatrix2D(center, 180,1)
 #image2 = cv2.warpAffine(image2, M, (w, h))
 
 cv2.imshow('image1',image1)
 cv2.imshow('image2',image2)
 cv2.waitKey(1)

 #time.sleep(10)
cv2.destroyAllWindows() 
cap1.release()
cap2.release()
#cv2.destroyAllWindows() 
