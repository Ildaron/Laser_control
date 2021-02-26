import cv2

x1=000
y1=100
x2=900
y2=100

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',frame)


cap = cv2.VideoCapture(2)
ret, frame = cap.read()

cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',frame)





image1 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',0)
line_thickness = 2
cv2.line(image1, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
cv2.imshow('image1',image1)
image2 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',0)
line_thickness = 2
cv2.line(image2, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
cv2.imshow('image2',image2)


