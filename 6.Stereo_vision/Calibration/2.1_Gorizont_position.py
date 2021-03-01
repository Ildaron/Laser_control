import cv2

x1=000
y1=100
x2=900
y2=100

cap1 = cv2.VideoCapture(1)
ret, frame = cap1.read()
cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png',frame)


cap2 = cv2.VideoCapture(2)
ret, frame = cap2.read()

cv2.imwrite('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png',frame)


image1 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image1.png')
line_thickness = 2
cv2.line(image1, (x1, y1+50), (x2, y2+50), (10, 155, 10), thickness=line_thickness)
cv2.line(image1, (x1, y1-50), (x2, y2-50), (90, 55, 60), thickness=line_thickness)
cv2.line(image1, (x1, y1), (x2, y2), (60, 70, 100), thickness=line_thickness)
cv2.imshow('image1',image1)
image2 = cv2.imread('C:/Users/rakhmatulin/Desktop/jre_new/2021/4.Stereo_vision/programme/image2.png')
line_thickness = 2
cv2.line(image2, (x1, y1+50), (x2, y2+50), (10, 155, 10), thickness=line_thickness)
cv2.line(image2, (x1, y1-50), (x2, y2-50), (90, 55, 60), thickness=line_thickness)
cv2.line(image2, (x1, y1), (x2, y2), (60, 70, 100), thickness=line_thickness)
cv2.imshow('image2',image2)


cap1.release()
cap2.release()
#cv2.destroyAllWindows() 
