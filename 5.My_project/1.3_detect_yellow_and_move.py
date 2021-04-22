import cv2
import spidev
import time
import numpy

colorLower = (86, 255, 255)
colorUpper = (239, 255, 255)

#SPI
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=1000000
first = '0011' # upper 
second = '1011' # lower
print ("ok2")

def dac (channel, voltage):
 voltage1=bin((voltage))
 voltage2=voltage1[2:]
 s = channel + (voltage2).zfill(12) 
 part_1, part_2=s[:8],s[8:] 
 byte1=int (part_1,2)
 byte2=int (part_2,2)
 to_send =  [byte1, byte2]             
 spi.xfer2(to_send)

camera = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=400, height=400, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

print ("ok5")
axi_z=1330
while 1:  
 (grabbed, frame) = camera.read()
 cv2.imshow("Frame", frame)
 key = cv2.waitKey(1) & 0xFF

 frame = cv2.erode(frame, None, iterations=6)
 frame = cv2.dilate(frame, None, iterations=2)
  #frame = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_RGB2GRAY)
  #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 frames=frame
  
# dac (second, axi_z)
# dac (first, 2600)
 (grabbed, frame) = camera.read()
 cv2.imshow("Frame", frame)
 
 frame = cv2.inRange(frame, colorLower, colorUpper)
 cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
 radius = 0




 for c in cnts:
  ((x, y), radius) = cv2.minEnclosingCircle(cnts[0])
   
 # my = 10.2665*x+246#.8017
 # mx = 6.7691*y-68#.7463 

  my = int (10*x+246)#.8017 
  mx = int (6*y-68)#.7463 

   
  dac (first,  my)
  dac (second, mx)

  cv2.imshow("Frame2", frame)
  key = cv2.waitKey(1) & 0xFF
#  print (radius)

 if radius > 0:
  cv2.circle(frames, (int(x), int(y)), int(2*radius),(0, 255, 255), 2)
  print ("x,y",x,y)
  cv2.putText(frames, 'object was found', (int(x) , int(y)), cv2.FONT_ITALIC, 0.5, 255)
 
  cv2.imshow("Frames", frames)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
   break

cap.release()
cv2.destroyAllWindows()
