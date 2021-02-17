import cv2
import spidev
import time
import numpy

colorLower = (255, 255, 255)
colorUpper = (255, 255, 255)

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

print ("ok4")
dac (first, 1740)
#dac (second, 440)

x=145
radius = 8

print ("ok5")
axi_z=[640]
while 1:
 for axi in axi_z:  
  time.sleep(0.5)
  (grabbed, frame) = camera.read()
  #frame = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_RGB2GRAY)
 #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  frames=frame
  dac (second, axi)
  dac (first, 1980)


  frame = cv2.inRange(frame, colorLower, colorUpper)
  cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]



  for c in cnts:  
   ((x, y), radius) = cv2.minEnclosingCircle(c)
   print (radius)
#  cv2.imshow("Frame", frame)
#  key = cv2.waitKey(1) & 0xFF
   if radius > 0:
    cv2.circle(frames, (int(x), int(y)), int(2*radius),(0, 255, 255), 2)
    cv2.putText(frames, 'object was found', (int(x) , int(y)), cv2.FONT_ITALIC, 0.5, 255)
    
    cv2.imshow("Frames", frames)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
     break



cap.release()
cv2.destroyAllWindows()
