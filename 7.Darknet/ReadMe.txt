1. Label
https://github.com/tzutalin/labelImg
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py


2. Run real-time
./darknet detector demo cfg/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_best.weights test.mp4 -benchmark

./darknet detector demo cfg/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_best.weights "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=416, height=416, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

2. 1. move to folder
%cd /content/drive/MyDrive/yolov4/darknet
2. 2.Get access for folder
!chmod +x ./darknet
2. 3. Start train
! ./darknet detector train data/obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map

3. Yolo-tiny costum.cng
change class from 80 to 1





