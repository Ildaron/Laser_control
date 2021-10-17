also - yolov4-tiny-custom_best.weights
start real-time 
./darknet detector demo cfg/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_best.weights "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=416, height=416, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2  ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
