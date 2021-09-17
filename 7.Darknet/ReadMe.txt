Label  https://github.com/tzutalin/labelImg

pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py


./darknet detector demo cfg/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_best.weights "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=416, height=416, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"


2. Runtime -GPU

2. 1. Двигаемся в папку
%cd /content/drive/MyDrive/yolov4/darknet

2. 2. Получить разрешение на работу в папке
!chmod +x ./darknet

2. 3. Запуск train
! ./darknet detector train data/obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map

3. Yolo-tiny costum.cng
здемы мы в двух метсах изменнили класс с 80 до 1
и изменели количетсво слоев у послденего выхода, сделали 

4. start video


./darknet detector demo cfg/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_best.weights test.mp4 -benchmark



