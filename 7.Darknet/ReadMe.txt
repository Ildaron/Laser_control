Label  https://github.com/tzutalin/labelImg

pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py





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
