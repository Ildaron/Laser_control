Runtime -GPU

%cd /content/drive/MyDrive/yolov4/darknet

1. Получить разрешение на работу в папке
!chmod +x ./darknet

2. Запуск train
! ./darknet detector train data/obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29
