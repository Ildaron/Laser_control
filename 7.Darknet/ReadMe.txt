Runtime -GPU

1. Двигаемся в папку
%cd /content/drive/MyDrive/yolov4/darknet

2. Получить разрешение на работу в папке
!chmod +x ./darknet

3. Запуск train
! ./darknet detector train data/obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map
