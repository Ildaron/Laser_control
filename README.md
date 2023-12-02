Do not repeat, dangerous, you can only use laser pointer - 1mW!  
# Laser device for neutralizing - mosquitoes, asian hornet, weeds and pests (Open-source) 
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Laser%20for%20%20control%20mosquitoes%20%20and%20any%20insect%20&url=https://github.com/Ildaron/Laser_control&hashtags=laser,mosquitoes,python,opensource)  
[![Hardware demonstrations](https://img.shields.io/badge/Licence-FREE-blue)](https://github.com/Ildaron/Laser_control/blob/master/license.txt)      
[![Hardware demonstrations](https://img.shields.io/badge/Youtube-view-red)](https://youtu.be/2BKtM5cxOik)   


Here I will post information for creating a laser device.

  ![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/gen_view.JPG "general view")  

-  [A warning!!](https://github.com/Ildaron/Laser_control#a-warning)
-  [How It Works](https://github.com/Ildaron/Laser_control/blob/master/README.md#how-it-works)   
-  [General information](https://github.com/Ildaron/Laser_control#general-information)    
-  [Dimensions](https://github.com/Ildaron/Laser_control#dimensions)    
-  [Galvanometer setting](https://github.com/Ildaron/Laser_control#galvanometer-setting)
-  [Determining the coordinates of an object](https://github.com/Ildaron/Laser_control/blob/master/README.md#determining-the-coordinates-of-an-object)
-  [Demonstrations](https://github.com/Ildaron/Laser_control#demonstrations)
-  [We need more FPS](https://github.com/Ildaron/Laser_control#we-need-more-fps)
-  [Security questions](https://github.com/Ildaron/Laser_control/blob/master/README.md#security-questions)
-  [Discussion and Future work](https://github.com/Ildaron/Laser_control#contacts)  
-  [Publication and Citation](https://github.com/Ildaron/Laser_control#publication-and-citation)    
-  [Contacts](https://github.com/Ildaron/Laser_control#contacts)    


### A warning!!
#### Don't use the power laser!
The main limiting factor in the development of this technology is the danger of the laser may damage the eyes. The laser can enter a blood vessel and clog it, it can get into a blind spot where nerves from all over the eye go to the brain, you can burn out a line of "pixels" And then the damaged retina can begin to flake off, and this is the path to complete and irreversible loss of vision. This is dangerous because a person may not notice at the beginning of damage from a laser hit: there are no pain receptors there, the brain completes objects in damaged areas (remapping of dead pixels), and only when the damaged area becomes large enough person starts to notice that some objects not visible.
We can develop additional security systems, such as human detection, audio sensors, etc. But in any case, we are not able to make the installation 100% safe, since even a laser can be reflected and damage the eye of a person who is not in the field of view of the device and at a distant distance. Therefore, this technology should not be used at home. 
My strong recommendation - don't use the power laser! I recommend making a device that will track an object using a safe laser pointer.

#### How It Works
To detect x,y coordinates initially we used Haar cascades in RaspberryPI after that yolov4-tiny in Jetson nano.
For Y coordinates - stereo vision.    
Calculation necessary value for the angle of mirrors.    
RaspberryPI/JetsonNano by SPI sends a command for galvanometer via DAC mcp4922. Electrical scheme ([here](https://github.com/Ildaron/Laser_control/tree/master/2.Jetson_code/2.1_mirror_control)). From mcp4922 bibolar analog signal go to amplifair. Finally, we have -12 and + 12 V for control positions of the mirrors.       

#### General information 
The principle of operation  
![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/scheme...bmp "general view")  
Single board computer to processes the digital signal from the camera and determines positioning to the object, and transmits the digital signal to the analog display - 3, where digital-to-analog converts the signal to the range of 0-5V. Using a board with an operational amplifier, we get a bipolar voltage, from which the boards with the motor driver for the galvanometer are powered - 4, from where the signal goes to galvanometers -7. The galvanometer uses mirrors to change the direction of the laser - 6. The system is powered by the power supply - 5. Cameras 2  determine the distance to the object. The camera detects mosquito and transmits data to the galvanometer, which sets the mirrors in the correct position, and then the laser turns on.  

### Dimensions
![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/dimension.bmp "general view")    
1 - PI cameras, 2 - galvanometer, 3 - Jetson nano, 4 - adjusting the position to the object, 5 - laser device, 6 - power supply, 7 - galvanometer driver boards, 8 - analog conversion boards  

#### Galvanometer setting 
In practice, the maximum deflection angle of the mirrors is set at the factory, but before use, it is necessary to check, for example, according to the documentation, our galvanometer had a step width of 30, but as it turned out we have only 20
![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/galv_angle.bmp "general view")  
 Maximum and minimum positions of galvanometer mirrors:   
 a - lower position - 350 for x mirror;   
 b - upper position - 550 for x mirror;   
 c - lower position - 00 for y mirror;   
 d - upper position - 250 for y mirror;  
 
### Determining the coordinates of an object

#### X,Y - coordinate 
![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/detect_moment.bmp "Example of result for Fast Fourier  transform")

#### Z-coordinate 
We created GUI, source [here](https://github.com/Ildaron/OpenCV-stereovision-tuner-for-windows).
At the expense of computer vision, the position of the object in the X, Y plane is determined - based on which its ROI area is taken. Then we use stereo vision to compile a depth map and for a given ROI with the NumPy library tool - np.average we calculated the average value for the pixels of this area, which will allow us to calculate the distance to the object.  
![alt tag](https://github.com/Ildaron/OpenCV-stereovision-tuner-for-windows/blob/master/pic.2.bmp "Example of result for Fast Fourier  transform")

You can find more detail in the published paper in preprint - [Low-Cost Stereovision System (Disparity Map) For Few Dollars](https://www.preprints.org/manuscript/202104.0282/v1)     

### Determining the angle of galvanometer mirror
#### angle of galvanometer mirror theory
The laser beam obeys all the optical laws of physics, therefore, depending on the design of the galvanometer, the required angle of inclination of the mirror – α, can be calculated through the geometrical formulas. In our case, through the tangent of the angle α, where it is equal to the ratio of the opposing side – X(Y) (position calculated by deep learning) to the adjacent side - Z (calculated by stereo vision).  
 ![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/Z_position.bmp "general view")  

#### angle of galvanometer mirror practice  
![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/z_practice.bmp "general view")  

#### We need more FPS
For single boards, computers are actual problems with FPS. For one object with Jetson was reached the next result for the Yolov4-tiny model.  

Framework                    
with Keras: 4-5 FPS   
with Darknet: 12-15 FPS     
with Darknet Tensor RT: 24-27 FPS      
with Darknet DeepStream: 23-26 FPS    
with tkDNN: 30-35 FPS

You can find more detail in the published paper in arxiv - [Increasing FPS for single board computers and embedded computers in 2021 (Jetson nano and YOVOv4-tiny). Practice and review]( https://arxiv.org/abs/2107.12148)     


### Demonstrations
In this video - a laser (the red point) tries to catch a yellow LED.  It is an adjusting process but in fact, instead, a yellow LED can be a mosquito, and instead, the red laser can be a powerful laser.  
[![Hardware demonstrations](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/demonstration.bmp)](https://www.youtube.com/watch?v=2BKtM5cxOik)    

In the video below, you can see, how the YoloV4tiny detected a cockroach and after that - turn on the laser and send a signal to the galvanometer to set the position of the mirror. When using a more powerful laser, the efficiency is much higher. But the video cannot be shot with a powerful laser, the light is too bright.   
[![Hardware demonstrations](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/coac1.bmp)](https://youtu.be/VwZunUqHbiU)   

### Security questions
An additional device - a security module that will turn off the laser:  
- Use additional cameras to fix people  
- Audio sensors to capture voice and noise
- To mechanically shoot down the laser  
- To use a thermal camera if there is any warm effect, turn it off - this is probably also possible to protect against fires consider not to overheat.
- Teach the system to record the process of laser reflection from any random glass or other mirror surfaces (maybe before turning on the power laser - for checking turn on the simple laser). 

### Discussion and Future work
We can try light up the room to reflect the mosquito - and then use the library functions - OpenCV in range or haar cascades to detect the object. With a bright background, they will be detected without problems. This is for low-power single-board computers - Raspberry, Orange, Banana, etc.
For jetson nano, we can use yolov4-tiny which, using the tkDNN library, is able to give 30-35 FPS

#### Research laser effect
Use a lower the laser power as much as possible.  The laser should burn the wings of mosquitos but should be safe for the eyes. That is to do research on the topic of laser power, laser wavelength, and their efficiency for mosquitos. This is for safety, the lower the power, the better.
#### Remote control
Laser control on a stationary computer. The IP camera installed next to the laser only transmits video to the computer, and the computer already analyzes it on a powerful processor video card and transmits back coordinates for the laser via Wi-Fi. In this case, we can use very powerful computing processors.
#### PCB boards
Make the device completely on our electronic boards. It is a galvanometer for a laser show and changes positions 20,000 times per second, which is why there are such powerful and big drivers for motors. It is useful to make a small PCB board to change the position of the laser only 200 times per second. In finally, so to speak, the pocket version.

#### Publication and Citation 
- Ildar, R. (2021). Machine vision for low-cost remote control of mosquitoes by power laser. Journal of Real-Time Image Processing   
  available [here]( https://www.researchgate.net/publication/349226713_Machine_vision_for_low-cost_remote_control_of_mosquitoes_by_power_laser) https://doi.org/10.1007/s11554-021-01079-x      
- Rakhmatulin I, Andreasen C. (2020). A Concept of a Compact and Inexpensive Device for Controlling Weeds with Laser Beams. Agronomy  
  available [here](https://www.mdpi.com/2073-4395/10/10/1616) https://doi.org/10.3390/agronomy10101616  
- Rakhmatuiln I, Kamilaris A, Andreasen C. Deep Neural Networks to Detect Weeds from Crops in Agricultural Environments in Real-Time: A Review. Remote Sensing. 2021;   13(21):4486. https://doi.org/10.3390/rs13214486
- Rakhmatuiln, I., Lihoreau, M., & Pueyo, J. (2022). Selective neutralisation and deterring of cockroaches with laser automated by machine vision. Oriental Insects, https://doi.org/10.1080/00305316.2022.2121777  

#### Contacts
For any questions write to me by mail - ildarr2016@gmail.com

