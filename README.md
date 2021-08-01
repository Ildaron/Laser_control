# Laser device for neutralizing - mosquitoes, weeds and pests (in progress)
Here I will post information for creating a laser device.

-  [A warning!!](https://github.com/Ildaron/Laser_control#a-warning)    
[General information ](https://github.com/Ildaron/Laser_control#general-information)    
[Publication and Citation](https://github.com/Ildaron/Laser_control#publication-and-citation)    
[Contacts](https://github.com/Ildaron/Laser_control#contacts)    


### A warning!!


#### General information 
The principle of operation  
![alt tag](https://github.com/Ildaron/Laser_control/blob/master/Supplementary%20files/scheme.bmp "general view")  
Single board computer Raspberry PI 3 B +, processes the digital signal from the video and determines
positioning to the object, and transmits the digital signal to the analog display - 3, where digital-to-analog
the converter converts the signal to the range of 0-5V. Next, using a board with an operational amplifier, we
get a bipolar voltage - plus and minus 5 V, from which the boards with the motor driver for the galvanometer are powered -
4, from where the signal goes to galvanometers -7. The galvanometer uses mirrors to change
the direction of the laser - 6. The system is powered by the power supply - 5. Cameras 2 
determine the distance to the object.
The mosquito box is located 300 mm from the laser system. The camera detects
mosquito and transmits data to the galvanometer, which sets the mirrors in the correct position,
and then the laser turns on.  

### Demonstrations

#### Publication and Citation 
- Ildar, R. (2021). Machine vision for low-cost remote control of mosquitoes by power laser. Journal of Real-Time Image Processing   
  availabe [here]( https://www.researchgate.net/publication/349226713_Machine_vision_for_low-cost_remote_control_of_mosquitoes_by_power_laser)    
- Rakhmatulin I, Andreasen C. (2020). A Concept of a Compact and Inexpensive Device for Controlling Weeds with Laser Beams. Agronomy  
  availabe [here](https://www.mdpi.com/2073-4395/10/10/1616)  

#### Contacts
For any questions write to me by mail - ildar.o2010@yandex.ru  

