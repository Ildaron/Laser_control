import spidev
print (1)
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=1000000
first = '0011'
second = '1011'

Voltage_A=4.5
Voltage_B=3

def dac (channel, voltage):
#voltage = 900 #
 voltage1=bin((voltage))
 voltage2=voltage1[2:]

 s = channel + (voltage2).zfill(12) 
 part_1, part_2=s[:8],s[8:] 
 byte1=int (part_1,2)
 byte2=int (part_2,2)
 to_send =  [byte1, byte2]             
 spi.xfer2(to_send)
         

while 1:
 dac (first, int (Voltage_A*819))
 dac (second, int (Voltage_B*819))
