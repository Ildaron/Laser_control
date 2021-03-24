import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize    
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.Qt import *
import random
import threading 
import time
import numpy as np
import pandas as pd
from scipy import signal
import serial



global fs
fs = 50   # Частота дискретизации или частота дискретизации, F S , представляет собой среднее число образцов , полученных в одну секунду ( выборок в секунду ),

global cutoff
cutoff = 1
global cutoffs
cutoffs = 10
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
               
        self.setMinimumSize(QSize(300, 200))    
        self.setWindowTitle("BCI") 
        pybutton = QPushButton('Click me', self)        
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(100,32)
        pybutton.move(50, 50)        
    def clickMethod(self):
        print('Clicked Pyqt button.')

        seconWin.show()
        mainWin.close()

class second_window(QWidget):

    def __init__(self):
        print ("start")       
        QWidget.__init__(self)

        self.setMinimumSize(QSize(600, 500))    
        self.setWindowTitle("Iron_BCI") 
              

        self.figure = plt.figure(figsize=(0,2,),facecolor='y',  edgecolor='r') #  color only     
        self.figure1 = plt.figure(figsize=(0,2),facecolor='y') # color only
        self.figure2 = plt.figure(figsize=(0,2),facecolor='y')
        self.figure3 = plt.figure(figsize=(0,2),facecolor='y')
                
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(0.2, 0.4, 0.8, 1)  # only graph 
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.subplots_adjust(0.2, 0.4, 0.8, 1)  # only graph 
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure2.subplots_adjust(0.2, 0.4, 0.8, 1)
        self.canvas3 = FigureCanvas(self.figure3)
        self.figure3.subplots_adjust(0.2, 0.4, 0.8, 1)
       # self.toolbar = NavigationToolbar(self.canvas, self)
       # self.toolbar1 = NavigationToolbar(self.canvas1, self)
        
        pybutton = QPushButton('graph', self)

        
        global axis_x
        axis_x=0

        pybutton.clicked.connect(self.clickMethod)
        pybutton.move(350, 10)
        pybutton.resize(100,32)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(50,100,0,11) # move background
        layout.setGeometry(QRect(0, 0, 80, 68))# nothing  
      # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)    
      # layout.addWidget(self.toolbar1)
        layout.addWidget(self.canvas1)        
        layout.addWidget(self.canvas2)  # background
        layout.addWidget(self.canvas3)
       # input dataufoff value  
        self.le_num1 = QLineEdit()
        self.le_num1.setFixedSize(50, 20) # size                        
        self.pb_num1 = QPushButton('HPS')
        self.pb_num1.setFixedSize(50, 60) # size
        self.pb_num1.clicked.connect(self.show_dialog_num1)
        layout.addWidget(self.le_num1)       
        self.pb_num1.move(10, 0)        
        layout.addWidget(self.pb_num1)
        self.setLayout(layout)
        # stop input data
        # start input data fps
        self.le_num2 = QLineEdit()
        self.le_num2.setFixedSize(50, 20) # size                        
        self.pb_num2 = QPushButton('fs')
        self.pb_num2.setFixedSize(50, 60) # size
        self.pb_num2.clicked.connect(self.show_dialog_num2)
        layout.addWidget(self.le_num2)       
        self.pb_num1.move(90, 100)        
        layout.addWidget(self.pb_num2)
        self.setLayout(layout)
        # stop input data fps
        # start input data low filter
        self.le_num3 = QLineEdit()
        self.le_num3.setFixedSize(50, 20) # size                        
        self.pb_num3 = QPushButton('LPF')
        self.pb_num3.setFixedSize(50, 60) # size
        self.pb_num3.clicked.connect(self.show_dialog_num3)     
        layout.addWidget(self.le_num3)       
        self.pb_num1.move(190, 100)        
        layout.addWidget(self.pb_num3)
        self.setLayout(layout)
        # stop input data filter
    
    def clickMethod(self):



#прогать здесь






        
         thread=threading.Thread(target=self.clickMethod, args=())
         thread.start()
         
# input data
    def show_dialog_num1(self):
        value, r = QInputDialog.getInt(self, 'Input dialog', 'HPS:')
        global cutoff
        cutoff = value
        print (cutoff)
    def show_dialog_num2(self):
        value, r = QInputDialog.getInt(self, 'Input dialog', 'fs:')
        global fs
        fs = value
        print (fs)
    def show_dialog_num3(self):
        value, r = QInputDialog.getInt(self, 'Input dialog', 'LPF:')
        global cutoffs
        cutoffs = value
        print (cutoffs)
        
       # thread.start()
       # mainWin.show()
       # seconWin.close()  

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    seconWin = second_window()
   
    
    sys.exit( app.exec_() )
