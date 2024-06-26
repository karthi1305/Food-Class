# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FOOD.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import pandas as pd
import cv2
import os
from time import sleep


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.IMAGESHOW = QtWidgets.QLabel(self.centralwidget)
        self.IMAGESHOW.setGeometry(QtCore.QRect(200, 30, 211, 131))
        self.IMAGESHOW.setFrameShape(QtWidgets.QFrame.Box)
        self.IMAGESHOW.setText("")
        self.IMAGESHOW.setObjectName("IMAGESHOW")
        self.BROWSE_IMAGE = QtWidgets.QPushButton(self.centralwidget)
        self.BROWSE_IMAGE.setGeometry(QtCore.QRect(190, 190, 91, 23))
        self.BROWSE_IMAGE.setObjectName("BROWSE_IMAGE")
        self.CLASSIFY = QtWidgets.QPushButton(self.centralwidget)
        self.CLASSIFY.setGeometry(QtCore.QRect(340, 190, 75, 23))
        self.CLASSIFY.setObjectName("CLASSIFY")
        self.FOOD_NAME = QtWidgets.QLabel(self.centralwidget)
        self.FOOD_NAME.setGeometry(QtCore.QRect(180, 240, 131, 31))
        self.FOOD_NAME.setObjectName("FOOD_NAME")
        self.PROTIEN = QtWidgets.QLabel(self.centralwidget)
        self.PROTIEN.setGeometry(QtCore.QRect(180, 280, 101, 31))
        self.PROTIEN.setObjectName("PROTIEN")
        self.CALCIUM = QtWidgets.QLabel(self.centralwidget)
        self.CALCIUM.setGeometry(QtCore.QRect(180, 330, 101, 31))
        self.CALCIUM.setObjectName("CALCIUM")
        self.SUGGESTION = QtWidgets.QLabel(self.centralwidget)
        self.SUGGESTION.setGeometry(QtCore.QRect(180, 380, 101, 31))
        self.SUGGESTION.setObjectName("SUGGESTION")
        self.FOOD_NAME_VALUE = QtWidgets.QLabel(self.centralwidget)
        self.FOOD_NAME_VALUE.setGeometry(QtCore.QRect(330, 240, 161, 31))
        self.FOOD_NAME_VALUE.setFrameShape(QtWidgets.QFrame.Box)
        self.FOOD_NAME_VALUE.setText("")
        self.FOOD_NAME_VALUE.setObjectName("FOOD_NAME_VALUE")
        self.PROTIEN_VALUE = QtWidgets.QLabel(self.centralwidget)
        self.PROTIEN_VALUE.setGeometry(QtCore.QRect(330, 280, 161, 31))
        self.PROTIEN_VALUE.setFrameShape(QtWidgets.QFrame.Box)
        self.PROTIEN_VALUE.setText("")
        self.PROTIEN_VALUE.setObjectName("PROTIEN_VALUE")
        self.CALCIUM_VALUE = QtWidgets.QLabel(self.centralwidget)
        self.CALCIUM_VALUE.setGeometry(QtCore.QRect(330, 330, 161, 31))
        self.CALCIUM_VALUE.setFrameShape(QtWidgets.QFrame.Box)
        self.CALCIUM_VALUE.setText("")
        self.CALCIUM_VALUE.setObjectName("CALCIUM_VALUE")
        self.lSUGGESTION_VALUE = QtWidgets.QLabel(self.centralwidget)
        self.lSUGGESTION_VALUE.setGeometry(QtCore.QRect(330, 380, 301, 31))
        self.lSUGGESTION_VALUE.setFrameShape(QtWidgets.QFrame.Box)
        self.lSUGGESTION_VALUE.setText("")
        self.lSUGGESTION_VALUE.setObjectName("lSUGGESTION_VALUE")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.BROWSE_IMAGE.clicked.connect(self.loadImage)
        self.CLASSIFY.clicked.connect(self.classifyFunction)

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.IMAGESHOW.width(), self.IMAGESHOW.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.IMAGESHOW.setPixmap(pixmap) # Set the pixmap onto the label
            self.IMAGESHOW.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center

    def classifyFunction(self):
        json_file = open('model1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model1.h5")
        print("Loaded model from disk")
        path2=self.file
        test_image = image.load_img(path2, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)


        a = np.round(result[0][0])
        b = np.round(result[0][1])
        c = np.round(result[0][2])
        d = np.round(result[0][3])
        e = np.round(result[0][4])
        f = np.round(result[0][5])
        g = np.round(result[0][6])
        h = np.round(result[0][7])

        print(a)
        print(b)
        print(c)
        print(d)
        print(e)
        print(f)
        print(g)
        print(h)

        if a == 1:
            prediction = 'burger'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '271'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '260'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "DIABETIC, SUGAR AND PRESSURE PATIENT DON'T EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif b == 1:
            prediction = 'chicken briyani'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "DIABETIC, SUGAR AND PRESSURE PATIENT DON'T EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif c == 1:
            prediction = 'Dosa'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif d == 1:
            prediction = 'Idly'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif e == 1:
            prediction = 'Pizza'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "DIABETIC, SUGAR AND PRESSURE PATIENT DON'T EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif f == 1:
            prediction = 'Pongal'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "DIABETIC, SUGAR AND PRESSURE PATIENT EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif g == 1:
            prediction = 'Poori'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "DIABETIC, SUGAR AND PRESSURE PATIENT DON'T EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

        elif h == 1:
            prediction = 'White Rice'
            label=prediction
            self.FOOD_NAME_VALUE.setText(label)
            prediction1 =  '400'
            label2 = prediction1
            self.PROTIEN_VALUE.setText(label2)
            price = '150'
            label_3 = price
            self.CALCIUM_VALUE.setText(label_3)
            price1 = "EAT"
            label_3 = price1
            self.lSUGGESTION_VALUE.setText(label_3)

            
        
        # Add other classification conditions similarly

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BROWSE_IMAGE.setText(_translate("MainWindow", "BROWSE_IMAGE"))
        self.CLASSIFY.setText(_translate("MainWindow", "CLASSIFY"))
        self.FOOD_NAME.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">FOOD NAME</span></p></body></html>"))
        self.PROTIEN.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">PROTEIN VALUE</span></p></body></html>"))
        self.CALCIUM.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">CALCIUM</span></p></body></html>"))
        self.SUGGESTION.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">SUGGESTION</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
