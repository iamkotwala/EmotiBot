# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'asdf.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(944, 596)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.on = QtWidgets.QPushButton(self.centralwidget)
        self.on.setGeometry(QtCore.QRect(620, 500, 141, 51))
        self.on.setObjectName("on")
        self.off = QtWidgets.QPushButton(self.centralwidget)
        self.off.setGeometry(QtCore.QRect(780, 500, 131, 51))
        self.off.setObjectName("off")
        self.cam = QtWidgets.QFrame(self.centralwidget)
        self.cam.setGeometry(QtCore.QRect(10, 10, 581, 541))
        self.cam.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cam.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cam.setObjectName("cam")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(600, 320, 331, 161))
        self.frame.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalScrollBar = QtWidgets.QScrollBar(self.frame)
        self.verticalScrollBar.setGeometry(QtCore.QRect(300, 0, 16, 161))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(750, 290, 31, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(600, -40, 331, 321))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 944, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuWindow = QtWidgets.QMenu(self.menubar)
        self.menuWindow.setObjectName("menuWindow")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuWindow.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.on.setText(_translate("MainWindow", "PLAY"))
        self.off.setText(_translate("MainWindow", "STOP"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">LOGS</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><img src=\":/yes/D:/0 Internship/3 Presentation for university/MidSem/Logo.png\"/></p></body></html>"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))

import asdf_rc
