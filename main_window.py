# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 768)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selectPic = QtWidgets.QPushButton(self.centralwidget)
        self.selectPic.setGeometry(QtCore.QRect(80, 20, 75, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.selectPic.setFont(font)
        self.selectPic.setStyleSheet("background-color: rgb(239, 239, 239);\n"
"border: 1px solid rgb(224, 130, 43);")
        self.selectPic.setObjectName("selectPic")
        self.showImage_1 = QtWidgets.QLabel(self.centralwidget)
        self.showImage_1.setGeometry(QtCore.QRect(80, 73, 512, 512))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(25)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.showImage_1.sizePolicy().hasHeightForWidth())
        self.showImage_1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(24)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.showImage_1.setFont(font)
        self.showImage_1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.showImage_1.setAutoFillBackground(False)
        self.showImage_1.setStyleSheet("border: 2px solid rgb(224, 130, 43);\n"
"background-color:rgb(239, 239, 239)")
        self.showImage_1.setText("")
        self.showImage_1.setAlignment(QtCore.Qt.AlignCenter)
        self.showImage_1.setWordWrap(False)
        self.showImage_1.setObjectName("showImage_1")
        self.showImage_2 = QtWidgets.QLabel(self.centralwidget)
        self.showImage_2.setGeometry(QtCore.QRect(680, 73, 512, 512))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(25)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.showImage_2.sizePolicy().hasHeightForWidth())
        self.showImage_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(24)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.showImage_2.setFont(font)
        self.showImage_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.showImage_2.setAutoFillBackground(False)
        self.showImage_2.setStyleSheet("border: 2px solid rgb(224, 130, 43);\n"
"background-color:rgb(239, 239, 239)")
        self.showImage_2.setText("")
        self.showImage_2.setAlignment(QtCore.Qt.AlignCenter)
        self.showImage_2.setWordWrap(False)
        self.showImage_2.setObjectName("showImage_2")
        self.segPic = QtWidgets.QPushButton(self.centralwidget)
        self.segPic.setGeometry(QtCore.QRect(680, 20, 75, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.segPic.setFont(font)
        self.segPic.setStyleSheet("background-color: rgb(239, 239, 239);\n"
"border: 1px solid rgb(224, 130, 43);")
        self.segPic.setObjectName("segPic")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(200, 20, 150, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("background-color: rgb(239, 239, 239);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(380, 20, 130, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("border-color: rgb(220, 130, 43);")
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(520, 20, 70, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(224, 130, 43);")
        self.lineEdit.setObjectName("lineEdit")
        self.savePic = QtWidgets.QPushButton(self.centralwidget)
        self.savePic.setGeometry(QtCore.QRect(780, 20, 75, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.savePic.setFont(font)
        self.savePic.setStyleSheet("background-color: rgb(239, 239, 239);\n"
"border: 1px solid rgb(224, 130, 43);")
        self.savePic.setObjectName("savePic")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(80, 624, 1111, 100))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser.setFont(font)
        self.textBrowser.setStyleSheet("border: 2px solid rgb(224, 130, 43);\n"
"background-color:rgb(239, 239, 239)")
        self.textBrowser.setObjectName("textBrowser")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(80, 564, 1111, 75))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setBold(True)
        font.setWeight(75)
        self.layoutWidget.setFont(font)
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(100)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.batchPic = QtWidgets.QPushButton(self.centralwidget)
        self.batchPic.setGeometry(QtCore.QRect(880, 20, 150, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.batchPic.setFont(font)
        self.batchPic.setStyleSheet("background-color: rgb(239, 239, 239);\n"
"border: 1px solid rgb(224, 130, 43);")
        self.batchPic.setObjectName("batchPic")
        self.resetButton = QtWidgets.QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QtCore.QRect(1116, 20, 75, 35))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.resetButton.setFont(font)
        self.resetButton.setStyleSheet("background-color:rgb(224, 130, 43);")
        self.resetButton.setObjectName("resetButton")
        self.layoutWidget.raise_()
        self.selectPic.raise_()
        self.segPic.raise_()
        self.comboBox.raise_()
        self.label.raise_()
        self.lineEdit.raise_()
        self.savePic.raise_()
        self.textBrowser.raise_()
        self.showImage_1.raise_()
        self.showImage_2.raise_()
        self.batchPic.raise_()
        self.resetButton.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setTitle("")
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像分割主界面"))
        self.selectPic.setText(_translate("MainWindow", "图像载入"))
        self.segPic.setText(_translate("MainWindow", "执行分割"))
        self.comboBox.setItemText(0, _translate("MainWindow", "阈值分割"))
        self.comboBox.setItemText(1, _translate("MainWindow", "OTSU分割"))
        self.comboBox.setItemText(2, _translate("MainWindow", "亮点分割"))
        self.label.setText(_translate("MainWindow", "输入阈值(0-255)："))
        self.savePic.setText(_translate("MainWindow", "保存"))
        self.label_2.setText(_translate("MainWindow", "原始图像"))
        self.label_3.setText(_translate("MainWindow", "分割图像"))
        self.batchPic.setText(_translate("MainWindow", "批量处理文件夹"))
        self.resetButton.setText(_translate("MainWindow", "重置"))
