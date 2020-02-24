# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SRAmainGUI.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *

from mplwidget import MplWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1084, 588)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.graphWidget = MplWidget(self.centralwidget)
        self.graphWidget.setObjectName(u"graphWidget")
        self.graphWidget.setGeometry(QRect(30, 60, 421, 431))
        self.tableWidget_Soil = QTableWidget(self.centralwidget)
        if (self.tableWidget_Soil.columnCount() < 3):
            self.tableWidget_Soil.setColumnCount(3)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.tableWidget_Soil.setObjectName(u"tableWidget_Soil")
        self.tableWidget_Soil.setGeometry(QRect(490, 60, 421, 192))
        self.tableWidget_Soil.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.pushButton_addSoil = QPushButton(self.centralwidget)
        self.pushButton_addSoil.setObjectName(u"pushButton_addSoil")
        self.pushButton_addSoil.setGeometry(QRect(940, 90, 93, 28))
        self.label_SoilProp = QLabel(self.centralwidget)
        self.label_SoilProp.setObjectName(u"label_SoilProp")
        self.label_SoilProp.setGeometry(QRect(500, 30, 101, 16))
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_SoilProp.setFont(font)
        self.pushButton_removeSoil = QPushButton(self.centralwidget)
        self.pushButton_removeSoil.setObjectName(u"pushButton_removeSoil")
        self.pushButton_removeSoil.setGeometry(QRect(940, 130, 93, 28))
        self.tableWidget_Profile = QTableWidget(self.centralwidget)
        if (self.tableWidget_Profile.columnCount() < 4):
            self.tableWidget_Profile.setColumnCount(4)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(0, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(1, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(2, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(3, __qtablewidgetitem6)
        self.tableWidget_Profile.setObjectName(u"tableWidget_Profile")
        self.tableWidget_Profile.setGeometry(QRect(490, 300, 421, 192))
        self.tableWidget_Profile.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.label_profileProp = QLabel(self.centralwidget)
        self.label_profileProp.setObjectName(u"label_profileProp")
        self.label_profileProp.setGeometry(QRect(490, 270, 101, 16))
        self.label_profileProp.setFont(font)
        self.pushButton_removeProfile = QPushButton(self.centralwidget)
        self.pushButton_removeProfile.setObjectName(u"pushButton_removeProfile")
        self.pushButton_removeProfile.setGeometry(QRect(940, 400, 93, 28))
        self.pushButton_addProfile = QPushButton(self.centralwidget)
        self.pushButton_addProfile.setObjectName(u"pushButton_addProfile")
        self.pushButton_addProfile.setGeometry(QRect(940, 360, 93, 28))
        self.pushButton_drawProfile = QPushButton(self.centralwidget)
        self.pushButton_drawProfile.setObjectName(u"pushButton_drawProfile")
        self.pushButton_drawProfile.setGeometry(QRect(940, 440, 93, 28))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(490, 520, 55, 16))
        self.label.setFont(font)
        self.lineEdit_bedWeight = QLineEdit(self.centralwidget)
        self.lineEdit_bedWeight.setObjectName(u"lineEdit_bedWeight")
        self.lineEdit_bedWeight.setGeometry(QRect(580, 520, 113, 22))
        self.label_bedUnits = QLabel(self.centralwidget)
        self.label_bedUnits.setObjectName(u"label_bedUnits")
        self.label_bedUnits.setGeometry(QRect(710, 520, 55, 16))
        font1 = QFont()
        font1.setBold(False)
        font1.setWeight(50)
        self.label_bedUnits.setFont(font1)
        self.label_bedUnits.setTextFormat(Qt.MarkdownText)
        self.lineEdit_bedVelocity = QLineEdit(self.centralwidget)
        self.lineEdit_bedVelocity.setObjectName(u"lineEdit_bedVelocity")
        self.lineEdit_bedVelocity.setGeometry(QRect(780, 520, 113, 22))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(910, 520, 55, 16))
        self.label_2.setFont(font1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1084, 26))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SRA batcher", None))
        ___qtablewidgetitem = self.tableWidget_Soil.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Name", None));
        ___qtablewidgetitem1 = self.tableWidget_Soil.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Unit Weight", None));
        ___qtablewidgetitem2 = self.tableWidget_Soil.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Degradation curve", None));
        self.pushButton_addSoil.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.label_SoilProp.setText(QCoreApplication.translate("MainWindow", u"Soil properties", None))
        self.pushButton_removeSoil.setText(QCoreApplication.translate("MainWindow", u"-", None))
        ___qtablewidgetitem3 = self.tableWidget_Profile.horizontalHeaderItem(0)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"Depth [m]", None));
        ___qtablewidgetitem4 = self.tableWidget_Profile.horizontalHeaderItem(1)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Thickness [m]", None));
        ___qtablewidgetitem5 = self.tableWidget_Profile.horizontalHeaderItem(2)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"Soil type", None));
        ___qtablewidgetitem6 = self.tableWidget_Profile.horizontalHeaderItem(3)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"Vs [m/s]", None));
        self.label_profileProp.setText(QCoreApplication.translate("MainWindow", u"Profile", None))
        self.pushButton_removeProfile.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.pushButton_addProfile.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.pushButton_drawProfile.setText(QCoreApplication.translate("MainWindow", u"Draw profile", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Bedrock", None))
        self.label_bedUnits.setText(QCoreApplication.translate("MainWindow", u"KN/m<sup>3</sup>", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"m/s", None))
    # retranslateUi

