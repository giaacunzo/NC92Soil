# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SRAmainGUI.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

from mplwidget import MplWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1323, 901)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.graphWidget = MplWidget(self.centralwidget)
        self.graphWidget.setObjectName(u"graphWidget")
        self.graphWidget.setGeometry(QRect(30, 70, 551, 371))
        self.tableWidget_Soil = QTableWidget(self.centralwidget)
        if (self.tableWidget_Soil.columnCount() < 6):
            self.tableWidget_Soil.setColumnCount(6)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(4, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(5, __qtablewidgetitem5)
        self.tableWidget_Soil.setObjectName(u"tableWidget_Soil")
        self.tableWidget_Soil.setGeometry(QRect(600, 60, 551, 221))
        self.tableWidget_Soil.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableWidget_Soil.verticalHeader().setDefaultSectionSize(30)
        self.pushButton_addSoil = QPushButton(self.centralwidget)
        self.pushButton_addSoil.setObjectName(u"pushButton_addSoil")
        self.pushButton_addSoil.setGeometry(QRect(1180, 90, 93, 28))
        self.label_SoilProp = QLabel(self.centralwidget)
        self.label_SoilProp.setObjectName(u"label_SoilProp")
        self.label_SoilProp.setGeometry(QRect(610, 30, 101, 16))
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_SoilProp.setFont(font)
        self.pushButton_removeSoil = QPushButton(self.centralwidget)
        self.pushButton_removeSoil.setObjectName(u"pushButton_removeSoil")
        self.pushButton_removeSoil.setGeometry(QRect(1180, 130, 93, 28))
        self.tableWidget_Profile = QTableWidget(self.centralwidget)
        if (self.tableWidget_Profile.columnCount() < 3):
            self.tableWidget_Profile.setColumnCount(3)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(0, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(1, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(2, __qtablewidgetitem8)
        self.tableWidget_Profile.setObjectName(u"tableWidget_Profile")
        self.tableWidget_Profile.setGeometry(QRect(600, 330, 551, 191))
        self.tableWidget_Profile.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableWidget_Profile.verticalHeader().setDefaultSectionSize(30)
        self.label_profileProp = QLabel(self.centralwidget)
        self.label_profileProp.setObjectName(u"label_profileProp")
        self.label_profileProp.setGeometry(QRect(600, 300, 101, 16))
        self.label_profileProp.setFont(font)
        self.pushButton_removeProfile = QPushButton(self.centralwidget)
        self.pushButton_removeProfile.setObjectName(u"pushButton_removeProfile")
        self.pushButton_removeProfile.setGeometry(QRect(1180, 450, 93, 28))
        self.pushButton_addProfile = QPushButton(self.centralwidget)
        self.pushButton_addProfile.setObjectName(u"pushButton_addProfile")
        self.pushButton_addProfile.setGeometry(QRect(1180, 410, 93, 28))
        self.pushButton_drawProfile = QPushButton(self.centralwidget)
        self.pushButton_drawProfile.setObjectName(u"pushButton_drawProfile")
        self.pushButton_drawProfile.setGeometry(QRect(1180, 490, 93, 28))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(600, 530, 55, 16))
        self.label.setFont(font)
        self.lineEdit_bedWeight = QLineEdit(self.centralwidget)
        self.lineEdit_bedWeight.setObjectName(u"lineEdit_bedWeight")
        self.lineEdit_bedWeight.setGeometry(QRect(670, 530, 71, 22))
        self.label_bedUnits = QLabel(self.centralwidget)
        self.label_bedUnits.setObjectName(u"label_bedUnits")
        self.label_bedUnits.setGeometry(QRect(758, 530, 71, 16))
        font1 = QFont()
        font1.setBold(False)
        font1.setWeight(50)
        self.label_bedUnits.setFont(font1)
        self.label_bedUnits.setTextFormat(Qt.RichText)
        self.lineEdit_bedVelocity = QLineEdit(self.centralwidget)
        self.lineEdit_bedVelocity.setObjectName(u"lineEdit_bedVelocity")
        self.lineEdit_bedVelocity.setGeometry(QRect(870, 530, 71, 22))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(950, 530, 55, 16))
        self.label_2.setFont(font1)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(30, 560, 1261, 16))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.groupBox_output = QGroupBox(self.centralwidget)
        self.groupBox_output.setObjectName(u"groupBox_output")
        self.groupBox_output.setGeometry(QRect(780, 580, 361, 161))
        self.checkBox_outRS = QCheckBox(self.groupBox_output)
        self.checkBox_outRS.setObjectName(u"checkBox_outRS")
        self.checkBox_outRS.setGeometry(QRect(20, 20, 151, 20))
        self.checkBox_outRS.setChecked(True)
        self.lineEdit_RSDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_RSDepth.setObjectName(u"lineEdit_RSDepth")
        self.lineEdit_RSDepth.setGeometry(QRect(202, 20, 91, 22))
        self.label_3 = QLabel(self.groupBox_output)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(300, 20, 21, 16))
        self.label_3.setFont(font1)
        self.checkBox_outStrain = QCheckBox(self.groupBox_output)
        self.checkBox_outStrain.setObjectName(u"checkBox_outStrain")
        self.checkBox_outStrain.setGeometry(QRect(20, 80, 151, 20))
        self.lineEdit_strainDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_strainDepth.setObjectName(u"lineEdit_strainDepth")
        self.lineEdit_strainDepth.setGeometry(QRect(200, 80, 91, 22))
        self.label_4 = QLabel(self.groupBox_output)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(300, 80, 21, 16))
        self.label_4.setFont(font1)
        self.checkBox_outAcc = QCheckBox(self.groupBox_output)
        self.checkBox_outAcc.setObjectName(u"checkBox_outAcc")
        self.checkBox_outAcc.setGeometry(QRect(20, 50, 151, 20))
        self.lineEdit_accDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_accDepth.setObjectName(u"lineEdit_accDepth")
        self.lineEdit_accDepth.setGeometry(QRect(202, 50, 91, 22))
        self.label_5 = QLabel(self.groupBox_output)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(300, 50, 21, 16))
        self.label_5.setFont(font1)
        self.checkBox_outBrief = QCheckBox(self.groupBox_output)
        self.checkBox_outBrief.setObjectName(u"checkBox_outBrief")
        self.checkBox_outBrief.setEnabled(True)
        self.checkBox_outBrief.setGeometry(QRect(20, 110, 151, 20))
        self.checkBox_outBrief.setChecked(True)
        self.lineEdit_briefDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_briefDepth.setObjectName(u"lineEdit_briefDepth")
        self.lineEdit_briefDepth.setEnabled(False)
        self.lineEdit_briefDepth.setGeometry(QRect(200, 110, 91, 22))
        self.label_17 = QLabel(self.groupBox_output)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setEnabled(False)
        self.label_17.setGeometry(QRect(300, 110, 21, 16))
        self.label_17.setFont(font1)
        self.pushButton_run = QPushButton(self.centralwidget)
        self.pushButton_run.setObjectName(u"pushButton_run")
        self.pushButton_run.setGeometry(QRect(1020, 840, 121, 28))
        self.lineEdit_maxFreq = QLineEdit(self.centralwidget)
        self.lineEdit_maxFreq.setObjectName(u"lineEdit_maxFreq")
        self.lineEdit_maxFreq.setGeometry(QRect(160, 530, 71, 22))
        self.lineEdit_waveLength = QLineEdit(self.centralwidget)
        self.lineEdit_waveLength.setObjectName(u"lineEdit_waveLength")
        self.lineEdit_waveLength.setGeometry(QRect(380, 530, 71, 22))
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(30, 530, 121, 16))
        self.label_9.setFont(font1)
        self.checkBox_autoDiscretize = QCheckBox(self.centralwidget)
        self.checkBox_autoDiscretize.setObjectName(u"checkBox_autoDiscretize")
        self.checkBox_autoDiscretize.setGeometry(QRect(470, 530, 111, 20))
        self.checkBox_autoDiscretize.setChecked(True)
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(250, 530, 121, 16))
        self.label_10.setFont(font1)
        self.lineEdit_bedDamping = QLineEdit(self.centralwidget)
        self.lineEdit_bedDamping.setObjectName(u"lineEdit_bedDamping")
        self.lineEdit_bedDamping.setGeometry(QRect(1052, 530, 71, 22))
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(1130, 530, 55, 16))
        self.label_6.setFont(font1)
        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(780, 790, 121, 16))
        self.label_11.setFont(font1)
        self.lineEdit_strainRatio = QLineEdit(self.centralwidget)
        self.lineEdit_strainRatio.setObjectName(u"lineEdit_strainRatio")
        self.lineEdit_strainRatio.setGeometry(QRect(920, 790, 61, 22))
        self.lineEdit_maxIter = QLineEdit(self.centralwidget)
        self.lineEdit_maxIter.setObjectName(u"lineEdit_maxIter")
        self.lineEdit_maxIter.setGeometry(QRect(920, 820, 61, 22))
        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(780, 820, 121, 16))
        self.label_12.setFont(font1)
        self.lineEdit_maxTol = QLineEdit(self.centralwidget)
        self.lineEdit_maxTol.setObjectName(u"lineEdit_maxTol")
        self.lineEdit_maxTol.setGeometry(QRect(920, 850, 61, 22))
        self.label_13 = QLabel(self.centralwidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(780, 850, 121, 16))
        self.label_13.setFont(font1)
        self.comboBox_analysisType = QComboBox(self.centralwidget)
        self.comboBox_analysisType.addItem("")
        self.comboBox_analysisType.addItem("")
        self.comboBox_analysisType.setObjectName(u"comboBox_analysisType")
        self.comboBox_analysisType.setGeometry(QRect(1180, 10, 141, 22))
        self.label_15 = QLabel(self.centralwidget)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(1160, 330, 61, 16))
        self.label_15.setFont(font1)
        self.lineEdit_brickSize = QLineEdit(self.centralwidget)
        self.lineEdit_brickSize.setObjectName(u"lineEdit_brickSize")
        self.lineEdit_brickSize.setEnabled(False)
        self.lineEdit_brickSize.setGeometry(QRect(1250, 330, 41, 22))
        self.lineEdit_brickSize.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_16 = QLabel(self.centralwidget)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(1300, 330, 21, 16))
        self.label_16.setFont(font1)
        self.plainTextEdit_overview = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_overview.setObjectName(u"plainTextEdit_overview")
        self.plainTextEdit_overview.setEnabled(False)
        self.plainTextEdit_overview.setGeometry(QRect(30, 450, 551, 71))
        self.plainTextEdit_overview.setTextInteractionFlags(Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)
        self.groupBox_TH = QGroupBox(self.centralwidget)
        self.groupBox_TH.setObjectName(u"groupBox_TH")
        self.groupBox_TH.setGeometry(QRect(20, 570, 741, 281))
        self.groupBox_TH.setAutoFillBackground(True)
        self.graphWidget_TH = MplWidget(self.groupBox_TH)
        self.graphWidget_TH.setObjectName(u"graphWidget_TH")
        self.graphWidget_TH.setGeometry(QRect(10, 20, 701, 151))
        self.lineEdit_FS = QLineEdit(self.groupBox_TH)
        self.lineEdit_FS.setObjectName(u"lineEdit_FS")
        self.lineEdit_FS.setGeometry(QRect(70, 190, 113, 22))
        self.label_7 = QLabel(self.groupBox_TH)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(10, 190, 51, 16))
        self.label_7.setFont(font1)
        self.label_8 = QLabel(self.groupBox_TH)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(260, 190, 81, 16))
        self.label_8.setFont(font1)
        self.comboBox_Units = QComboBox(self.groupBox_TH)
        self.comboBox_Units.addItem("")
        self.comboBox_Units.addItem("")
        self.comboBox_Units.addItem("")
        self.comboBox_Units.setObjectName(u"comboBox_Units")
        self.comboBox_Units.setGeometry(QRect(350, 190, 91, 22))
        self.pushButton_loadTH = QPushButton(self.groupBox_TH)
        self.pushButton_loadTH.setObjectName(u"pushButton_loadTH")
        self.pushButton_loadTH.setGeometry(QRect(590, 190, 121, 28))
        self.comboBox_eventList = QComboBox(self.groupBox_TH)
        self.comboBox_eventList.setObjectName(u"comboBox_eventList")
        self.comboBox_eventList.setEnabled(False)
        self.comboBox_eventList.setGeometry(QRect(100, 240, 621, 22))
        self.label_14 = QLabel(self.groupBox_TH)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(10, 240, 71, 16))
        self.label_14.setFont(font1)
        self.comboBox_THorRVT = QComboBox(self.centralwidget)
        self.comboBox_THorRVT.addItem("")
        self.comboBox_THorRVT.addItem("")
        self.comboBox_THorRVT.setObjectName(u"comboBox_THorRVT")
        self.comboBox_THorRVT.setGeometry(QRect(1020, 790, 121, 22))
        self.groupBox_RVT = QGroupBox(self.centralwidget)
        self.groupBox_RVT.setObjectName(u"groupBox_RVT")
        self.groupBox_RVT.setGeometry(QRect(20, 570, 741, 301))
        self.groupBox_RVT.setAutoFillBackground(True)
        self.label_18 = QLabel(self.groupBox_RVT)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(10, 240, 71, 16))
        self.comboBox_spectraList = QComboBox(self.groupBox_RVT)
        self.comboBox_spectraList.setObjectName(u"comboBox_spectraList")
        self.comboBox_spectraList.setEnabled(False)
        self.comboBox_spectraList.setGeometry(QRect(100, 240, 621, 22))
        self.graphWidget_Spectrum = MplWidget(self.groupBox_RVT)
        self.graphWidget_Spectrum.setObjectName(u"graphWidget_Spectrum")
        self.graphWidget_Spectrum.setGeometry(QRect(10, 20, 701, 151))
        self.label_19 = QLabel(self.groupBox_RVT)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(360, 190, 81, 16))
        self.label_19.setFont(font1)
        self.pushButton_loadSpectra = QPushButton(self.groupBox_RVT)
        self.pushButton_loadSpectra.setObjectName(u"pushButton_loadSpectra")
        self.pushButton_loadSpectra.setGeometry(QRect(590, 190, 121, 28))
        self.comboBox_SpectraUnits = QComboBox(self.groupBox_RVT)
        self.comboBox_SpectraUnits.addItem("")
        self.comboBox_SpectraUnits.addItem("")
        self.comboBox_SpectraUnits.addItem("")
        self.comboBox_SpectraUnits.setObjectName(u"comboBox_SpectraUnits")
        self.comboBox_SpectraUnits.setGeometry(QRect(450, 190, 91, 22))
        self.label_20 = QLabel(self.groupBox_RVT)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(10, 190, 81, 16))
        self.lineEdit_duration = QLineEdit(self.groupBox_RVT)
        self.lineEdit_duration.setObjectName(u"lineEdit_duration")
        self.lineEdit_duration.setGeometry(QRect(100, 190, 51, 22))
        self.lineEdit_damping = QLineEdit(self.groupBox_RVT)
        self.lineEdit_damping.setObjectName(u"lineEdit_damping")
        self.lineEdit_damping.setGeometry(QRect(270, 190, 51, 22))
        self.label_21 = QLabel(self.groupBox_RVT)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(200, 190, 61, 16))
        self.comboBox_showWhat = QComboBox(self.groupBox_RVT)
        self.comboBox_showWhat.addItem("")
        self.comboBox_showWhat.addItem("")
        self.comboBox_showWhat.setObjectName(u"comboBox_showWhat")
        self.comboBox_showWhat.setGeometry(QRect(10, 270, 131, 22))
        self.checkBox_xlog = QCheckBox(self.groupBox_RVT)
        self.checkBox_xlog.setObjectName(u"checkBox_xlog")
        self.checkBox_xlog.setGeometry(QRect(180, 270, 111, 20))
        self.checkBox_ylog = QCheckBox(self.groupBox_RVT)
        self.checkBox_ylog.setObjectName(u"checkBox_ylog")
        self.checkBox_ylog.setGeometry(QRect(280, 270, 111, 20))
        self.tableWidget_Permutations = QTableWidget(self.centralwidget)
        if (self.tableWidget_Permutations.columnCount() < 2):
            self.tableWidget_Permutations.setColumnCount(2)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.tableWidget_Permutations.setHorizontalHeaderItem(0, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.tableWidget_Permutations.setHorizontalHeaderItem(1, __qtablewidgetitem10)
        self.tableWidget_Permutations.setObjectName(u"tableWidget_Permutations")
        self.tableWidget_Permutations.setGeometry(QRect(600, 330, 551, 191))
        self.tableWidget_Permutations.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableWidget_Permutations.verticalHeader().setDefaultSectionSize(30)
        self.lineEdit_bedDepth = QLineEdit(self.centralwidget)
        self.lineEdit_bedDepth.setObjectName(u"lineEdit_bedDepth")
        self.lineEdit_bedDepth.setEnabled(False)
        self.lineEdit_bedDepth.setGeometry(QRect(1250, 360, 41, 22))
        self.lineEdit_bedDepth.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_22 = QLabel(self.centralwidget)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(1300, 360, 21, 16))
        self.label_22.setFont(font1)
        self.label_23 = QLabel(self.centralwidget)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(1160, 360, 91, 20))
        self.label_23.setFont(font1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.tableWidget_Permutations.raise_()
        self.tableWidget_Profile.raise_()
        self.groupBox_TH.raise_()
        self.groupBox_RVT.raise_()
        self.graphWidget.raise_()
        self.tableWidget_Soil.raise_()
        self.pushButton_addSoil.raise_()
        self.label_SoilProp.raise_()
        self.pushButton_removeSoil.raise_()
        self.label_profileProp.raise_()
        self.pushButton_removeProfile.raise_()
        self.pushButton_addProfile.raise_()
        self.pushButton_drawProfile.raise_()
        self.label.raise_()
        self.lineEdit_bedWeight.raise_()
        self.label_bedUnits.raise_()
        self.lineEdit_bedVelocity.raise_()
        self.label_2.raise_()
        self.line.raise_()
        self.groupBox_output.raise_()
        self.pushButton_run.raise_()
        self.lineEdit_maxFreq.raise_()
        self.lineEdit_waveLength.raise_()
        self.label_9.raise_()
        self.checkBox_autoDiscretize.raise_()
        self.label_10.raise_()
        self.lineEdit_bedDamping.raise_()
        self.label_6.raise_()
        self.label_11.raise_()
        self.lineEdit_strainRatio.raise_()
        self.lineEdit_maxIter.raise_()
        self.label_12.raise_()
        self.lineEdit_maxTol.raise_()
        self.label_13.raise_()
        self.comboBox_analysisType.raise_()
        self.label_15.raise_()
        self.lineEdit_brickSize.raise_()
        self.label_16.raise_()
        self.plainTextEdit_overview.raise_()
        self.comboBox_THorRVT.raise_()
        self.lineEdit_bedDepth.raise_()
        self.label_22.raise_()
        self.label_23.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1323, 26))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"NC92-Soil", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        ___qtablewidgetitem = self.tableWidget_Soil.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Name", None));
        ___qtablewidgetitem1 = self.tableWidget_Soil.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Unit Weight", None));
        ___qtablewidgetitem2 = self.tableWidget_Soil.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"From [m]", None));
        ___qtablewidgetitem3 = self.tableWidget_Soil.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"To [m]", None));
        ___qtablewidgetitem4 = self.tableWidget_Soil.horizontalHeaderItem(4)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Vs [m/s]", None));
        ___qtablewidgetitem5 = self.tableWidget_Soil.horizontalHeaderItem(5)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"Degradation curve", None));
        self.pushButton_addSoil.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.label_SoilProp.setText(QCoreApplication.translate("MainWindow", u"Soil properties", None))
        self.pushButton_removeSoil.setText(QCoreApplication.translate("MainWindow", u"-", None))
        ___qtablewidgetitem6 = self.tableWidget_Profile.horizontalHeaderItem(0)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"Depth [m]", None));
        ___qtablewidgetitem7 = self.tableWidget_Profile.horizontalHeaderItem(1)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"Thickness [m]", None));
        ___qtablewidgetitem8 = self.tableWidget_Profile.horizontalHeaderItem(2)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("MainWindow", u"Soil type", None));
        self.label_profileProp.setText(QCoreApplication.translate("MainWindow", u"Profile", None))
        self.pushButton_removeProfile.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.pushButton_addProfile.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.pushButton_drawProfile.setText(QCoreApplication.translate("MainWindow", u"Create profile", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Bedrock", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedWeight.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock unit weight", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedWeight.setText(QCoreApplication.translate("MainWindow", u"22", None))
        self.label_bedUnits.setText(QCoreApplication.translate("MainWindow", u"KN/m<sup>3</sup>", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedVelocity.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock velocity", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedVelocity.setText(QCoreApplication.translate("MainWindow", u"800", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"m/s", None))
        self.groupBox_output.setTitle(QCoreApplication.translate("MainWindow", u"Output", None))
        self.checkBox_outRS.setText(QCoreApplication.translate("MainWindow", u"Response spectrum", None))
        self.lineEdit_RSDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_outStrain.setText(QCoreApplication.translate("MainWindow", u"Strain time history", None))
        self.lineEdit_strainDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_outAcc.setText(QCoreApplication.translate("MainWindow", u"Acceleration", None))
        self.lineEdit_accDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_outBrief.setText(QCoreApplication.translate("MainWindow", u"PGA, PGV, FA", None))
        self.lineEdit_briefDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.pushButton_run.setText(QCoreApplication.translate("MainWindow", u"Run analysis", None))
        self.lineEdit_maxFreq.setText(QCoreApplication.translate("MainWindow", u"20", None))
        self.lineEdit_waveLength.setText(QCoreApplication.translate("MainWindow", u"0.2", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Max frequency [Hz]", None))
        self.checkBox_autoDiscretize.setText(QCoreApplication.translate("MainWindow", u"Auto discretize", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Wavelength ratio", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedDamping.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock damping", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedDamping.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Effective strain ratio", None))
        self.lineEdit_strainRatio.setText(QCoreApplication.translate("MainWindow", u"0.65", None))
        self.lineEdit_maxIter.setText(QCoreApplication.translate("MainWindow", u"10", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Max iterations", None))
        self.lineEdit_maxTol.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Error tolerance [%]", None))
        self.comboBox_analysisType.setItemText(0, QCoreApplication.translate("MainWindow", u"Regular analysis", None))
        self.comboBox_analysisType.setItemText(1, QCoreApplication.translate("MainWindow", u"Permutations", None))

        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Brick size", None))
        self.lineEdit_brickSize.setText(QCoreApplication.translate("MainWindow", u"3", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.groupBox_TH.setTitle(QCoreApplication.translate("MainWindow", u"Time history", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_FS.setToolTip(QCoreApplication.translate("MainWindow", u"If specified in input file, sample frequency will be automatically imported from input file", None))
#endif // QT_CONFIG(tooltip)
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"FS [Hz]", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Original units", None))
        self.comboBox_Units.setItemText(0, QCoreApplication.translate("MainWindow", u"g", None))
        self.comboBox_Units.setItemText(1, QCoreApplication.translate("MainWindow", u"cm/s^2", None))
        self.comboBox_Units.setItemText(2, QCoreApplication.translate("MainWindow", u"m/s^2", None))

        self.pushButton_loadTH.setText(QCoreApplication.translate("MainWindow", u"Load time history", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"File name", None))
        self.comboBox_THorRVT.setItemText(0, QCoreApplication.translate("MainWindow", u"Time History", None))
        self.comboBox_THorRVT.setItemText(1, QCoreApplication.translate("MainWindow", u"RVT", None))

        self.groupBox_RVT.setTitle(QCoreApplication.translate("MainWindow", u"RVT", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"File name", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Original units", None))
        self.pushButton_loadSpectra.setText(QCoreApplication.translate("MainWindow", u"Load target spectra", None))
        self.comboBox_SpectraUnits.setItemText(0, QCoreApplication.translate("MainWindow", u"g", None))
        self.comboBox_SpectraUnits.setItemText(1, QCoreApplication.translate("MainWindow", u"cm/s^2", None))
        self.comboBox_SpectraUnits.setItemText(2, QCoreApplication.translate("MainWindow", u"m/s^2", None))

        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Duration [s]", None))
        self.lineEdit_duration.setText(QCoreApplication.translate("MainWindow", u"5", None))
        self.lineEdit_damping.setText(QCoreApplication.translate("MainWindow", u"0.05", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Damping", None))
        self.comboBox_showWhat.setItemText(0, QCoreApplication.translate("MainWindow", u"Show RS", None))
        self.comboBox_showWhat.setItemText(1, QCoreApplication.translate("MainWindow", u"Show FAS", None))

        self.checkBox_xlog.setText(QCoreApplication.translate("MainWindow", u"X log scale", None))
        self.checkBox_ylog.setText(QCoreApplication.translate("MainWindow", u"Y log scale", None))
        ___qtablewidgetitem9 = self.tableWidget_Permutations.horizontalHeaderItem(0)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("MainWindow", u"Soil type", None));
        ___qtablewidgetitem10 = self.tableWidget_Permutations.horizontalHeaderItem(1)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("MainWindow", u"Percentage", None));
#if QT_CONFIG(tooltip)
        self.lineEdit_bedDepth.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock unit weight", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedDepth.setText(QCoreApplication.translate("MainWindow", u"30", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Bedrock depth", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"?", None))
    # retranslateUi

