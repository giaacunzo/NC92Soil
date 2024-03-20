# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SRAmainGUI_new_rev3.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QCheckBox, QComboBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLayout, QLineEdit,
    QMainWindow, QMenu, QMenuBar, QPlainTextEdit,
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from mplwidget import MplWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1551, 828)
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.actionGenerateStochastic = QAction(MainWindow)
        self.actionGenerateStochastic.setObjectName(u"actionGenerateStochastic")
        self.actionGeneratePermutated = QAction(MainWindow)
        self.actionGeneratePermutated.setObjectName(u"actionGeneratePermutated")
        self.actionGenerate_NTC = QAction(MainWindow)
        self.actionGenerate_NTC.setObjectName(u"actionGenerate_NTC")
        self.actionGenerate_master_and_sub = QAction(MainWindow)
        self.actionGenerate_master_and_sub.setObjectName(u"actionGenerate_master_and_sub")
        self.actionGenerate_master_report = QAction(MainWindow)
        self.actionGenerate_master_report.setObjectName(u"actionGenerate_master_report")
        self.actionGenerate_only_master = QAction(MainWindow)
        self.actionGenerate_only_master.setObjectName(u"actionGenerate_only_master")
        self.actionLoadspectra = QAction(MainWindow)
        self.actionLoadspectra.setObjectName(u"actionLoadspectra")
        self.actionRun_analysis = QAction(MainWindow)
        self.actionRun_analysis.setObjectName(u"actionRun_analysis")
        self.actionRun_analysis.setEnabled(False)
        self.actionGenerate_UHS_spectra = QAction(MainWindow)
        self.actionGenerate_UHS_spectra.setObjectName(u"actionGenerate_UHS_spectra")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.gridLayout.setVerticalSpacing(-1)
        self.gridLayout_soil = QGridLayout()
        self.gridLayout_soil.setObjectName(u"gridLayout_soil")
        self.gridLayout_soil.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.label_profileProp = QLabel(self.centralwidget)
        self.label_profileProp.setObjectName(u"label_profileProp")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_profileProp.sizePolicy().hasHeightForWidth())
        self.label_profileProp.setSizePolicy(sizePolicy)
        font = QFont()
        font.setBold(True)
        self.label_profileProp.setFont(font)

        self.gridLayout_soil.addWidget(self.label_profileProp, 5, 0, 1, 1)

        self.label_SoilProp = QLabel(self.centralwidget)
        self.label_SoilProp.setObjectName(u"label_SoilProp")
        sizePolicy.setHeightForWidth(self.label_SoilProp.sizePolicy().hasHeightForWidth())
        self.label_SoilProp.setSizePolicy(sizePolicy)
        self.label_SoilProp.setFont(font)

        self.gridLayout_soil.addWidget(self.label_SoilProp, 0, 0, 1, 1)

        self.horizontalLayout_profile = QHBoxLayout()
        self.horizontalLayout_profile.setObjectName(u"horizontalLayout_profile")
        self.tableWidget_Profile = QTableWidget(self.centralwidget)
        if (self.tableWidget_Profile.columnCount() < 3):
            self.tableWidget_Profile.setColumnCount(3)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget_Profile.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.tableWidget_Profile.setObjectName(u"tableWidget_Profile")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tableWidget_Profile.sizePolicy().hasHeightForWidth())
        self.tableWidget_Profile.setSizePolicy(sizePolicy1)
        self.tableWidget_Profile.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableWidget_Profile.verticalHeader().setDefaultSectionSize(30)

        self.horizontalLayout_profile.addWidget(self.tableWidget_Profile)

        self.tableWidget_Permutations = QTableWidget(self.centralwidget)
        if (self.tableWidget_Permutations.columnCount() < 2):
            self.tableWidget_Permutations.setColumnCount(2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget_Permutations.setHorizontalHeaderItem(0, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget_Permutations.setHorizontalHeaderItem(1, __qtablewidgetitem4)
        self.tableWidget_Permutations.setObjectName(u"tableWidget_Permutations")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.tableWidget_Permutations.sizePolicy().hasHeightForWidth())
        self.tableWidget_Permutations.setSizePolicy(sizePolicy2)
        self.tableWidget_Permutations.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableWidget_Permutations.verticalHeader().setDefaultSectionSize(30)

        self.horizontalLayout_profile.addWidget(self.tableWidget_Permutations)


        self.gridLayout_soil.addLayout(self.horizontalLayout_profile, 6, 0, 1, 1)

        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        sizePolicy3 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy3)
        self.splitter.setOrientation(Qt.Horizontal)
        self.pushButton_addSoil = QPushButton(self.splitter)
        self.pushButton_addSoil.setObjectName(u"pushButton_addSoil")
        self.splitter.addWidget(self.pushButton_addSoil)
        self.pushButton_removeSoil = QPushButton(self.splitter)
        self.pushButton_removeSoil.setObjectName(u"pushButton_removeSoil")
        self.splitter.addWidget(self.pushButton_removeSoil)

        self.gridLayout_soil.addWidget(self.splitter, 2, 1, 1, 1)

        self.horizontalLayout_bedrock = QHBoxLayout()
        self.horizontalLayout_bedrock.setObjectName(u"horizontalLayout_bedrock")
        self.horizontalLayout_bedrock.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.label_35 = QLabel(self.centralwidget)
        self.label_35.setObjectName(u"label_35")
        sizePolicy.setHeightForWidth(self.label_35.sizePolicy().hasHeightForWidth())
        self.label_35.setSizePolicy(sizePolicy)
        self.label_35.setFont(font)

        self.horizontalLayout_bedrock.addWidget(self.label_35)

        self.lineEdit_bedWeight = QLineEdit(self.centralwidget)
        self.lineEdit_bedWeight.setObjectName(u"lineEdit_bedWeight")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.lineEdit_bedWeight.sizePolicy().hasHeightForWidth())
        self.lineEdit_bedWeight.setSizePolicy(sizePolicy4)

        self.horizontalLayout_bedrock.addWidget(self.lineEdit_bedWeight)

        self.label_bedUnits_2 = QLabel(self.centralwidget)
        self.label_bedUnits_2.setObjectName(u"label_bedUnits_2")
        sizePolicy.setHeightForWidth(self.label_bedUnits_2.sizePolicy().hasHeightForWidth())
        self.label_bedUnits_2.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setBold(False)
        self.label_bedUnits_2.setFont(font1)
        self.label_bedUnits_2.setTextFormat(Qt.RichText)

        self.horizontalLayout_bedrock.addWidget(self.label_bedUnits_2)

        self.lineEdit_bedVelocity = QLineEdit(self.centralwidget)
        self.lineEdit_bedVelocity.setObjectName(u"lineEdit_bedVelocity")
        sizePolicy4.setHeightForWidth(self.lineEdit_bedVelocity.sizePolicy().hasHeightForWidth())
        self.lineEdit_bedVelocity.setSizePolicy(sizePolicy4)

        self.horizontalLayout_bedrock.addWidget(self.lineEdit_bedVelocity)

        self.label_36 = QLabel(self.centralwidget)
        self.label_36.setObjectName(u"label_36")
        sizePolicy.setHeightForWidth(self.label_36.sizePolicy().hasHeightForWidth())
        self.label_36.setSizePolicy(sizePolicy)
        self.label_36.setFont(font1)

        self.horizontalLayout_bedrock.addWidget(self.label_36)

        self.lineEdit_bedDamping = QLineEdit(self.centralwidget)
        self.lineEdit_bedDamping.setObjectName(u"lineEdit_bedDamping")
        sizePolicy4.setHeightForWidth(self.lineEdit_bedDamping.sizePolicy().hasHeightForWidth())
        self.lineEdit_bedDamping.setSizePolicy(sizePolicy4)

        self.horizontalLayout_bedrock.addWidget(self.lineEdit_bedDamping)

        self.label_37 = QLabel(self.centralwidget)
        self.label_37.setObjectName(u"label_37")
        sizePolicy.setHeightForWidth(self.label_37.sizePolicy().hasHeightForWidth())
        self.label_37.setSizePolicy(sizePolicy)
        self.label_37.setFont(font1)

        self.horizontalLayout_bedrock.addWidget(self.label_37)


        self.gridLayout_soil.addLayout(self.horizontalLayout_bedrock, 8, 0, 1, 1)

        self.tableWidget_Soil = QTableWidget(self.centralwidget)
        if (self.tableWidget_Soil.columnCount() < 6):
            self.tableWidget_Soil.setColumnCount(6)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(0, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(1, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(2, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(3, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(4, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.tableWidget_Soil.setHorizontalHeaderItem(5, __qtablewidgetitem10)
        self.tableWidget_Soil.setObjectName(u"tableWidget_Soil")
        sizePolicy1.setHeightForWidth(self.tableWidget_Soil.sizePolicy().hasHeightForWidth())
        self.tableWidget_Soil.setSizePolicy(sizePolicy1)
        self.tableWidget_Soil.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.tableWidget_Soil.verticalHeader().setDefaultSectionSize(30)

        self.gridLayout_soil.addWidget(self.tableWidget_Soil, 2, 0, 1, 1)

        self.horizontalLayout_vel_damp = QHBoxLayout()
        self.horizontalLayout_vel_damp.setObjectName(u"horizontalLayout_vel_damp")

        self.gridLayout_soil.addLayout(self.horizontalLayout_vel_damp, 8, 1, 1, 1)

        self.horizontalLayout_create = QHBoxLayout()
        self.horizontalLayout_create.setObjectName(u"horizontalLayout_create")
        self.pushButton_drawProfile = QPushButton(self.centralwidget)
        self.pushButton_drawProfile.setObjectName(u"pushButton_drawProfile")

        self.horizontalLayout_create.addWidget(self.pushButton_drawProfile)

        self.pushButton_loadBatch = QPushButton(self.centralwidget)
        self.pushButton_loadBatch.setObjectName(u"pushButton_loadBatch")

        self.horizontalLayout_create.addWidget(self.pushButton_loadBatch)


        self.gridLayout_soil.addLayout(self.horizontalLayout_create, 5, 1, 1, 1)

        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        sizePolicy3.setHeightForWidth(self.splitter_2.sizePolicy().hasHeightForWidth())
        self.splitter_2.setSizePolicy(sizePolicy3)
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.pushButton_addProfile = QPushButton(self.splitter_2)
        self.pushButton_addProfile.setObjectName(u"pushButton_addProfile")
        self.splitter_2.addWidget(self.pushButton_addProfile)
        self.pushButton_removeProfile = QPushButton(self.splitter_2)
        self.pushButton_removeProfile.setObjectName(u"pushButton_removeProfile")
        self.splitter_2.addWidget(self.pushButton_removeProfile)

        self.gridLayout_soil.addWidget(self.splitter_2, 6, 1, 1, 1)

        self.gridLayout_soil.setColumnStretch(0, 3)

        self.gridLayout.addLayout(self.gridLayout_soil, 0, 1, 1, 1)

        self.verticalLayout_display_and_console = QVBoxLayout()
        self.verticalLayout_display_and_console.setObjectName(u"verticalLayout_display_and_console")
        self.verticalLayout_display_and_console.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.graphWidget = MplWidget(self.centralwidget)
        self.graphWidget.setObjectName(u"graphWidget")
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.graphWidget.sizePolicy().hasHeightForWidth())
        self.graphWidget.setSizePolicy(sizePolicy5)
        self.graphWidget.setMinimumSize(QSize(0, 0))

        self.verticalLayout_display_and_console.addWidget(self.graphWidget)

        self.plainTextEdit_overview = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_overview.setObjectName(u"plainTextEdit_overview")
        self.plainTextEdit_overview.setEnabled(False)
        sizePolicy5.setHeightForWidth(self.plainTextEdit_overview.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_overview.setSizePolicy(sizePolicy5)
        self.plainTextEdit_overview.setTextInteractionFlags(Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.verticalLayout_display_and_console.addWidget(self.plainTextEdit_overview)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.label_15 = QLabel(self.centralwidget)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setFont(font1)

        self.horizontalLayout_13.addWidget(self.label_15)

        self.lineEdit_maxFreq = QLineEdit(self.centralwidget)
        self.lineEdit_maxFreq.setObjectName(u"lineEdit_maxFreq")

        self.horizontalLayout_13.addWidget(self.lineEdit_maxFreq)

        self.label_16 = QLabel(self.centralwidget)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setFont(font1)

        self.horizontalLayout_13.addWidget(self.label_16)

        self.lineEdit_waveLength = QLineEdit(self.centralwidget)
        self.lineEdit_waveLength.setObjectName(u"lineEdit_waveLength")

        self.horizontalLayout_13.addWidget(self.lineEdit_waveLength)

        self.checkBox_autoDiscretize = QCheckBox(self.centralwidget)
        self.checkBox_autoDiscretize.setObjectName(u"checkBox_autoDiscretize")
        self.checkBox_autoDiscretize.setChecked(True)

        self.horizontalLayout_13.addWidget(self.checkBox_autoDiscretize)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_2)


        self.verticalLayout_display_and_console.addLayout(self.horizontalLayout_13)

        self.verticalLayout_display_and_console.setStretch(1, 2)
        self.verticalLayout_display_and_console.setStretch(2, 1)

        self.gridLayout.addLayout(self.verticalLayout_display_and_console, 0, 0, 1, 1)

        self.verticalLayout_input = QVBoxLayout()
        self.verticalLayout_input.setObjectName(u"verticalLayout_input")
        self.verticalLayout_input.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.comboBox_THorRVT = QComboBox(self.centralwidget)
        self.comboBox_THorRVT.addItem("")
        self.comboBox_THorRVT.addItem("")
        self.comboBox_THorRVT.setObjectName(u"comboBox_THorRVT")

        self.horizontalLayout.addWidget(self.comboBox_THorRVT)


        self.verticalLayout_input.addLayout(self.horizontalLayout)

        self.horizontalLayout_input_settings = QHBoxLayout()
        self.horizontalLayout_input_settings.setObjectName(u"horizontalLayout_input_settings")
        self.groupBox_TH = QGroupBox(self.centralwidget)
        self.groupBox_TH.setObjectName(u"groupBox_TH")
        self.groupBox_TH.setAutoFillBackground(True)
        self.verticalLayout = QVBoxLayout(self.groupBox_TH)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.graphWidget_TH = MplWidget(self.groupBox_TH)
        self.graphWidget_TH.setObjectName(u"graphWidget_TH")
        sizePolicy5.setHeightForWidth(self.graphWidget_TH.sizePolicy().hasHeightForWidth())
        self.graphWidget_TH.setSizePolicy(sizePolicy5)
        self.graphWidget_TH.setMinimumSize(QSize(0, 0))

        self.verticalLayout.addWidget(self.graphWidget_TH)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_7 = QLabel(self.groupBox_TH)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font1)

        self.horizontalLayout_2.addWidget(self.label_7)

        self.lineEdit_FS = QLineEdit(self.groupBox_TH)
        self.lineEdit_FS.setObjectName(u"lineEdit_FS")

        self.horizontalLayout_2.addWidget(self.lineEdit_FS)

        self.label_8 = QLabel(self.groupBox_TH)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font1)

        self.horizontalLayout_2.addWidget(self.label_8)

        self.comboBox_Units = QComboBox(self.groupBox_TH)
        self.comboBox_Units.addItem("")
        self.comboBox_Units.addItem("")
        self.comboBox_Units.addItem("")
        self.comboBox_Units.setObjectName(u"comboBox_Units")

        self.horizontalLayout_2.addWidget(self.comboBox_Units)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_6)

        self.pushButton_loadTH = QPushButton(self.groupBox_TH)
        self.pushButton_loadTH.setObjectName(u"pushButton_loadTH")

        self.horizontalLayout_2.addWidget(self.pushButton_loadTH)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_14 = QLabel(self.groupBox_TH)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setFont(font1)

        self.horizontalLayout_3.addWidget(self.label_14)

        self.comboBox_eventList = QComboBox(self.groupBox_TH)
        self.comboBox_eventList.setObjectName(u"comboBox_eventList")
        self.comboBox_eventList.setEnabled(False)

        self.horizontalLayout_3.addWidget(self.comboBox_eventList)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)


        self.verticalLayout.addLayout(self.horizontalLayout_3)


        self.horizontalLayout_input_settings.addWidget(self.groupBox_TH)

        self.groupBox_RVT = QGroupBox(self.centralwidget)
        self.groupBox_RVT.setObjectName(u"groupBox_RVT")
        self.groupBox_RVT.setAutoFillBackground(True)
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_RVT)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.graphWidget_Spectrum = MplWidget(self.groupBox_RVT)
        self.graphWidget_Spectrum.setObjectName(u"graphWidget_Spectrum")
        sizePolicy5.setHeightForWidth(self.graphWidget_Spectrum.sizePolicy().hasHeightForWidth())
        self.graphWidget_Spectrum.setSizePolicy(sizePolicy5)
        self.graphWidget_Spectrum.setMinimumSize(QSize(0, 0))

        self.verticalLayout_6.addWidget(self.graphWidget_Spectrum)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_31 = QLabel(self.groupBox_RVT)
        self.label_31.setObjectName(u"label_31")

        self.horizontalLayout_10.addWidget(self.label_31)

        self.lineEdit_duration = QLineEdit(self.groupBox_RVT)
        self.lineEdit_duration.setObjectName(u"lineEdit_duration")

        self.horizontalLayout_10.addWidget(self.lineEdit_duration)

        self.label_32 = QLabel(self.groupBox_RVT)
        self.label_32.setObjectName(u"label_32")

        self.horizontalLayout_10.addWidget(self.label_32)

        self.lineEdit_damping = QLineEdit(self.groupBox_RVT)
        self.lineEdit_damping.setObjectName(u"lineEdit_damping")

        self.horizontalLayout_10.addWidget(self.lineEdit_damping)

        self.label_33 = QLabel(self.groupBox_RVT)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setFont(font1)

        self.horizontalLayout_10.addWidget(self.label_33)

        self.comboBox_SpectraUnits = QComboBox(self.groupBox_RVT)
        self.comboBox_SpectraUnits.addItem("")
        self.comboBox_SpectraUnits.addItem("")
        self.comboBox_SpectraUnits.addItem("")
        self.comboBox_SpectraUnits.setObjectName(u"comboBox_SpectraUnits")

        self.horizontalLayout_10.addWidget(self.comboBox_SpectraUnits)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_7)

        self.pushButton_loadSpectra = QPushButton(self.groupBox_RVT)
        self.pushButton_loadSpectra.setObjectName(u"pushButton_loadSpectra")

        self.horizontalLayout_10.addWidget(self.pushButton_loadSpectra)


        self.verticalLayout_6.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_34 = QLabel(self.groupBox_RVT)
        self.label_34.setObjectName(u"label_34")

        self.horizontalLayout_11.addWidget(self.label_34)

        self.comboBox_spectraList = QComboBox(self.groupBox_RVT)
        self.comboBox_spectraList.setObjectName(u"comboBox_spectraList")
        self.comboBox_spectraList.setEnabled(False)

        self.horizontalLayout_11.addWidget(self.comboBox_spectraList)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_5)

        self.comboBox_showWhat = QComboBox(self.groupBox_RVT)
        self.comboBox_showWhat.addItem("")
        self.comboBox_showWhat.addItem("")
        self.comboBox_showWhat.setObjectName(u"comboBox_showWhat")

        self.horizontalLayout_11.addWidget(self.comboBox_showWhat)

        self.checkBox_xlog = QCheckBox(self.groupBox_RVT)
        self.checkBox_xlog.setObjectName(u"checkBox_xlog")

        self.horizontalLayout_11.addWidget(self.checkBox_xlog)

        self.checkBox_ylog = QCheckBox(self.groupBox_RVT)
        self.checkBox_ylog.setObjectName(u"checkBox_ylog")

        self.horizontalLayout_11.addWidget(self.checkBox_ylog)


        self.verticalLayout_6.addLayout(self.horizontalLayout_11)

        self.verticalLayout_6.setStretch(0, 4)
        self.verticalLayout_6.setStretch(1, 1)
        self.verticalLayout_6.setStretch(2, 1)

        self.horizontalLayout_input_settings.addWidget(self.groupBox_RVT)


        self.verticalLayout_input.addLayout(self.horizontalLayout_input_settings)


        self.gridLayout.addLayout(self.verticalLayout_input, 2, 1, 1, 1)

        self.verticalLayout_output_and_settings = QVBoxLayout()
        self.verticalLayout_output_and_settings.setObjectName(u"verticalLayout_output_and_settings")
        self.verticalLayout_output_and_settings.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.verticalLayout_analysis_and_bricks = QVBoxLayout()
        self.verticalLayout_analysis_and_bricks.setObjectName(u"verticalLayout_analysis_and_bricks")
        self.verticalLayout_analysis_and_bricks.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout_analysis_type = QHBoxLayout()
        self.horizontalLayout_analysis_type.setObjectName(u"horizontalLayout_analysis_type")
        self.comboBox_analysisType = QComboBox(self.centralwidget)
        self.comboBox_analysisType.addItem("")
        self.comboBox_analysisType.addItem("")
        self.comboBox_analysisType.addItem("")
        self.comboBox_analysisType.setObjectName(u"comboBox_analysisType")

        self.horizontalLayout_analysis_type.addWidget(self.comboBox_analysisType)

        self.checkBox_updatePlots = QCheckBox(self.centralwidget)
        self.checkBox_updatePlots.setObjectName(u"checkBox_updatePlots")
        self.checkBox_updatePlots.setEnabled(False)

        self.horizontalLayout_analysis_type.addWidget(self.checkBox_updatePlots)

        self.checkBox_multithread = QCheckBox(self.centralwidget)
        self.checkBox_multithread.setObjectName(u"checkBox_multithread")
        self.checkBox_multithread.setEnabled(False)

        self.horizontalLayout_analysis_type.addWidget(self.checkBox_multithread)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_analysis_type.addItem(self.horizontalSpacer_8)


        self.verticalLayout_analysis_and_bricks.addLayout(self.horizontalLayout_analysis_type)

        self.horizontalLayout_bricks = QHBoxLayout()
        self.horizontalLayout_bricks.setObjectName(u"horizontalLayout_bricks")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label_18 = QLabel(self.centralwidget)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setFont(font1)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_18)

        self.lineEdit_brickSize = QLineEdit(self.centralwidget)
        self.lineEdit_brickSize.setObjectName(u"lineEdit_brickSize")
        self.lineEdit_brickSize.setEnabled(False)
        self.lineEdit_brickSize.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.lineEdit_brickSize)

        self.label_25 = QLabel(self.centralwidget)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setFont(font1)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_25)

        self.lineEdit_bedDepth = QLineEdit(self.centralwidget)
        self.lineEdit_bedDepth.setObjectName(u"lineEdit_bedDepth")
        self.lineEdit_bedDepth.setEnabled(False)
        self.lineEdit_bedDepth.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.lineEdit_bedDepth)


        self.horizontalLayout_bricks.addLayout(self.formLayout)

        self.formLayout_generic_options = QFormLayout()
        self.formLayout_generic_options.setObjectName(u"formLayout_generic_options")
        self.label_11 = QLabel(self.centralwidget)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font1)

        self.formLayout_generic_options.setWidget(0, QFormLayout.LabelRole, self.label_11)

        self.lineEdit_strainRatio = QLineEdit(self.centralwidget)
        self.lineEdit_strainRatio.setObjectName(u"lineEdit_strainRatio")

        self.formLayout_generic_options.setWidget(0, QFormLayout.FieldRole, self.lineEdit_strainRatio)

        self.label_12 = QLabel(self.centralwidget)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font1)

        self.formLayout_generic_options.setWidget(1, QFormLayout.LabelRole, self.label_12)

        self.lineEdit_maxIter = QLineEdit(self.centralwidget)
        self.lineEdit_maxIter.setObjectName(u"lineEdit_maxIter")

        self.formLayout_generic_options.setWidget(1, QFormLayout.FieldRole, self.lineEdit_maxIter)

        self.label_13 = QLabel(self.centralwidget)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setFont(font1)

        self.formLayout_generic_options.setWidget(2, QFormLayout.LabelRole, self.label_13)

        self.lineEdit_maxTol = QLineEdit(self.centralwidget)
        self.lineEdit_maxTol.setObjectName(u"lineEdit_maxTol")

        self.formLayout_generic_options.setWidget(2, QFormLayout.FieldRole, self.lineEdit_maxTol)


        self.horizontalLayout_bricks.addLayout(self.formLayout_generic_options)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_bricks.addItem(self.horizontalSpacer_9)


        self.verticalLayout_analysis_and_bricks.addLayout(self.horizontalLayout_bricks)


        self.verticalLayout_output_and_settings.addLayout(self.verticalLayout_analysis_and_bricks)

        self.groupBox_output = QGroupBox(self.centralwidget)
        self.groupBox_output.setObjectName(u"groupBox_output")
        self.gridLayout_2 = QGridLayout(self.groupBox_output)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.lineEdit_strainDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_strainDepth.setObjectName(u"lineEdit_strainDepth")

        self.gridLayout_2.addWidget(self.lineEdit_strainDepth, 2, 1, 1, 1)

        self.label_17 = QLabel(self.groupBox_output)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setEnabled(False)
        self.label_17.setFont(font1)

        self.gridLayout_2.addWidget(self.label_17, 3, 2, 1, 1)

        self.lineEdit_briefDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_briefDepth.setObjectName(u"lineEdit_briefDepth")
        self.lineEdit_briefDepth.setEnabled(False)

        self.gridLayout_2.addWidget(self.lineEdit_briefDepth, 3, 1, 1, 1)

        self.lineEdit_FourierDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_FourierDepth.setObjectName(u"lineEdit_FourierDepth")
        self.lineEdit_FourierDepth.setEnabled(True)

        self.gridLayout_2.addWidget(self.lineEdit_FourierDepth, 4, 1, 1, 1)

        self.checkBox_Fourier = QCheckBox(self.groupBox_output)
        self.checkBox_Fourier.setObjectName(u"checkBox_Fourier")
        self.checkBox_Fourier.setEnabled(True)
        self.checkBox_Fourier.setChecked(False)

        self.gridLayout_2.addWidget(self.checkBox_Fourier, 4, 0, 1, 1)

        self.lineEdit_accDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_accDepth.setObjectName(u"lineEdit_accDepth")

        self.gridLayout_2.addWidget(self.lineEdit_accDepth, 1, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 3, 1, 1)

        self.label_5 = QLabel(self.groupBox_output)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font1)

        self.gridLayout_2.addWidget(self.label_5, 1, 2, 1, 1)

        self.checkBox_outAcc = QCheckBox(self.groupBox_output)
        self.checkBox_outAcc.setObjectName(u"checkBox_outAcc")

        self.gridLayout_2.addWidget(self.checkBox_outAcc, 1, 0, 1, 1)

        self.checkBox_outRS = QCheckBox(self.groupBox_output)
        self.checkBox_outRS.setObjectName(u"checkBox_outRS")
        self.checkBox_outRS.setChecked(True)

        self.gridLayout_2.addWidget(self.checkBox_outRS, 0, 0, 1, 1)

        self.label_24 = QLabel(self.groupBox_output)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setEnabled(False)
        self.label_24.setFont(font1)

        self.gridLayout_2.addWidget(self.label_24, 4, 2, 1, 1)

        self.checkBox_outStrain = QCheckBox(self.groupBox_output)
        self.checkBox_outStrain.setObjectName(u"checkBox_outStrain")

        self.gridLayout_2.addWidget(self.checkBox_outStrain, 2, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox_output)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font1)

        self.gridLayout_2.addWidget(self.label_4, 2, 2, 1, 1)

        self.lineEdit_RSDepth = QLineEdit(self.groupBox_output)
        self.lineEdit_RSDepth.setObjectName(u"lineEdit_RSDepth")

        self.gridLayout_2.addWidget(self.lineEdit_RSDepth, 0, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox_output)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font1)

        self.gridLayout_2.addWidget(self.label_3, 0, 2, 1, 1)

        self.checkBox_outBrief = QCheckBox(self.groupBox_output)
        self.checkBox_outBrief.setObjectName(u"checkBox_outBrief")
        self.checkBox_outBrief.setEnabled(True)
        self.checkBox_outBrief.setChecked(True)

        self.gridLayout_2.addWidget(self.checkBox_outBrief, 3, 0, 1, 1)


        self.verticalLayout_output_and_settings.addWidget(self.groupBox_output)

        self.pushButton_run = QPushButton(self.centralwidget)
        self.pushButton_run.setObjectName(u"pushButton_run")

        self.verticalLayout_output_and_settings.addWidget(self.pushButton_run)


        self.gridLayout.addLayout(self.verticalLayout_output_and_settings, 2, 0, 1, 1)

        self.gridLayout.setRowStretch(0, 5)
        self.gridLayout.setRowStretch(2, 5)
        self.gridLayout.setRowMinimumHeight(0, 50)
        self.gridLayout.setRowMinimumHeight(2, 50)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1551, 37))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.menuTools = QMenu(self.menubar)
        self.menuTools.setObjectName(u"menuTools")
        self.menuMerge_and_make_stats = QMenu(self.menuTools)
        self.menuMerge_and_make_stats.setObjectName(u"menuMerge_and_make_stats")
        self.menuSupport = QMenu(self.menubar)
        self.menuSupport.setObjectName(u"menuSupport")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuSupport.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.actionAbout)
        self.menuTools.addAction(self.actionGenerateStochastic)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.actionGeneratePermutated)
        self.menuTools.addAction(self.actionGenerate_NTC)
        self.menuTools.addAction(self.actionGenerate_UHS_spectra)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.menuMerge_and_make_stats.menuAction())
        self.menuMerge_and_make_stats.addAction(self.actionGenerate_only_master)
        self.menuMerge_and_make_stats.addAction(self.actionGenerate_master_and_sub)
        self.menuSupport.addAction(self.actionLoadspectra)
        self.menuSupport.addAction(self.actionRun_analysis)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"NC92-Soil", None))
        self.actionAbout.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.actionGenerateStochastic.setText(QCoreApplication.translate("MainWindow", u"Generate stochastic profiles", None))
        self.actionGeneratePermutated.setText(QCoreApplication.translate("MainWindow", u"Generate permutated profiles (beta)", None))
        self.actionGenerate_NTC.setText(QCoreApplication.translate("MainWindow", u"Generate NTC spectra", None))
        self.actionGenerate_master_and_sub.setText(QCoreApplication.translate("MainWindow", u"Generate master report with subreport", None))
        self.actionGenerate_master_report.setText(QCoreApplication.translate("MainWindow", u"Generate master report", None))
        self.actionGenerate_only_master.setText(QCoreApplication.translate("MainWindow", u"Generate master report", None))
        self.actionLoadspectra.setText(QCoreApplication.translate("MainWindow", u"Load target spectra", None))
        self.actionRun_analysis.setText(QCoreApplication.translate("MainWindow", u"Run batch analysis", None))
        self.actionGenerate_UHS_spectra.setText(QCoreApplication.translate("MainWindow", u"Generate UHS spectra", None))
        self.label_profileProp.setText(QCoreApplication.translate("MainWindow", u"Profile", None))
        self.label_SoilProp.setText(QCoreApplication.translate("MainWindow", u"Soil properties", None))
        ___qtablewidgetitem = self.tableWidget_Profile.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Depth [m]", None));
        ___qtablewidgetitem1 = self.tableWidget_Profile.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Thickness [m]", None));
        ___qtablewidgetitem2 = self.tableWidget_Profile.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Soil type", None));
        ___qtablewidgetitem3 = self.tableWidget_Permutations.horizontalHeaderItem(0)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"Soil type", None));
        ___qtablewidgetitem4 = self.tableWidget_Permutations.horizontalHeaderItem(1)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Percentage", None));
        self.pushButton_addSoil.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.pushButton_removeSoil.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"Bedrock", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedWeight.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock unit weight", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedWeight.setText(QCoreApplication.translate("MainWindow", u"22", None))
        self.label_bedUnits_2.setText(QCoreApplication.translate("MainWindow", u"KN/m<sup>3</sup>", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedVelocity.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock velocity", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedVelocity.setText(QCoreApplication.translate("MainWindow", u"800", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"m/s", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedDamping.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock damping", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedDamping.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"%", None))
        ___qtablewidgetitem5 = self.tableWidget_Soil.horizontalHeaderItem(0)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"Name", None));
        ___qtablewidgetitem6 = self.tableWidget_Soil.horizontalHeaderItem(1)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"Unit Weight", None));
        ___qtablewidgetitem7 = self.tableWidget_Soil.horizontalHeaderItem(2)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"From [m]", None));
        ___qtablewidgetitem8 = self.tableWidget_Soil.horizontalHeaderItem(3)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("MainWindow", u"To [m]", None));
        ___qtablewidgetitem9 = self.tableWidget_Soil.horizontalHeaderItem(4)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("MainWindow", u"Vs [m/s]", None));
        ___qtablewidgetitem10 = self.tableWidget_Soil.horizontalHeaderItem(5)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("MainWindow", u"Degradation curve", None));
        self.pushButton_drawProfile.setText(QCoreApplication.translate("MainWindow", u"Create profile", None))
        self.pushButton_loadBatch.setText(QCoreApplication.translate("MainWindow", u"Load batch file", None))
        self.pushButton_addProfile.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.pushButton_removeProfile.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Max frequency [Hz]", None))
        self.lineEdit_maxFreq.setText(QCoreApplication.translate("MainWindow", u"20", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Wavelength ratio", None))
        self.lineEdit_waveLength.setText(QCoreApplication.translate("MainWindow", u"0.2", None))
        self.checkBox_autoDiscretize.setText(QCoreApplication.translate("MainWindow", u"Auto discretize", None))
        self.comboBox_THorRVT.setItemText(0, QCoreApplication.translate("MainWindow", u"Time History", None))
        self.comboBox_THorRVT.setItemText(1, QCoreApplication.translate("MainWindow", u"RVT", None))

        self.groupBox_TH.setTitle(QCoreApplication.translate("MainWindow", u"Time history", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"FS [Hz]", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_FS.setToolTip(QCoreApplication.translate("MainWindow", u"If specified in input file, sample frequency will be automatically imported from input file", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_FS.setText(QCoreApplication.translate("MainWindow", u"200", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Original units", None))
        self.comboBox_Units.setItemText(0, QCoreApplication.translate("MainWindow", u"g", None))
        self.comboBox_Units.setItemText(1, QCoreApplication.translate("MainWindow", u"cm/s^2", None))
        self.comboBox_Units.setItemText(2, QCoreApplication.translate("MainWindow", u"m/s^2", None))

        self.pushButton_loadTH.setText(QCoreApplication.translate("MainWindow", u"Load time history", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"File name", None))
        self.groupBox_RVT.setTitle(QCoreApplication.translate("MainWindow", u"RVT", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Duration [s]", None))
        self.lineEdit_duration.setText(QCoreApplication.translate("MainWindow", u"5", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"Damping", None))
        self.lineEdit_damping.setText(QCoreApplication.translate("MainWindow", u"0.05", None))
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"Original units", None))
        self.comboBox_SpectraUnits.setItemText(0, QCoreApplication.translate("MainWindow", u"g", None))
        self.comboBox_SpectraUnits.setItemText(1, QCoreApplication.translate("MainWindow", u"cm/s^2", None))
        self.comboBox_SpectraUnits.setItemText(2, QCoreApplication.translate("MainWindow", u"m/s^2", None))

        self.pushButton_loadSpectra.setText(QCoreApplication.translate("MainWindow", u"Load target spectra", None))
        self.label_34.setText(QCoreApplication.translate("MainWindow", u"File name", None))
        self.comboBox_showWhat.setItemText(0, QCoreApplication.translate("MainWindow", u"Show RS", None))
        self.comboBox_showWhat.setItemText(1, QCoreApplication.translate("MainWindow", u"Show FAS", None))

        self.checkBox_xlog.setText(QCoreApplication.translate("MainWindow", u"X log scale", None))
        self.checkBox_ylog.setText(QCoreApplication.translate("MainWindow", u"Y log scale", None))
        self.comboBox_analysisType.setItemText(0, QCoreApplication.translate("MainWindow", u"Regular analysis", None))
        self.comboBox_analysisType.setItemText(1, QCoreApplication.translate("MainWindow", u"Permutations", None))
        self.comboBox_analysisType.setItemText(2, QCoreApplication.translate("MainWindow", u"Batch analysis", None))

        self.checkBox_updatePlots.setText(QCoreApplication.translate("MainWindow", u"Update plot (batch only)", None))
        self.checkBox_multithread.setText(QCoreApplication.translate("MainWindow", u"Multithread (beta)", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Brick size [m]", None))
        self.lineEdit_brickSize.setText(QCoreApplication.translate("MainWindow", u"3", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Bedrock depth [m]", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_bedDepth.setToolTip(QCoreApplication.translate("MainWindow", u"Bedrock unit weight", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_bedDepth.setText(QCoreApplication.translate("MainWindow", u"30", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Effective strain ratio", None))
        self.lineEdit_strainRatio.setText(QCoreApplication.translate("MainWindow", u"0.65", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Max iterations", None))
        self.lineEdit_maxIter.setText(QCoreApplication.translate("MainWindow", u"10", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Error tolerance [%]", None))
        self.lineEdit_maxTol.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.groupBox_output.setTitle(QCoreApplication.translate("MainWindow", u"Output", None))
        self.lineEdit_strainDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.lineEdit_briefDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_FourierDepth.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Choose the depth at which the Fourier spectrum must be calculated.</p><p>Enabling this option, also the Fourier spectrum at bedrock depth (within and outcrop) will be exported by default</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.lineEdit_FourierDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.checkBox_Fourier.setText(QCoreApplication.translate("MainWindow", u"Transfer functions", None))
        self.lineEdit_accDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_outAcc.setText(QCoreApplication.translate("MainWindow", u"Acceleration", None))
        self.checkBox_outRS.setText(QCoreApplication.translate("MainWindow", u"Response spectrum", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_outStrain.setText(QCoreApplication.translate("MainWindow", u"Strain time history", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.lineEdit_RSDepth.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"m", None))
        self.checkBox_outBrief.setText(QCoreApplication.translate("MainWindow", u"PGA, PGV, FA", None))
        self.pushButton_run.setText(QCoreApplication.translate("MainWindow", u"Run analysis", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"?", None))
        self.menuTools.setTitle(QCoreApplication.translate("MainWindow", u"Tools", None))
        self.menuMerge_and_make_stats.setTitle(QCoreApplication.translate("MainWindow", u"Merge and make stats", None))
        self.menuSupport.setTitle(QCoreApplication.translate("MainWindow", u"Support", None))
    # retranslateUi

