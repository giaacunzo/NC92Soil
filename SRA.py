from warnings import filterwarnings

filterwarnings('ignore', category=UserWarning)
filterwarnings('ignore', category=RuntimeWarning)

from PySide6.QtWidgets import *
from PySide6 import QtCore
from SRAmainGUI import Ui_MainWindow
from SRAClasses import BatchAnalyzer, StochasticAnalyzer, NTCCalculator, ClusterToMOPS
import numpy as np
import sys
import platform
import SRALibrary as SRALib
import pandas as pd
import os
from time import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = ''

from pygame import mixer, error as pygameexception


NCVERSION = 0.93


def aboutMessage():
    Messaggio = QMessageBox()
    Messaggio.setText(f"NC92-Soil\nversion {NCVERSION} beta\n"
                      "\nCNR IGAG")
    Messaggio.setWindowTitle(f"NC92-Soil rev {NCVERSION}")
    Messaggio.setText("Il software NC92Soil è stato realizzato dal CNR IGAG nell'ambito del contratto "
                      "concernente l’affidamento di servizi per il “Programma per il supporto al "
                      "rafforzamento della governance in materia di riduzione del rischio sismico e vulcanico "
                      "ai fini di protezione civile nell’ambito del Pon Governance e Capacità Istituzionale 2014-2020"
                      "\"CIG 6980737E65 – CUP J59G16000160006")
    Messaggio.setWindowTitle("NC92Soil 1.0")
    Messaggio.setFixedWidth(1200)

    try:
        mixer.init()
        mixer.music.load('about.mp3')
        mixer.music.play()
        mixer.music.set_volume(0.3)
    # except pygameexception:
    #     pass
    except:
        pass

    Messaggio.exec_()
    try:
        mixer.music.stop()
    except:
        pass


# noinspection PyCallByClass
class SRAApp(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        # Inizializzazione attributi principali
        self.curveDB = dict()
        self.userModified = True
        self.inputMotion = dict()
        self.batchObject = list()

        self.setupUi(self)
        self.assignWidgets()
        self.setDefault()
        self.dialogOptions = self.get_system_options()
        # self.dialogOptions = QFileDialog.Options()

        # For testing
        print('NC92Soil GUI correctly loaded')
        App.processEvents()

        self.show()
        aboutMessage()

    def setDefault(self):
        # Formattazione delle tabelle
        self.tableWidget_Soil.resizeColumnsToContents()
        self.tableWidget_Soil.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_Profile.resizeColumnsToContents()
        self.tableWidget_Profile.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_Permutations.setVisible(False)
        self.pushButton_loadBatch.setVisible(False)
        self.changeInputPanel()

        # Caricamento database curve di decadimento
        try:
            self.curveDB = SRALib.degradationCurves('CurveDB.xlsx')
        except FileNotFoundError:
            msg = "File CurveDB.xlsx has not been found in the program folder"
            QMessageBox.critical(QMessageBox(), "Check database", msg)
            sys.exit(-1)

    def assignWidgets(self):
        self.pushButton_addSoil.clicked.connect(self.addRow)
        self.pushButton_addProfile.clicked.connect(self.addRow)
        self.pushButton_removeSoil.clicked.connect(self.removeRow)
        self.pushButton_removeProfile.clicked.connect(self.removeRow)
        self.tableWidget_Profile.cellChanged.connect(self.profileChanged)
        self.pushButton_drawProfile.clicked.connect(self.makeProfile)
        self.pushButton_loadTH.clicked.connect(self.loadTH)
        self.pushButton_run.clicked.connect(self.runAnalysis)
        self.comboBox_eventList.currentIndexChanged.connect(self.viewTH)
        self.comboBox_spectraList.currentIndexChanged.connect(self.viewSpectra)
        self.comboBox_analysisType.currentIndexChanged.connect(self.changeAnalysis)
        self.comboBox_showWhat.currentIndexChanged.connect(self.viewSpectra)
        self.comboBox_THorRVT.currentIndexChanged.connect(self.changeInputPanel)
        self.pushButton_loadSpectra.clicked.connect(self.loadSpectra)
        self.checkBox_xlog.stateChanged.connect(self.viewSpectra)
        self.checkBox_ylog.stateChanged.connect(self.viewSpectra)
        self.lineEdit_RSDepth.textChanged.connect(self.updateOutputInfo)
        self.checkBox_outBrief.stateChanged.connect(self.updateOutputInfo)
        self.checkBox_outRS.stateChanged.connect(self.updateOutputInfo)
        self.actionAbout.triggered.connect(aboutMessage)
        self.pushButton_loadBatch.clicked.connect(self.loadBatch)
        self.actionGenerateStochastic.triggered.connect(self.loadStochastic)
        self.actionGeneratePermutated.triggered.connect(self.generatePermutatedProfiles)
        self.actionGenerate_NTC.triggered.connect(self.generateNTC)
        self.actionGenerate_only_master.triggered.connect(self.makeStats)
        self.actionGenerate_master_and_sub.triggered.connect(self.makeStats)
        self.actionLoadspectra.triggered.connect(self.loadSpectra)
        self.actionRun_analysis.triggered.connect(self.runBatch)

    def updateOutputInfo(self):
        if self.sender() is self.lineEdit_RSDepth:
            self.lineEdit_briefDepth.setText(self.sender().text())
        # elif self.sender() is self.checkBox_outBrief:
        #     if self.sender().isChecked():
        #         self.checkBox_outRS.setChecked(True)
        # elif self.sender() is self.checkBox_outRS:
        #     if not self.sender().isChecked():
        #         self.checkBox_outBrief.setChecked(False)

    @staticmethod
    def get_system_options():
        if platform.system() == 'Darwin':
            return QFileDialog.DontUseNativeDialog
        else:
            return QFileDialog.Options()

    def addRow(self, sender):
        if not sender:
            senderName = self.sender().objectName()
        else:
            senderName = sender

        if senderName == 'pushButton_addProfile' and self.comboBox_analysisType.currentText() == 'Permutations':
            tableName = 'tableWidget_Permutations'
        else:
            tableName = 'tableWidget_' + senderName.split('_add')[-1]
        currentTable = getattr(self, tableName)
        currentRow = currentTable.rowCount()
        currentTable.insertRow(currentRow)

        self.userModified = False

        if tableName.split('_')[-1] == 'Soil':
            curveComboBox = QComboBox(self)
            curveComboBox.addItems(self.curveDB.keys())
            currentTable.setCellWidget(currentRow, 5, curveComboBox)

        elif tableName.split('_')[-1] == 'Profile':
            if self.comboBox_analysisType.currentText() == 'Regular analysis':
                if currentRow == 0:  # Inserimento prima riga
                    currentTable.setItem(currentRow, 0, QTableWidgetItem('0.00'))
                    currentTable.setItem(currentRow, 1, QTableWidgetItem('0.00'))
                else:
                    currentDepth = float(currentTable.item(currentRow - 1, 0).text()) + \
                                   float(currentTable.item(currentRow - 1, 1).text())
                    currentTable.setItem(currentRow, 0, QTableWidgetItem(str(currentDepth)))
                    currentTable.setItem(currentRow, 1, QTableWidgetItem('0.00'))
                currentTable.item(currentRow, 0).setFlags(QtCore.Qt.ItemIsEnabled)

        self.userModified = True

    def removeRow(self):
        senderName = self.sender().objectName()
        tableName = 'tableWidget_' + senderName.split('_remove')[-1]
        selectedRow = getattr(self, tableName).selectionModel().selectedRows()

        if len(selectedRow) > 0:
            getattr(self, tableName).model().removeRow(selectedRow[0].row())

    def makeProfile(self):
        if not getattr(self.graphWidget, 'axes', False):
            self.graphWidget.axes = self.graphWidget.figure.add_subplot(111)

        currentAnalysis = self.comboBox_analysisType.currentText()
        brickSize = float(self.lineEdit_brickSize.text())
        bedrockDepth = float(self.lineEdit_bedDepth.text())
        if currentAnalysis == 'Permutations':
            profileList = SRALib.makeBricks(self.tableWidget_Permutations, brickSize, bedrockDepth)
            totalPermutations, percString, uniqueBricks = SRALib.calcPermutations(profileList)

            # Writing informations in the overview console
            messageString = "Total bricks: {} - Minimum brick size: {}m - Brick types: {}\nSoil percentage -> {}\n" \
                            "Total permutations: {}".format(len(profileList), brickSize, uniqueBricks, percString,
                                                            "{:,}".format(int(totalPermutations)).replace(',', "'")
                                                            )
            self.plainTextEdit_overview.setPlainText(messageString)
        else:
            profileList = SRALib.table2list(self.tableWidget_Soil, self.tableWidget_Profile,
                                            self.tableWidget_Permutations)[1]

        SRALib.drawProfile(profileList, self.graphWidget.axes)
        self.graphWidget.draw()

    def profileChanged(self):
        if not self.userModified:
            return None
        rowNumber = self.tableWidget_Profile.rowCount()

        self.userModified = False
        for riga in range(1, rowNumber):
            currentDepth = float(self.tableWidget_Profile.item(riga - 1, 0).text()) + \
                           float(self.tableWidget_Profile.item(riga - 1, 1).text())
            self.tableWidget_Profile.setItem(riga, 0, QTableWidgetItem(str(currentDepth)))
        self.userModified = True

    def viewTH(self):
        try:
            currentData = self.inputMotion[self.comboBox_eventList.currentText()]
        except KeyError:  # Combobox has been cleaned
            return

        if not getattr(self.graphWidget_TH, 'axes', False):
            self.graphWidget_TH.axes = self.graphWidget_TH.figure.add_subplot(111)

        SRALib.drawTH(currentData[0], self.graphWidget_TH.axes)
        self.graphWidget_TH.draw()

        self.lineEdit_FS.setText(str(1 / currentData[0].time_step))
        self.comboBox_Units.setCurrentText(currentData[-1])

    def viewSpectra(self):
        try:
            currentData = self.inputMotion[self.comboBox_spectraList.currentText()]
        except KeyError:  # Combobox has been cleaned
            return

        if not getattr(self.graphWidget_Spectrum, 'axes', False):
            self.graphWidget_Spectrum.axes = self.graphWidget_Spectrum.figure.add_subplot(111)

        xlog = self.checkBox_xlog.isChecked()
        ylog = self.checkBox_ylog.isChecked()

        whatToShow = self.comboBox_showWhat.currentText()

        if whatToShow == 'Show RS':
            # Passing the original spectrum and the event ID to drawer (index 1 and 2, respectively)
            SRALib.drawSpectrum(currentData[1], currentData[2], self.graphWidget_Spectrum.axes,
                                xlog=xlog, ylog=ylog)
        else:
            # Passing the computed input motion and the event ID to drawer (index 0 and 2, respectively)
            SRALib.drawFAS(currentData[0], currentData[2], self.graphWidget_Spectrum.axes,
                           xlog=xlog, ylog=ylog)

        self.graphWidget_Spectrum.draw()

        self.lineEdit_duration.setText(str(currentData[0].duration))
        self.lineEdit_damping.setText(str(currentData[3]))
        self.comboBox_Units.setCurrentText(currentData[4])

    def changeAnalysis(self):
        buttonList = ['pushButton_addProfile', 'pushButton_removeProfile', 'pushButton_addSoil',
                      'pushButton_removeSoil']
        currentData = self.comboBox_analysisType.currentText()
        if currentData == 'Permutations':
            self.lineEdit_brickSize.setEnabled(True)
            self.plainTextEdit_overview.setEnabled(True)
            self.lineEdit_bedDepth.setEnabled(True)
            self.tableWidget_Permutations.show()
            self.tableWidget_Profile.hide()
            self.label_SoilProp.setText('Soil properties')
            self.label_profileProp.setText('Soil percentage')
            self.tableWidget_Soil.setEnabled(True)
            self.tableWidget_Permutations.setEnabled(True)
            self.tableWidget_Profile.setEnabled(True)
            self.pushButton_drawProfile.show()
            self.pushButton_loadBatch.hide()
            self.checkBox_updatePlots.setEnabled(False)

            for element in buttonList:
                getattr(self, element).setEnabled(True)

            # Switching signal for run button to manual analysis
            self.pushButton_run.clicked.disconnect()
            self.pushButton_run.clicked.connect(self.runAnalysis)

            # Deactivating support button
            self.actionRun_analysis.setEnabled(False)

        elif currentData == 'Regular analysis':
            self.lineEdit_brickSize.setEnabled(False)
            self.plainTextEdit_overview.setEnabled(False)
            self.lineEdit_bedDepth.setEnabled(False)
            self.tableWidget_Permutations.hide()
            self.tableWidget_Profile.show()
            self.label_SoilProp.setText('Soil properties')
            self.label_profileProp.setText('Profile')
            self.tableWidget_Soil.setEnabled(True)
            self.tableWidget_Profile.setEnabled(True)
            self.tableWidget_Permutations.setEnabled(True)
            self.pushButton_drawProfile.show()
            self.pushButton_loadBatch.hide()
            self.checkBox_updatePlots.setEnabled(False)

            for element in buttonList:
                getattr(self, element).setEnabled(True)

            # Switching signal for run button to manual analysis
            self.pushButton_run.clicked.disconnect()
            self.pushButton_run.clicked.connect(self.runAnalysis)

            # Deactivating support button
            self.actionRun_analysis.setEnabled(False)

        elif currentData == 'Batch analysis':
            self.lineEdit_brickSize.setEnabled(False)
            self.plainTextEdit_overview.setEnabled(False)
            self.lineEdit_bedDepth.setEnabled(False)
            self.tableWidget_Permutations.show()
            self.tableWidget_Profile.hide()
            self.label_profileProp.setText('Soil percentage (from input file)')
            self.label_SoilProp.setText('Soil properties (from input file)')
            self.tableWidget_Soil.setEnabled(False)
            self.tableWidget_Permutations.setEnabled(False)
            self.tableWidget_Profile.setEnabled(False)
            self.pushButton_drawProfile.hide()
            self.pushButton_loadBatch.show()
            self.checkBox_updatePlots.setEnabled(True)

            for element in buttonList:
                getattr(self, element).setEnabled(False)

            # Switching signal for run button to batch analysis
            self.pushButton_run.clicked.disconnect()
            self.pushButton_run.clicked.connect(self.runBatch)

            # Activating support button
            self.actionRun_analysis.setEnabled(True)

    def changeInputPanel(self):
        if self.comboBox_THorRVT.currentText() == 'RVT':
            self.groupBox_TH.hide()
            self.groupBox_RVT.show()

            # Disabling TH based outputs
            self.checkBox_outStrain.setEnabled(False)
            self.checkBox_outStrain.setChecked(False)
            self.checkBox_outAcc.setEnabled(False)
            self.checkBox_outAcc.setChecked(False)
            self.lineEdit_accDepth.setEnabled(False)
            self.lineEdit_strainDepth.setEnabled(False)
            self.checkBox_outBrief.setEnabled(True)
        else:
            self.groupBox_RVT.hide()
            self.groupBox_TH.show()

            # Enabling TH based outputs
            self.checkBox_outStrain.setEnabled(True)
            self.checkBox_outAcc.setEnabled(True)
            self.lineEdit_accDepth.setEnabled(True)
            self.lineEdit_strainDepth.setEnabled(True)
            self.checkBox_outBrief.setEnabled(False)

    def loadTH(self):
        timeHistoryFiles = QFileDialog.getOpenFileNames(self, caption='Choose input motion files"',
                                                        filter='Text files (*.txt);;All files (*.*)',
                                                        options=self.dialogOptions)[0]
        if len(timeHistoryFiles) == 0:
            return None

        try:
            currentFS = float(self.lineEdit_FS.text())
        except ValueError:
            currentFS = None

        currentUnits = self.comboBox_Units.currentText()

        inputMotionDict = SRALib.loadTH(timeHistoryFiles, currentFS, currentUnits)
        self.inputMotion = inputMotionDict
        self.comboBox_eventList.clear()
        self.comboBox_eventList.addItems(inputMotionDict.keys())
        self.comboBox_eventList.setCurrentIndex(0)
        self.comboBox_eventList.setEnabled(True)

    def loadSpectra(self):
        spectraFiles = QFileDialog.getOpenFileNames(self, caption='Choose input motion files"',
                                                    filter='Text files (*.txt)',
                                                    options=self.dialogOptions)[0]
        if len(spectraFiles) == 0:
            return None
        currentUnits = self.comboBox_SpectraUnits.currentText()

        try:
            Duration = float(self.lineEdit_duration.text())
            Damping = float(self.lineEdit_damping.text())
        except ValueError:
            msg = "Damping and duration must be numeric values"
            QMessageBox.warning(QMessageBox(), "Check values", msg)
            return None

        if Duration < 0:
            msg = "Duration must be a positive value"
            QMessageBox.warning(QMessageBox(), "Check duration", msg)
            return None
        elif any([Damping < 0, Damping > 100]):
            msg = "Damping value must be in the range 0-100"
            QMessageBox.warning(QMessageBox(), "Check damping", msg)
            return None

        waitBar = QProgressDialog("Importing {} files..".
                                  format(len(spectraFiles)), "Cancel", 0, len(spectraFiles))
        waitBar.show()
        App.processEvents()

        inputMotionDict = SRALib.loadSpectra(spectraFiles, float(self.lineEdit_damping.text()),
                                             float(self.lineEdit_duration.text()), currentUnits, waitBar, App)
        self.inputMotion = inputMotionDict
        self.comboBox_spectraList.clear()
        self.comboBox_spectraList.addItems(inputMotionDict.keys())
        self.comboBox_spectraList.setCurrentIndex(0)
        self.comboBox_spectraList.setEnabled(True)

    def runAnalysis(self, batchAnalysis=False, batchOptions=None):
        if self.checkBox_autoDiscretize.isChecked():
            currentMaxFreq = float(self.lineEdit_maxFreq.text())
            currentWaveLength = float(self.lineEdit_waveLength.text())
        else:
            currentMaxFreq = currentWaveLength = -1

        analysisType = self.comboBox_analysisType.currentText()
        soilList, profileList, permutationList = SRALib.table2list(self.tableWidget_Soil, self.tableWidget_Profile,
                                                                   self.tableWidget_Permutations)
        outputList = [self.checkBox_outRS.isChecked(), self.checkBox_outAcc.isChecked(),
                      self.checkBox_outStrain.isChecked(), self.checkBox_outBrief.isChecked(),
                      self.checkBox_Fourier.isChecked()]
        outputParam = [float(x.text())
                       for x in [self.lineEdit_RSDepth, self.lineEdit_accDepth, self.lineEdit_strainDepth,
                                 self.lineEdit_briefDepth, self.lineEdit_FourierDepth]]
        LECalcOptions = [float(x.text())
                         for x in [self.lineEdit_strainRatio, self.lineEdit_maxTol, self.lineEdit_maxIter]]
        checkPreliminari = self.preAnalysisChecks(soilList, profileList, permutationList, outputList)
        if checkPreliminari is None:
            return None

        analysisDB = {'CurveDB': self.curveDB, 'Discretization': [currentMaxFreq, currentWaveLength],
                      'Bedrock': [self.lineEdit_bedWeight.text(), self.lineEdit_bedVelocity.text(),
                                  self.lineEdit_bedDamping.text()], 'OutputList': outputList,
                      'OutputParam': outputParam, 'LECalcOptions': LECalcOptions}

        if batchAnalysis:
            outputFolder = batchOptions['outputFolder']
        else:
            outputFolder = QFileDialog.getExistingDirectory(self, 'Choose a folder for output generation',
                                                            options=self.dialogOptions)
            if outputFolder == '':
                return None

        if batchAnalysis:
            profilePermutations = batchOptions['profilePermutations']
            vsList = batchOptions['vsList']
            stdList = batchOptions['curveStd']
        else:  # Analysis with input from GUI
            vsList = None
            stdList = None
            if analysisType == 'Permutations':
                bedrockDepth = float(self.lineEdit_bedDepth.text())
                brickSize = float(self.lineEdit_brickSize.text())
                brickProfile = SRALib.makeBricks(self.tableWidget_Permutations, brickSize, bedrockDepth)
                numberPermutations = SRALib.calcPermutations(brickProfile)[0]

                waitBar = QProgressDialog("Generating {} permutations..".
                                          format(int(numberPermutations)), "Cancel", 0, 1)
                waitBar.setWindowTitle('NC92-Soil permutator')
                waitBar.setValue(0)
                waitBar.setMinimumDuration(0)
                waitBar.show()
                App.processEvents()

                profilePermutations = SRALib.calcPermutations(brickProfile, returnpermutations=True)

                waitBar.setValue(1)
                App.processEvents()
            else:  # Regular analysis
                profilePermutations = [profileList]

        if batchAnalysis:
            currentData = {key: value for key, value in self.inputMotion.items() if key in batchOptions['inputSet']}
        else:
            currentData = self.inputMotion
        # totalMotions = len(currentData.keys())
        totalProfiles = len(profilePermutations)

        risultatiDict = dict()
        profiliDF = pd.DataFrame()

        if not batchAnalysis:
            waitBar = QProgressDialog("Running analysis, please wait..", "Cancel", 0, totalProfiles-1)
            waitBar.setWindowTitle('NC92-Soil')
            waitBar.setValue(0)
            waitBar.setMinimumDuration(0)
            waitBar.show()
            App.processEvents()
        else:
            waitBar = None

        for numberProfile, profiloCorrente in enumerate(profilePermutations):

            if waitBar:
                waitBar.setLabelText('Profile {} of {}..'.format(numberProfile+1, totalProfiles))

            profileList = SRALib.addDepths(profiloCorrente)  # Add depths to current profile
            currentSoilList, profileList, curveStd = SRALib.addVariableProperties(
                soilList, profileList, vsList, stdList)
            profileSoilNames = [strato[2] for strato in profileList]
            profileSoilThick = [str(strato[1]) for strato in profileList]
            profileVs = [strato[-1] for strato in profileList]
            profileSoilDesc = ["{} - {} m/s - {} m".format(name, vsValue, thickness)
                               for name, vsValue, thickness in zip(profileSoilNames, profileVs, profileSoilThick)]
            if batchAnalysis:
                if batchOptions['batchType'] == 'permutations':
                    profileCode = "{}/P{}".format(batchOptions['currentPrefix'], numberProfile + 1)
                else:
                    profileCode = "{}".format(batchOptions['currentPrefix'])
            else:
                profileCode = "P{}".format(numberProfile + 1)
            profiliDF[profileCode] = profileSoilDesc

            for fileName in currentData.keys():

                OutputResult = SRALib.runAnalysis(currentData[fileName][0], currentSoilList, profileList,
                                                  analysisDB, curveStd)

                # Esportazione output
                for risultato in OutputResult:
                    currentOutput = type(risultato).__name__
                    ascLabel = ''

                    if currentOutput == 'ResponseSpectrumOutput':
                        periodVect = risultato.refs[::-1] ** -1
                        PSAVect = risultato.values[::-1]
                        currentDF = pd.DataFrame(np.array([periodVect, PSAVect])).T
                        ascLabel = 'T [s]'
                    elif currentOutput == 'BriefReportOutput':
                        ascVect = risultato.refs
                        ordVect = risultato.values
                        currentDF = pd.DataFrame(ordVect).T
                        currentDF.columns = ascVect
                    else:
                        if currentOutput == 'TransferFunctionOutput':
                            currentOutput = "{} - ({})".format(currentOutput, risultato.tag)
                            ascLabel = 'Frequency [Hz]'
                        ascVect = risultato.refs
                        ordVect = risultato.values
                        currentDF = pd.DataFrame(np.array([ascVect, ordVect])).T

                    currentEvent = fileName

                    if currentOutput not in risultatiDict.keys():
                        risultatiDict[currentOutput] = dict()
                    if currentEvent not in risultatiDict[currentOutput].keys():
                        if currentOutput != 'BriefReportOutput':
                            # risultatiDict[currentOutput][currentEvent] = pd.DataFrame(currentDF[0])
                            risultatiDict[currentOutput][currentEvent] = currentDF[0].to_frame().rename(
                                columns={0: ascLabel})
                        else:
                            risultatiDict[currentOutput][currentEvent] = pd.DataFrame(columns=risultato.refs)

                    if currentOutput == 'BriefReportOutput':
                        newRow = pd.DataFrame(columns=risultato.refs)
                        # newRow = newRow.astype('float64')
                        # newRow['Shallow soil name'] = newRow.astype('str')
                        newRow.loc[0] = risultato.values
                        newRow.index = [profileCode]
                        # risultatiDict[currentOutput][currentEvent] = \
                        #     risultatiDict[currentOutput][currentEvent].append(newRow.loc[profileCode])
                        risultatiDict[currentOutput][currentEvent] = \
                            pd.concat([risultatiDict[currentOutput][currentEvent],
                                       newRow.loc[profileCode].to_frame().T])
                    else:
                        risultatiDict[currentOutput][currentEvent][profileCode] = currentDF[1].values

            if waitBar:
                waitBar.setValue(numberProfile)
                App.processEvents()

        # Scrittura dei risultati
        vociOutput = risultatiDict.keys()

        for voce in vociOutput:
            currentOutput = voce
            writeIndex = True if currentOutput == 'BriefReportOutput' else False
            # currentValues = risultatiDict[voce]
            currentExcelFile = os.path.join(outputFolder, "{}.xlsx".format(currentOutput))

            firstIter = True
            for evento in risultatiDict[voce].keys():
                if firstIter:
                    writeMode = 'w'
                else:
                    writeMode = 'a'

                with pd.ExcelWriter(currentExcelFile, mode=writeMode) as writer:
                    risultatiDict[voce][evento].to_excel(writer, sheet_name=evento, index=writeIndex)
                firstIter = False

        if not batchAnalysis:
            # Writing profile table
            profiliExcelFile = os.path.join(outputFolder, 'Profiles.xlsx')
            profiliDF.to_excel(profiliExcelFile, sheet_name='Profiles table', index=False)
            QMessageBox.information(QMessageBox(), 'OK', 'Analysis results have been correctly exported')

    def preAnalysisChecks(self, soilList, profileList, permutationList, outputList):
        currentAnalysis = self.comboBox_analysisType.currentText()

        # Controllo campi vuoti
        if len(soilList) == 0:
            msg = "Soil table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check soil", msg)
            return None
        elif len(profileList) == 0 and currentAnalysis == 'Regular analysis':
            msg = "Profile table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None
        elif len(permutationList) == 0 and currentAnalysis == 'Permutations':
            msg = "Soil percentage table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check soil percentage", msg)
            return None

        # Controllo valori non validi
        if soilList == 'SoilNan':
            msg = "The unit weight and velocity in soil table must be a numeric value"
            QMessageBox.warning(QMessageBox(), "Check soil", msg)
            return None
        elif profileList == 'ProfileNan' and currentAnalysis == 'Regular analysis':
            msg = "Layer thickness in profile table must be numeric value"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None
        elif permutationList == 'PermutationNan' and currentAnalysis == 'Permutations':
            msg = "Percentages in soil percentage table must be numeric values"
            QMessageBox.warning(QMessageBox(), "Check soil percentage", msg)
            return None

        # Check di completezza dati di input
        campiVuotiSuolo = [elemento == '' or elemento is None for riga in soilList for elemento in riga]
        campiVuotiProfilo = [elemento == '' or elemento is None for riga in profileList for elemento in riga]
        campiVuotiPermutations = [elemento == '' or elemento is None for riga in permutationList for elemento in riga]

        if any(campiVuotiSuolo):
            msg = "Fields in soil table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check soil", msg)
            return None
        elif any(campiVuotiProfilo) and currentAnalysis == 'Regular analysis':
            msg = "Fields in profile table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None
        elif any(campiVuotiPermutations) and currentAnalysis == 'Permutations':
            msg = "Fields in soil percentage table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None
        elif len(self.inputMotion) == 0:
            msg = "Import an input time history or a target spectrum before running analysis"
            QMessageBox.warning(QMessageBox(), "Check input", msg)
            return None
        elif not any(outputList):
            msg = "No option selected for output"
            QMessageBox.warning(QMessageBox(), "Check output", msg)
            return None
        return 0

    def loadBatch(self):
        batchFile = QFileDialog.getOpenFileNames(self, caption="Choose the input batch file",
                                                 filter='Excel Files (*.xlsx)',
                                                 options=self.dialogOptions)
        if len(batchFile[0]) == 0:
            return None
        else:
            self.batchObject.clear()

            waitBar = QProgressDialog("Importing {} batch files, please wait..".format(len(batchFile[0])),
                                      'Cancel', 0, len(batchFile[0]))
            waitBar.setWindowTitle('Importing batch')
            waitBar.setValue(0)
            waitBar.setMinimumDuration(0)
            waitBar.show()

            for index, element in enumerate(batchFile[0]):
                waitBar.setLabelText("Importing file {}\n({} of {})".format(element, index + 1, len(batchFile[0])))
                App.processEvents()

                self.batchObject.append(BatchAnalyzer(element))

                waitBar.setValue(index + 1)
                App.processEvents()

        QMessageBox.information(self, '', 'Batch input file has been correctly imported')

    def generatePermutatedProfiles(self):
        inputFile = QFileDialog.getOpenFileName(self, caption="Choose the input for cluster analysis",
                                                filter='Excel Files (*.xlsx)',
                                                options=self.dialogOptions)
        if inputFile[0] == "":
            return None

        outputFolder = QFileDialog.getExistingDirectory(self, caption="Choose the output folder for batch files",
                                                        options=self.dialogOptions)

        if outputFolder == "":
            return None

        iterations, ok_pressed = QInputDialog.getInt(None, "Iterations", "Choose the number of iterations", 100)

        if not ok_pressed:
            return None

        # loadClustersObj = ClusterPermutator(inputFile[0], outputFolder)
        loadClustersObj = ClusterToMOPS(inputFile[0], outputFolder, iterations)

        waitBar = QProgressDialog("Exporting batch files..", "Cancel", 0,
                                  loadClustersObj.number_clusters)

        waitBar.setWindowTitle('NC92-Soil Permutator')
        waitBar.setMinimumDuration(0)
        waitBar.show()
        App.processEvents()

        fileCounter = 1
        for index, id_code in enumerate(loadClustersObj.name_list):
            waitBar.setLabelText('Generating batch input for ID code "{}"'.format(id_code))
            App.processEvents()
            loadClustersObj.makeClusterProfiles(index)

            waitBar.setValue(fileCounter)
            fileCounter += 1

        QMessageBox.information(QMessageBox(), 'OK',
                                'Batch files for permutation analysis have been correctly exported')

    def loadStochastic(self):
        """
        Load the stochastic input Excel file to generate the profiles for batch analysis
        :return:
        """
        stochasticFile = QFileDialog.getOpenFileName(self, caption="Choose the input stochastic file",
                                                     filter='Excel Files (*.xlsx)',
                                                     options=self.dialogOptions)
        if stochasticFile[0] == "":
            return None
        else:
            stochasticObject = StochasticAnalyzer(stochasticFile[0])

        exportFolder = QFileDialog.getExistingDirectory(self, caption="Choose the output folder for batch files",
                                                        options=self.dialogOptions)

        if exportFolder == "":
            return None

        waitBar = QProgressDialog("Exporting batch files..", "Cancel", 0,
                                  len(stochasticObject.idList))

        waitBar.setWindowTitle('NC92-Soil Stochastic')
        # waitBar.setValue(0)
        waitBar.setMinimumDuration(0)
        waitBar.show()
        App.processEvents()

        fileCounter = 1
        for id_code in stochasticObject.idList:
            waitBar.setLabelText('Generating batch input for ID code "{}"'.format(id_code))
            App.processEvents()
            currentFilename = "{}.xlsx".format(id_code)
            stochasticObject.updateByID(id_code)

            for iteration in range(stochasticObject.numberIterations):
                # Generation of the random profile
                stochasticObject.generateRndProfile()

            stochasticObject.exportExcel(os.path.join(exportFolder, currentFilename))
            waitBar.setValue(fileCounter)
            fileCounter += 1

        QMessageBox.information(QMessageBox(), 'OK', 'Batch file for stochastic analysis has been correctly exported')

    def runBatch(self):
        if len(self.batchObject) == 0:
            msg = "Please import a valid batch input file before running the analysis"
            QMessageBox.warning(QMessageBox(), "Check batch input", msg)
            return None

        outputFolder = QFileDialog.getExistingDirectory(self, 'Choose a folder for output generation',
                                                        options=self.dialogOptions)
        if outputFolder == '':
            return None

        batchObjectList = self.batchObject

        totalAnalysis = sum([element.profileNumber for element in batchObjectList])
        if totalAnalysis > 0:
            waitBar = QProgressDialog("Analyzing {} profiles...", "Cancel", 0, totalAnalysis)
            waitBar.setWindowTitle('NC92-Soil Batch')
            waitBar.setValue(0)
            waitBar.setMinimumDuration(0)
            waitBar.show()
            App.processEvents()
        else:
            waitBar = None

        all_profile_counter = 0
        for fileIndex, batchObject in enumerate(batchObjectList):

            # TEST CODE FOR MULTITHREADING
            # Check per verifica tempi
            current_time = time()

            numberClusters = batchObject.clusterNumber
            numberProfiles = batchObject.profileNumber

            # Batch analysis with clusters
            if numberClusters > 0:
                self.tableWidget_Permutations.show()
                self.tableWidget_Profile.hide()
                clustersOutputFolder = os.path.join(outputFolder, 'Clusters')
                try:
                    os.mkdir(clustersOutputFolder)
                except FileExistsError:
                    pass

                for clusterIndex in range(numberClusters):
                    currentCluster, currentDepth, currentBrick = batchObject.getClusterInfo(clusterIndex)
                    currentMotions = batchObject.getInputNames(clusterIndex, element_type='clusters')
                    self.lineEdit_brickSize.setText(str(currentBrick))
                    self.lineEdit_bedDepth.setText(str(currentDepth))

                    # Generating permutations for current cluster
                    self.list2table(permutationTable=currentCluster)
                    bedrockDepth = float(self.lineEdit_bedDepth.text())
                    brickSize = float(self.lineEdit_brickSize.text())
                    brickProfile = SRALib.makeBricks(self.tableWidget_Permutations, brickSize, bedrockDepth)
                    numberPermutations = SRALib.calcPermutations(brickProfile)[0]

                    waitBar = QProgressDialog("Generating {} permutations..".format(
                        int(numberPermutations)), "Cancel", 0, 1)
                    waitBar.setWindowTitle('NC92-Soil permutator')
                    waitBar.setValue(0)
                    waitBar.setMinimumDuration(0)
                    waitBar.show()
                    App.processEvents()

                    profilePermutations = SRALib.calcPermutations(brickProfile, returnpermutations=True)

                    waitBar.setValue(1)
                    App.processEvents()

                    for vsIndex in range(batchObject.vsNumber):
                        currentSoil = batchObject.getSoilTable(vsIndex)
                        self.list2table(soilTable=currentSoil)
                        currentName = "{}-VS{}".format(
                            batchObject.getElementName(clusterIndex, 'clusters'), vsIndex + 1)
                        currentFolder = os.path.join(clustersOutputFolder, currentName)
                        try:
                            os.mkdir(currentFolder)
                        except FileExistsError:
                            pass

                        # Running analysis
                        batchOptions = {'batchType': 'permutations', 'inputSet': currentMotions,
                                        'outputFolder': currentFolder, 'currentPrefix': currentName,
                                        'profilePermutations': profilePermutations, 'vsList': None,
                                        'curveStd': None}
                        self.runAnalysis(batchAnalysis=True, batchOptions=batchOptions)

            # Batch analysis with profiles
            if numberProfiles > 0:
                if fileIndex == 0:
                    self.tableWidget_Permutations.hide()
                    self.tableWidget_Profile.show()

                currentProfilesID = os.path.splitext(os.path.split(batchObject.filename)[1])[0]
                self.label_profileProp.setText('Profile (from input file)')

                profilesOutputFolder = os.path.join(outputFolder, 'Profiles - {}'.format(currentProfilesID))
                try:
                    os.mkdir(profilesOutputFolder)
                except FileExistsError:
                    pass

                # Getting information about degradation curves std
                curveStdVector = batchObject.getDegradationCurveStd()

                # Updating waitbar
                if waitBar:
                    # waitBar.setLabelText('Processing profile {} in file {}'.format(
                    #     currentName, os.path.basename(batchObject.filename)))
                    waitBar.setLabelText('Processing file {}'.format(
                        os.path.basename(batchObject.filename)))
                    App.processEvents()

                for profileIndex in range(numberProfiles):
                    all_profile_counter += 1
                    currentProfile, currentVsList, currentBedrockVs = batchObject.getProfileInfo(profileIndex)

                    currentMotions = [inputName
                                      for inputName in batchObject.getInputNames(profileIndex, element_type='profiles')
                                      if inputName.strip() != ""]

                    if len(currentMotions) == 0 or currentMotions[0].upper() == 'ALL':
                        # No input specified, all the imported input will be used
                        currentMotions = list(self.inputMotion.keys())

                    # If VS is specified only one analysis is performed
                    if all([element == -1 for element in currentVsList]):
                        numberVs = batchObject.vsNumber
                    else:
                        numberVs = 1

                    for vsIndex in range(numberVs):
                        currentSoil = batchObject.getSoilTable(vsIndex)
                        if numberVs == 1:
                            currentName = "{}".format(batchObject.getElementName(profileIndex, 'profiles'))
                        else:
                            currentName = "{}-VS{}".format(
                                batchObject.getElementName(profileIndex, 'profiles'), vsIndex + 1)

                        currentFolder = os.path.join(profilesOutputFolder, currentName)
                        try:
                            os.mkdir(currentFolder)
                        except FileExistsError:
                            pass

                        # Updating bedrock Vs
                        self.lineEdit_bedVelocity.setText(currentBedrockVs)

                        self.list2table(currentSoil, profileTable=currentProfile)
                        self.userModified = True
                        self.profileChanged()

                        if self.checkBox_updatePlots.isChecked() and vsIndex == 0:
                            self.makeProfile()
                            App.processEvents()

                        # Running analysis
                        batchOptions = {'batchType': 'profiles', 'inputSet': currentMotions,
                                        'outputFolder': currentFolder,
                                        'currentPrefix': currentName, 'profilePermutations': [currentProfile],
                                        'vsList': currentVsList, 'curveStd': curveStdVector}
                        self.runAnalysis(batchAnalysis=True, batchOptions=batchOptions)

                if waitBar:
                    waitBar.setValue(all_profile_counter)

            print('File "{}" - {:.2f} s'.format(batchObject.filename, time()-current_time))
            App.processEvents()
        QMessageBox.information(QMessageBox(), 'OK', 'Batch analysis has been correctly completed')

    def list2table(self, soilTable=None, permutationTable=None, profileTable=None):
        if soilTable is not None:
            # Updating soil table
            self.tableWidget_Soil.clearContents()
            self.tableWidget_Soil.setRowCount(0)
            for rowIndex, row in enumerate(soilTable):
                self.addRow(sender='Soil')
                for elemIndex, element in enumerate(row):
                    if elemIndex == 5:
                        self.tableWidget_Soil.cellWidget(rowIndex, elemIndex).setCurrentText(element)
                    else:
                        currentItem = QTableWidgetItem(str(element)) if element is not None else element
                        self.tableWidget_Soil.setItem(rowIndex, elemIndex, currentItem)

        if permutationTable is not None:
            # Updating permutations table
            self.tableWidget_Permutations.clearContents()
            self.tableWidget_Permutations.setRowCount(0)
            for rowIndex, row in enumerate(permutationTable):
                self.addRow(sender='Permutations')
                for elemIndex, element in enumerate(row):
                    self.tableWidget_Permutations.setItem(rowIndex, elemIndex, QTableWidgetItem(str(element)))

        if profileTable is not None:
            self.tableWidget_Profile.clearContents()
            self.tableWidget_Profile.setRowCount(0)
            for rowIndex, row in enumerate(profileTable):
                self.addRow(sender='Profile')
                self.userModified = False
                if rowIndex == 0:
                    self.tableWidget_Profile.setItem(rowIndex, 0, QTableWidgetItem('0.0'))

                for elemIndex, element in enumerate(row):
                    self.tableWidget_Profile.setItem(rowIndex, elemIndex + 1, QTableWidgetItem(str(element)))

    def generateNTC(self):
        coordinate_file = QFileDialog.getOpenFileName(self, caption='Choose the input file with site coordinates"',
                                                      filter='Excel Files (*.xlsx)',
                                                      options=self.dialogOptions)[0]

        if len(coordinate_file) == 0:
            return None

        period_list = ['30', '50', '72', '101', '140', '201', '475', '975', '2475']
        selection, ok = QInputDialog.getItem(self, "Tr",
                                             "Choose the return period", period_list, 0, False)

        if ok:
            tr = str(selection)
        else:
            return None

        outputFolder = QFileDialog.getExistingDirectory(self, 'Choose a folder for output generation',
                                                        options=self.dialogOptions)
        if outputFolder == '':
            return None

        mops_coord = pd.read_excel(coordinate_file, sheet_name='Stochastic', dtype={'ID CODE': str}).\
            dropna(axis=0, subset=['Lon']).drop_duplicates(subset='ID CODE').reset_index()
        A = NTCCalculator('NTC2008.csv')

        number_rows = len(mops_coord)
        waitBar = QProgressDialog("Generating {} NTC spectra with a return period of {} years..".
                                  format(number_rows, tr), "Cancel", 0, number_rows)
        waitBar.setWindowTitle('NC92-Soil spectra generator')
        waitBar.setValue(0)
        waitBar.setMinimumDuration(0)
        waitBar.show()
        App.processEvents()

        for index, row in mops_coord.iterrows():
            waitBar.setLabelText('Spectrum for {} ({} of {})'.format(row['ID CODE'], index + 1, number_rows))
            App.processEvents()
            ag, F0, Tc = A.agNTC(row['Lon'], row['Lat'], tr=tr)
            current_spectrum = A.computeNTCSpectrum(ag, F0, Tc)
            current_file = os.path.join(outputFolder, "{}.txt".format(row['ID CODE']))
            np.savetxt(fname=current_file, X=current_spectrum, fmt='%f\t%f')
            waitBar.setValue(index + 1)
            App.processEvents()

        QMessageBox.information(QMessageBox(), 'OK', 'NTC spectra have been correctly saved')

    def makeStats(self):
        caller_obj = self.sender().objectName()

        analysis_folder = QFileDialog.getExistingDirectory(self, 'Choose the folder containining the analysis output',
                                                           options=self.dialogOptions)
        if analysis_folder == '':
            return None

        if caller_obj == 'actionGenerate_only_master':
            output_path = QFileDialog.getSaveFileName(self, caption='Choose the name of the merged report"',
                                                      filter='Excel Files (*.xlsx)',
                                                      options=self.dialogOptions)[0]
            if output_path == "":
                return None

            make_subs = False
        else:
            output_path = QFileDialog.getExistingDirectory(self,
                                                           'Choose the folder where the reports will be saved',
                                                           options=self.dialogOptions)
            if output_path == '':
                return None
            make_subs = True

        waitBar = QProgressDialog("Preparing merge, please wait...", "Cancel", 0, 1)
        waitBar.show()
        App.processEvents()
        exitCode = SRALib.makeStats(analysis_folder, output_path, make_subs, waitBar, App)

        if exitCode == 0:
            QMessageBox.information(QMessageBox(), 'OK', 'Merge has been correctly performed.')
        else:
            msg = "Permission error while writing file : \n{}\nIf the file is open, close it and run merge again".\
                format(exitCode)
            QMessageBox.warning(QMessageBox(), "Permission error", msg)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    MainWindow = SRAApp()
    MainWindow.show()
    App.exec_()
