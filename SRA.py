from warnings import filterwarnings

filterwarnings('ignore', category=UserWarning)
filterwarnings('ignore', category=RuntimeWarning)

from PySide2.QtWidgets import *
from PySide2 import QtCore
from SRAmainGUI import Ui_MainWindow
from SRAClasses import BatchAnalyzer
import numpy as np
import sys
import SRALibrary as SRALib
import pandas as pd
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = ''

from pygame import mixer, error as pygameexception


def aboutMessage():
    mixer.init()
    Messaggio = QMessageBox()
    Messaggio.setText("NC92-Soil\nversion 0.6 beta\n"
                      "\nCNR IGAG")
    Messaggio.setWindowTitle("NC92-Soil")
    try:
        mixer.music.load('about.mp3')
        mixer.music.play()
    except pygameexception:
        pass
    mixer.music.set_volume(0.5)
    Messaggio.exec_()
    mixer.music.stop()


# noinspection PyCallByClass
class SRAApp(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        # Inizializzazione attributi principali
        self.curveDB = dict()
        self.userModified = True
        self.inputMotion = dict()
        self.batchObject = None

        self.setupUi(self)
        self.assignWidgets()
        self.setDefault()
        self.show()

    def setDefault(self):
        # Formattazione delle tabelle
        self.tableWidget_Soil.resizeColumnsToContents()
        self.tableWidget_Soil.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_Profile.resizeColumnsToContents()
        self.tableWidget_Profile.horizontalHeader().setStretchLastSection(True)
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

    def updateOutputInfo(self):
        if self.sender() is self.lineEdit_RSDepth:
            self.lineEdit_briefDepth.setText(self.sender().text())
        elif self.sender() is self.checkBox_outBrief:
            if self.sender().isChecked():
                self.checkBox_outRS.setChecked(True)
        elif self.sender() is self.checkBox_outRS:
            if not self.sender().isChecked():
                self.checkBox_outBrief.setChecked(False)

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
            self.pushButton_drawProfile.show()
            self.pushButton_loadBatch.hide()

            for element in buttonList:
                getattr(self, element).setEnabled(True)

            # Switching signal for run button to manual analysis
            self.pushButton_run.clicked.disconnect()
            self.pushButton_run.clicked.connect(self.runAnalysis)
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
            self.pushButton_drawProfile.show()
            self.pushButton_loadBatch.hide()

            for element in buttonList:
                getattr(self, element).setEnabled(True)

            # Switching signal for run button to manual analysis
            self.pushButton_run.clicked.disconnect()
            self.pushButton_run.clicked.connect(self.runAnalysis)

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
            self.pushButton_drawProfile.hide()
            self.pushButton_loadBatch.show()

            for element in buttonList:
                getattr(self, element).setEnabled(False)

            # Switching signal for run button to batch analysis
            self.pushButton_run.clicked.disconnect()
            self.pushButton_run.clicked.connect(self.runBatch)

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
        timeHistoryFiles = QFileDialog.getOpenFileNames(self, caption='Choose input motion files"')[0]
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
        spectraFiles = QFileDialog.getOpenFileNames(self, caption='Choose input motion files"')[0]
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

        inputMotionDict = SRALib.loadSpectra(spectraFiles, float(self.lineEdit_damping.text()),
                                             float(self.lineEdit_duration.text()), currentUnits)
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
                      self.checkBox_outStrain.isChecked(), self.checkBox_outBrief.isChecked()]
        outputParam = [float(x.text())
                       for x in [self.lineEdit_RSDepth, self.lineEdit_accDepth, self.lineEdit_strainDepth,
                                 self.lineEdit_briefDepth]]
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
            outputFolder = QFileDialog.getExistingDirectory(self, 'Choose a folder for output generation')
            if outputFolder == '':
                return None

        if batchAnalysis:
            profilePermutations = batchOptions['profilePermutations']
            vsList = batchOptions['vsList']
        else:  # Analysis with input from GUI
            vsList = None
            if analysisType == 'Permutations':
                bedrockDepth = float(self.lineEdit_bedDepth.text())
                brickSize = float(self.lineEdit_brickSize.text())
                brickProfile = SRALib.makeBricks(self.tableWidget_Permutations, brickSize, bedrockDepth)
                numberPermutations = SRALib.calcPermutations(brickProfile)[0]

                waitBar = QProgressDialog("Generating {} permutations..".format(int(numberPermutations)), "Cancel", 0, 1)
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

        waitBar = QProgressDialog("Running analysis, please wait..", "Cancel", 0, totalProfiles-1)
        waitBar.setWindowTitle('NC92-Soil')
        waitBar.setValue(0)
        waitBar.setMinimumDuration(0)
        waitBar.show()
        App.processEvents()

        for numberProfile, profiloCorrente in enumerate(profilePermutations):

            waitBar.setLabelText('Profile {} of {}..'.format(numberProfile+1, totalProfiles))

            profileList = SRALib.addDepths(profiloCorrente)  # Add depths to current profile
            currentSoilList, profileList = SRALib.addVariableProperties(soilList, profileList, vsList)
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

                OutputResult = SRALib.runAnalysis(currentData[fileName][0], currentSoilList, profileList, analysisDB)

                # Esportazione output
                for risultato in OutputResult:
                    currentOutput = type(risultato).__name__
                    if currentOutput == 'ResponseSpectrumOutput':
                        periodVect = risultato.refs[::-1] ** -1
                        PSAVect = risultato.values[::-1]
                        currentDF = pd.DataFrame(np.array([periodVect, PSAVect])).T
                    elif currentOutput == 'BriefReportOutput':
                        ascVect = risultato.refs
                        ordVect = risultato.values
                        currentDF = pd.DataFrame(ordVect).T
                        currentDF.columns = ascVect
                    else:
                        ascVect = risultato.refs
                        ordVect = risultato.values
                        currentDF = pd.DataFrame(np.array([ascVect, ordVect])).T

                    currentEvent = fileName

                    if currentOutput not in risultatiDict.keys():
                        risultatiDict[currentOutput] = dict()
                    if currentEvent not in risultatiDict[currentOutput].keys():
                        if currentOutput != 'BriefReportOutput':
                            risultatiDict[currentOutput][currentEvent] = pd.DataFrame(currentDF[0])
                        else:
                            risultatiDict[currentOutput][currentEvent] = pd.DataFrame(columns=risultato.refs)

                    if currentOutput == 'BriefReportOutput':
                        newRow = pd.DataFrame(columns=risultato.refs)
                        # newRow = newRow.astype('float64')
                        # newRow['Shallow soil name'] = newRow.astype('str')
                        newRow.loc[0] = risultato.values
                        newRow.index = [profileCode]
                        risultatiDict[currentOutput][currentEvent] = \
                            risultatiDict[currentOutput][currentEvent].append(newRow.loc[profileCode])
                    else:
                        risultatiDict[currentOutput][currentEvent][profileCode] = currentDF[1].values

            waitBar.setValue(numberProfile)
            App.processEvents()

        # Scrittura dei risultati
        vociOutput = risultatiDict.keys()

        for voce in vociOutput:
            currentOutput = voce
            writeIndex = True if currentOutput == 'BriefReportOutput' else False
            currentValues = risultatiDict[voce]
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

        # Writing profile table
        profiliExcelFile = os.path.join(outputFolder, 'Profiles.xlsx')
        profiliDF.to_excel(profiliExcelFile, sheet_name='Profiles table', index=False)

        if not batchAnalysis:
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
        batchFile = QFileDialog.getOpenFileName(self, caption="Choose the input batch file",
                                                filter='Excel Files (*.xlsx)')
        if len(batchFile) == 0:
            return None
        else:
            self.batchObject = BatchAnalyzer(batchFile[0])

        QMessageBox.information(QMessageBox(), 'OK', 'Batch input file has been correctly imported')

    def runBatch(self):
        if self.batchObject is None:
            msg = "Please import a valid batch input file before running the analysis"
            QMessageBox.warning(QMessageBox(), "Check batch input", msg)
            return None

        outputFolder = QFileDialog.getExistingDirectory(self, 'Choose a folder for output generation')
        if outputFolder == '':
            return None

        batchObject = self.batchObject
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
                bedrockDepth = float(self.lineEdit_bedDepth.text())
                brickSize = float(self.lineEdit_brickSize.text())
                brickProfile = SRALib.makeBricks(self.tableWidget_Permutations, brickSize, bedrockDepth)
                numberPermutations = SRALib.calcPermutations(brickProfile)[0]

                waitBar = QProgressDialog("Generating {} permutations..".format(int(numberPermutations)), "Cancel", 0,
                                          1)
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
                    self.list2table(currentSoil, permutationTable=currentCluster)
                    currentName = "{}-VS{}".format(batchObject.getElementName(clusterIndex, 'clusters'), vsIndex + 1)
                    currentFolder = os.path.join(clustersOutputFolder, currentName)
                    try:
                        os.mkdir(currentFolder)
                    except FileExistsError:
                        pass

                    # Running analysis
                    batchOptions = {'batchType': 'permutations', 'inputSet': currentMotions,
                                    'outputFolder': currentFolder, 'currentPrefix': currentName,
                                    'profilePermutations': profilePermutations, 'vsList': None}
                    self.runAnalysis(batchAnalysis=True, batchOptions=batchOptions)

        # Batch analysis with profiles
        if numberProfiles > 0:
            self.tableWidget_Permutations.hide()
            self.tableWidget_Profile.show()
            profilesOutputFolder = os.path.join(outputFolder, 'Profiles')
            try:
                os.mkdir(profilesOutputFolder)
            except FileExistsError:
                pass

            for profileIndex in range(numberProfiles):
                currentProfile, currentVsList = batchObject.getProfileInfo(profileIndex)
                currentMotions = batchObject.getInputNames(profileIndex, element_type='profiles')

                # If VS is specified only one analysis is performed
                if all([element == -1 for element in currentVsList]):
                    numberVs = batchObject.vsNumber
                else:
                    numberVs = 1

                for vsIndex in range(numberVs):
                    currentSoil = batchObject.getSoilTable(vsIndex)
                    currentName = "{}-VS{}".format(batchObject.getElementName(profileIndex, 'profiles'), vsIndex + 1)
                    currentFolder = os.path.join(profilesOutputFolder, currentName)
                    try:
                        os.mkdir(currentFolder)
                    except FileExistsError:
                        pass

                    self.list2table(currentSoil, profileTable=currentProfile)
                    self.userModified = True
                    self.profileChanged()
                    # Running analysis
                    batchOptions = {'batchType': 'profiles', 'inputSet': currentMotions, 'outputFolder': currentFolder,
                                    'currentPrefix': currentName, 'profilePermutations': [currentProfile],
                                    'vsList': currentVsList}
                    self.runAnalysis(batchAnalysis=True, batchOptions=batchOptions)

        QMessageBox.information(QMessageBox(), 'OK', 'Batch analysis has been correctly completed')

    def list2table(self, soilTable, permutationTable=None, profileTable=None):
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


if __name__ == "__main__":
    App = QApplication(sys.argv)
    MainWindow = SRAApp()
    MainWindow.show()
    App.exec_()
