from warnings import filterwarnings

filterwarnings('ignore', category=UserWarning)
from PySide2.QtWidgets import *
from PySide2 import QtCore
from SRAmainGUI import Ui_MainWindow
import numpy as np
import sys
import SRALibrary as SRALib
import pandas as pd
import os


# noinspection PyCallByClass
class SRAApp(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        # Inizializzazione attributi principali
        self.curveDB = dict()
        self.userModified = True
        self.inputMotion = dict()

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
        self.comboBox_analysisType.currentIndexChanged.connect(self.changeAnalysis)

    def addRow(self):
        senderName = self.sender().objectName()
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
        if currentAnalysis == 'Permutations':
            profileList = SRALib.makeBricks(self.tableWidget_Profile, brickSize)
            totalPermutations, percString, uniqueBricks = SRALib.calcPermutations(profileList)

            # Writing informations in the overview console
            messageString = "Total bricks: {} - Minimum brick size: {}m - Brick types: {}\nSoil composition -> {}\n" \
                            "Total permutations: {}".format(len(profileList), brickSize, uniqueBricks, percString,
                                                            "{:,}".format(int(totalPermutations)).replace(',', "'")
                                                            )
            self.plainTextEdit_overview.setPlainText(messageString)
        else:
            profileList = SRALib.table2list(self.tableWidget_Soil, self.tableWidget_Profile)[1]

        SRALib.drawProfile(profileList, self.graphWidget.axes)
        self.graphWidget.draw()

    def profileChanged(self):
        if not self.userModified:
            return None
        rowNumber = self.sender().rowCount()

        self.userModified = False
        for riga in range(1, rowNumber):
            currentDepth = float(self.sender().item(riga - 1, 0).text()) + float(self.sender().item(riga - 1, 1).text())
            self.sender().setItem(riga, 0, QTableWidgetItem(str(currentDepth)))
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

        pass

    def changeAnalysis(self):
        currentData = self.comboBox_analysisType.currentText()
        if currentData == 'Permutations':
            self.lineEdit_brickSize.setEnabled(True)
            self.plainTextEdit_overview.setEnabled(True)
        else:
            self.lineEdit_brickSize.setEnabled(False)
            self.plainTextEdit_overview.setEnabled(False)

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

    def runAnalysis(self):
        if self.checkBox_autoDiscretize.isChecked():
            currentMaxFreq = float(self.lineEdit_maxFreq.text())
            currentWaveLength = float(self.lineEdit_waveLength.text())
        else:
            currentMaxFreq = currentWaveLength = -1

        analysisType = self.comboBox_analysisType.currentText()
        soilList, profileList = SRALib.table2list(self.tableWidget_Soil, self.tableWidget_Profile)
        outputList = [self.checkBox_outRS.isChecked(), self.checkBox_outAcc.isChecked(),
                      self.checkBox_outStrain.isChecked()]
        outputParam = [float(x.text())
                       for x in [self.lineEdit_RSDepth, self.lineEdit_accDepth, self.lineEdit_strainDepth]]
        LECalcOptions = [float(x.text())
                         for x in [self.lineEdit_strainRatio, self.lineEdit_maxTol, self.lineEdit_maxIter]]
        checkPreliminari = self.preAnalysisChecks(soilList, profileList, outputList)
        if checkPreliminari is None:
            return None

        analysisDB = {'CurveDB': self.curveDB, 'Discretization': [currentMaxFreq, currentWaveLength],
                      'Bedrock': [self.lineEdit_bedWeight.text(), self.lineEdit_bedVelocity.text(),
                                  self.lineEdit_bedDamping.text()], 'OutputList': outputList,
                      'OutputParam': outputParam, 'LECalcOptions': LECalcOptions}

        outputFolder = QFileDialog.getExistingDirectory(self, 'Choose a folder for output generation')
        if outputFolder == '':
            return None

        if analysisType == 'Permutations':
            brickSize = float(self.lineEdit_brickSize.text())
            brickProfile = SRALib.makeBricks(self.tableWidget_Profile, brickSize)
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
        else:
            profilePermutations = [profileList]

        currentData = self.inputMotion
        analysisCounter = 1
        totalMotions = len(currentData.keys())
        totalProfiles = len(profilePermutations)
        totalAnalysis = totalProfiles*totalMotions

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
            currentSoilList, profileList = SRALib.addVariableProperties(soilList, profileList)
            profileSoilNames = [strato[2] for strato in profileList]
            profileSoilThick = [str(strato[1]) for strato in profileList]
            profileVs = [strato[-1] for strato in profileList]
            profileSoilDesc = ["{} - {} m/s - {} m".format(name, vsValue, thickness)
                               for name, vsValue, thickness in zip(profileSoilNames, profileVs, profileSoilThick)]
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
                    else:
                        ascVect = risultato.refs
                        ordVect = risultato.values
                        currentDF = pd.DataFrame(np.array([ascVect, ordVect])).T

                    currentEvent = fileName

                    if currentOutput not in risultatiDict.keys():
                        risultatiDict[currentOutput] = dict()
                    if currentEvent not in risultatiDict[currentOutput].keys():
                        risultatiDict[currentOutput][currentEvent] = pd.DataFrame(currentDF[0])

                    risultatiDict[currentOutput][currentEvent][profileCode] = currentDF[1].values

            waitBar.setValue(numberProfile)
            App.processEvents()

        # Scrittura dei risultati

        vociOutput = risultatiDict.keys()

        for voce in vociOutput:
            currentOutput = voce
            currentValues = risultatiDict[voce]
            currentExcelFile = os.path.join(outputFolder, "{}.xlsx".format(currentOutput))

            firstIter = True
            for evento in risultatiDict[voce].keys():
                if firstIter:
                    writeMode = 'w'
                else:
                    writeMode = 'a'

                with pd.ExcelWriter(currentExcelFile, mode=writeMode) as writer:
                    risultatiDict[voce][evento].to_excel(writer, sheet_name=evento, index=False)
                firstIter = False

        # Writing profile table
        profiliExcelFile = os.path.join(outputFolder, 'Profiles.xlsx')
        profiliDF.to_excel(profiliExcelFile, sheet_name='Profiles table', index=False)
        A = 5
        QMessageBox.information(QMessageBox(), 'OK', 'Analysis results have been correctly exported')

    def preAnalysisChecks(self, soilList, profileList, outputList):
        # Controllo campi vuoti
        if len(soilList) == 0:
            msg = "Soil table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check soil", msg)
            return None
        elif len(profileList) == 0:
            msg = "Profile table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None

        # Controllo valori non validi
        if soilList == 'SoilNan':
            msg = "The unit weight and velocity in soil table must be a numeric value"
            QMessageBox.warning(QMessageBox(), "Check soil", msg)
            return None
        elif profileList == 'ProfileNan':
            msg = "Layer thickness in profile table must be numeric value"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None

        # Check di completezza dati di input
        campiVuotiSuolo = [elemento == '' or elemento is None for riga in soilList for elemento in riga]
        campiVuotiProfilo = [elemento == '' or elemento is None for riga in profileList for elemento in riga]
        if any(campiVuotiSuolo):
            msg = "Fields in soil table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check soil", msg)
            return None
        elif any(campiVuotiProfilo):
            msg = "Fields in profile table cannot be empty"
            QMessageBox.warning(QMessageBox(), "Check profile", msg)
            return None
        elif len(self.inputMotion) == 0:
            msg = "Import an input time history before running analysis"
            QMessageBox.warning(QMessageBox(), "Check input", msg)
            return None
        elif not any(outputList):
            msg = "No option selected for output"
            QMessageBox.warning(QMessageBox(), "Check output", msg)
            return None
        return 0


if __name__ == "__main__":
    App = QApplication(sys.argv)
    MainWindow = SRAApp()
    MainWindow.show()
    App.exec_()
