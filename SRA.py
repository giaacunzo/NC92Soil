from PySide2.QtWidgets import *
from PySide2 import QtCore
from SRAmainGUI import Ui_MainWindow
import numpy as np
import sys
import SRALibrary as SRALib


class SRAApp(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()

        # Inizializzazione attributi principali
        self.curveDB = dict()
        self.userModified = True

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
        self.curveDB = SRALib.degradationCurves('CurveDB.xlsx')

    def assignWidgets(self):
        self.pushButton_addSoil.clicked.connect(self.addRow)
        self.pushButton_addProfile.clicked.connect(self.addRow)
        self.pushButton_removeSoil.clicked.connect(self.removeRow)
        self.pushButton_removeProfile.clicked.connect(self.removeRow)
        self.tableWidget_Profile.cellChanged.connect(self.profileChanged)
        self.pushButton_drawProfile.clicked.connect(self.drawProfile)

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
            currentTable.setCellWidget(currentRow, 2, curveComboBox)

        elif tableName.split('_')[-1] == 'Profile':
            if currentRow == 0:  # Inserimento prima riga
                currentTable.setItem(currentRow, 0, QTableWidgetItem('0.00'))
                currentTable.setItem(currentRow, 1, QTableWidgetItem('0.00'))
            else:
                currentDepth = float(currentTable.item(currentRow-1, 0).text()) + \
                               float(currentTable.item(currentRow-1, 1).text())
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

    def drawProfile(self):
        if not getattr(self.graphWidget, 'axes', False):
            self.graphWidget.axes = self.graphWidget.figure.add_subplot(111)

        SRALib.drawProfile(self.tableWidget_Profile, self.graphWidget.axes)
        self.graphWidget.draw()

    def profileChanged(self):
        if not self.userModified:
            return None
        rowNumber = self.sender().rowCount()

        self.userModified = False
        for riga in range(1, rowNumber):
            currentDepth = float(self.sender().item(riga-1, 0).text()) + float(self.sender().item(riga-1, 1).text())
            self.sender().setItem(riga, 0, QTableWidgetItem(str(currentDepth)))
        self.userModified = True


if __name__ == "__main__":
    App = QApplication(sys.argv)
    MainWindow = SRAApp()
    MainWindow.show()
    App.exec_()
