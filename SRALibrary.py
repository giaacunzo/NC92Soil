import numpy as np
import pandas as pd
import re
import pysra
import os


def degradationCurves(fileName):
    """
    Caricamento delle curve di decadimento da file Excel

    INPUT
    fileName:       nome del file Excel contenente le curve di decadimento (Gamma e D in percentuale, G/Gmax decimale)

    :return:
    curveDict:  dizionario contenente la coppia {nomecurva:valori}
    """

    curveDB = pd.ExcelFile(fileName)
    curveDict = dict()

    for curva in curveDB.sheet_names:
        currentValues = curveDB.parse(curva).values
        currentValues[:, 0] = currentValues[:, 0] / 100
        currentValues[:, 2] = currentValues[:, 2] / 100
        curveDict[curva] = currentValues
        pass

    return curveDict


def drawProfile(profileTable, asseGrafico, lineLength=0):
    asseGrafico.clear()
    totalRows = profileTable.rowCount()

    currentThickness = currentDepth = 0

    if lineLength == 0:
        finalDepth = float(profileTable.item(totalRows - 1, 0).text()) + float(
            profileTable.item(totalRows - 1, 1).text())
        lineLength = 3 * finalDepth

    for riga in range(totalRows):
        currentDepth = float(profileTable.item(riga, 0).text())
        currentThickness = float(profileTable.item(riga, 1).text())
        currentCoord = [(0, -currentDepth), (lineLength, -currentDepth),
                        (lineLength, -(currentDepth + currentThickness)),
                        (0, -(currentDepth + currentThickness)), (0, -currentDepth)]
        CoordX, CoordY = zip(*currentCoord)
        asseGrafico.fill(CoordX, CoordY, linewidth=2.0, alpha=0.2)
        currentSoilName = profileTable.item(riga, 2)

        if currentSoilName is not None:
            currentSoilName = currentSoilName.text()
        else:
            currentSoilName = 'N/D'

        asseGrafico.text(lineLength / 2, -(currentDepth + currentThickness / 2), currentSoilName)
        asseGrafico.axis('equal')

    # Drawing bedrock
    bedrockDepth = currentThickness + currentDepth
    bedRockCoord = [(0, -bedrockDepth), (lineLength, -bedrockDepth),
                    (lineLength, -(bedrockDepth * 3 / 2)), (0, -bedrockDepth * 3 / 2),
                    (0, -bedrockDepth)]
    CoordX, CoordY = zip(*bedRockCoord)
    asseGrafico.fill(CoordX, CoordY, 'tab:gray', linewidth=2.0, alpha=0.1)
    asseGrafico.text(lineLength / 2, -bedrockDepth * 5 / 4, 'Bedrock')


def loadTH(fileName, Fs, Units):
    conversionFactorList = {'cm/s^2': 1 / 980.665, 'm/s^2': 1 / 9.80665, 'g': 1}
    with open(fileName) as accFile:
        Contenuto = accFile.read().splitlines()

    # Check di event ID, timestep e unità di misura
    eventID = [re.findall(r'EVENT_ID: (.*)', linea) for linea in Contenuto[:50] if linea.startswith('EVENT_ID')]
    timeStep = [re.findall(r'\d+\.*\d*', linea) for linea in Contenuto[:50] if linea.startswith('SAMPLING')]
    measureUnits = [re.findall(r'UNITS: (.*)', linea) for linea in Contenuto[:50] if linea.startswith('UNITS')]

    eventID = 'Input Motion' if len(eventID) == 0 else eventID[0][0]
    timeStep = 1/Fs if len(timeStep) == 0 else float(timeStep[0][0])
    measureUnits = Units if len(measureUnits) == 0 else measureUnits[0][0]

    conversionFactor = conversionFactorList[measureUnits]
    numericAcc = [conversionFactor * float(valore) for valore in Contenuto if
                  valore.replace('.', '').replace('-', '').isdigit()]
    inputMotion = pysra.motion.TimeSeriesMotion(os.path.split(fileName)[-1], eventID, timeStep, numericAcc)

    return inputMotion, measureUnits


def drawTH(inputMotion, asseGrafico):
    asseGrafico.plot(inputMotion.times, inputMotion.accels)
    asseGrafico.set_xlabel('Time [s]')
    asseGrafico.set_ylabel('Agg [g]')
    asseGrafico.grid()
    asseGrafico.set_title(inputMotion.description)


def runAnalysis(inputMotion, soilTable, profileTable, analysisDict):
    # Soil table analysis
    curveDict = analysisDict['CurveDB']
    soilDict = dict()
    for riga in range(soilTable.rowCount()):
        currentSoilName = soilTable.item(riga, 0).text()
        currentSoilWeight = soilTable.item(riga, 1).text()
        currentCurve = curveDict[soilTable.cellWidget(riga, 2).currentText()]

        currentNonLinearityG = pysra.site.NonlinearProperty('', currentCurve[:, 0],
                                                            currentCurve[:, 1], param='mod_reduc')
        currentNonLinearityD = pysra.site.NonlinearProperty('', currentCurve[:, 0],
                                                            currentCurve[:, 2], param='damping')
        currentSoilObj = pysra.site.SoilType(currentSoilName, float(currentSoilWeight), mod_reduc=currentNonLinearityG,
                                             damping=currentNonLinearityD)
        soilDict[currentSoilName] = currentSoilObj

    # Defining bedrock soil
    BedrockData = analysisDict['Bedrock']
    BedrockSoil = pysra.site.SoilType('Bedrock', float(BedrockData[0]), None, float(BedrockData[2]/100))

    # Creating profile
    profileLayer = list()
    for riga in range(profileTable.rowCount()):
        currentThickness = float(profileTable.item(riga, 0).text())
        currentSoilName = profileTable.item(riga, 2).text()
        currentVelocity = float(profileTable.item(riga, 3))

        profileLayer.append(pysra.site.Layer(soilDict[currentSoilName], currentThickness, currentVelocity))

    # Adding bedrock layer
    profileLayer.append(pysra.site.Layer(BedrockSoil, 0, float(BedrockData[1])))

    finalProfile = pysra.site.Profile(profileLayer)