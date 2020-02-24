import numpy as np
import pandas as pd


def degradationCurves(fileName):
    """
    Caricamento delle curve di decadimento da file Excel

    INPUT
    fileName:       nome del file Excel contenente le curve di decadimento

    :return:
    curveDict:  dizionario contenente la coppia {nomecurva:valori}
    """

    curveDB = pd.ExcelFile(fileName)
    curveDict = dict()

    for curva in curveDB.sheet_names:
        curveDict[curva] = curveDB.parse(curva).values

    return curveDict


def drawProfile(profileTable, asseGrafico, lineLength=0):
    asseGrafico.clear()
    totalRows = profileTable.rowCount()

    if lineLength == 0:
        finalDepth = float(profileTable.item(totalRows-1, 0).text()) + float(profileTable.item(totalRows-1, 1).text())
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

        asseGrafico.text(lineLength/2, -(currentDepth + currentThickness/2), currentSoilName)
        asseGrafico.axis('equal')

    # Drawing bedrock
    bedrockDepth = currentThickness + currentDepth
    bedRockCoord = [(0, -bedrockDepth), (lineLength, -bedrockDepth),
                    (lineLength, -(bedrockDepth*3/2)), (0, -bedrockDepth*3/2),
                    (0, -bedrockDepth)]
    CoordX, CoordY = zip(*bedRockCoord)
    asseGrafico.fill(CoordX, CoordY, 'tab:gray', linewidth=2.0, alpha=0.1)
    asseGrafico.text(lineLength/2, -bedrockDepth*5/4, 'Bedrock')
