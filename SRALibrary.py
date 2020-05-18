import numpy as np
import pandas as pd
import re
import pysra
import os
from itertools import permutations


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


def calcPermutations(profileList, returnpermutations=False):
    totalBricks = len(profileList)
    soilBricks = [strato[1] for strato in profileList]
    uniqueSoil = sorted(set(soilBricks))

    totalHeight = sum([element[0] for element in profileList])
    profileSum = list()
    for soil in uniqueSoil:
        profileSum.append([soil, sum([element[0] for element in profileList if element[1] == soil])])

    percentageList = [(value[0], 100*value[1]/totalHeight) for value in profileSum]

    # Computing real number of combinations
    profileTuple = [tuple(elemento) for elemento in profileList]
    uniqueBricks = set(profileTuple)
    bricksOccurrence = [profileTuple.count(elemento) for elemento in uniqueBricks]

    percStrings = ["{}: {:.1f}%".format(*value) for value in percentageList]
    finalString = " , ".join(percStrings)

    denomFact = np.prod([np.math.factorial(occurrence) for occurrence in bricksOccurrence])
    numFact = np.math.factorial(totalBricks)
    if not returnpermutations:
        return numFact/denomFact, finalString, len(uniqueBricks)
    else:
        permutationList = list(set(permutations(profileTuple)))
        return permutationList


def drawProfile(profileList, asseGrafico, lineLength=0):
    asseGrafico.clear()
    currentThickness = currentDepth = 0

    profileList = addDepths(profileList)

    if lineLength == 0:
        finalDepth = profileList[-1][0] + profileList[-1][1]
        lineLength = 3 * finalDepth

    # Assegna un colore agli strati
    colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    uniqueNames = sorted(set([strato[2] for strato in profileList]))
    colorDict = dict()
    for indice, nome in enumerate(uniqueNames):
        colorDict[nome] = colorList[indice]

    for strato in profileList:
        currentDepth = strato[0]
        currentThickness = strato[1]
        currentSoilName = strato[2]
        currentCoord = [(0, -currentDepth), (lineLength, -currentDepth),
                        (lineLength, -(currentDepth + currentThickness)),
                        (0, -(currentDepth + currentThickness)), (0, -currentDepth)]
        CoordX, CoordY = zip(*currentCoord)
        asseGrafico.fill(CoordX, CoordY, colorDict[currentSoilName], linewidth=2.0, alpha=0.2, edgecolor='k')

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


def loadTH(fileNames, Fs, Units):
    conversionFactorList = {'cm/s^2': 1 / 980.665, 'm/s^2': 1 / 9.80665, 'g': 1}

    inputMotionsList = list()
    THDict = dict()

    for fileName in fileNames:
        with open(fileName) as accFile:
            Contenuto = accFile.read().splitlines()

        # Check di event ID, timestep e unità di misura
        eventID = [re.findall(r'EVENT_ID: (.*)', linea) for linea in Contenuto[:50] if linea.startswith('EVENT_ID')]
        timeStep = [re.findall(r'\d+\.*\d*', linea) for linea in Contenuto[:50] if linea.startswith('SAMPLING')]
        measureUnits = [re.findall(r'UNITS: (.*)', linea) for linea in Contenuto[:50] if linea.startswith('UNITS')]

        eventID = 'Input Motion' if len(eventID) == 0 else eventID[0][0]
        timeStep = 1 / Fs if len(timeStep) == 0 else float(timeStep[0][0])
        measureUnits = Units if len(measureUnits) == 0 else measureUnits[0][0]

        onlyName = os.path.basename(fileName)

        conversionFactor = conversionFactorList[measureUnits]
        numericAcc = [conversionFactor * float(valore) for valore in Contenuto if
                      valore.replace('.', '').replace('-', '').isdigit()]
        inputMotion = pysra.motion.TimeSeriesMotion(
            os.path.split(fileName)[-1], eventID, timeStep, numericAcc)

        THDict[onlyName] = [inputMotion, eventID, timeStep, measureUnits]

    return THDict


def drawTH(inputMotion, asseGrafico):
    asseGrafico.clear()
    asseGrafico.plot(inputMotion.times, inputMotion.accels)
    asseGrafico.set_xlabel('Time [s]')
    asseGrafico.set_ylabel('Acc [g]')
    asseGrafico.grid('on')
    asseGrafico.set_title(inputMotion.description)


def makeBricks(profileTable, brickSize):
    totalRows = profileTable.rowCount()
    newProfile = list()
    for riga in range(totalRows):
        currentThickness = float(profileTable.item(riga, 1).text())
        currentNameCell = profileTable.item(riga, 2)
        # currentVelocity = float(profileTable.item(riga, 3).text()) if profileTable.item(riga, 3) is not None else 0
        currentName = profileTable.item(riga, 2).text() if currentNameCell is not None else 'N/D'
        numberBricks = int(np.floor(currentThickness/brickSize))

        for strato in range(numberBricks):
            if strato != numberBricks - 1:
                stratoThickness = brickSize
            else:
                stratoThickness = currentThickness - brickSize*(numberBricks - 1)
            newProfile.append([stratoThickness, currentName])
    return newProfile


def addDepths(profileList):
    currentDepth = 0
    newProfile = list()
    for strato in profileList:
        newProfile.append([currentDepth] + list(strato))
        currentDepth += strato[0]
    return newProfile


def table2list(soilTable, profileTable):
    soilList = list()
    for riga in range(soilTable.rowCount()):
        currentSoilName = soilTable.item(riga, 0).text() if soilTable.item(riga, 0) is not None else ''
        currentSoilWeight = soilTable.item(riga, 1).text() if soilTable.item(riga, 1) is not None else ''
        currentSoilVs_From = soilTable.item(riga, 2).text() if soilTable.item(riga, 2) is not None else ''
        currentSoilVs_To = soilTable.item(riga, 3).text() if soilTable.item(riga, 3) is not None else 1e4
        currentVs = soilTable.item(riga, 4).text() if soilTable.item(riga, 4) is not None else ''
        currentCurve = soilTable.cellWidget(riga, 5).currentText()

        try:
            soilList.append([currentSoilName, float(currentSoilWeight), currentSoilVs_From, currentSoilVs_To,
                             float(currentVs), currentCurve])
        except ValueError:
            soilList = 'SoilNan'
            break

    profileList = list()
    for riga in range(profileTable.rowCount()):
        currentDepth = profileTable.item(riga, 0).text() if profileTable.item(riga, 0) is not None else ''
        currentThickness = profileTable.item(riga, 1).text() if profileTable.item(riga, 0) is not None else ''
        currentSoilName = profileTable.item(riga, 2).text() if profileTable.item(riga, 2) is not None else ''

        try:
            profileList.append([float(currentThickness), currentSoilName])
        except ValueError:
            profileList = 'ProfileNan'
            break

    return soilList, profileList


def runAnalysis(inputMotion, soilList, profileList, analysisDict, graphWait=None):
    if graphWait is not None:
        waitBar = graphWait[0]
        App = graphWait[1]

        waitBar.setLabelText('Building layers...')
        waitBar.setValue(1)
        App.processEvents()
    else:
        waitBar = App = None

    # Soil table analysis
    curveDict = analysisDict['CurveDB']
    soilDict = dict()
    for riga in soilList:
        currentSoilName, currentSoilWeight, currentCurveName = riga
        currentCurve = curveDict[currentCurveName]
        currentNonLinearityG = pysra.site.NonlinearProperty('', currentCurve[:, 0],
                                                            currentCurve[:, 1], param='mod_reduc')
        currentNonLinearityD = pysra.site.NonlinearProperty('', currentCurve[:, 0],
                                                            currentCurve[:, 2], param='damping')
        currentSoilObj = pysra.site.SoilType(currentSoilName, float(currentSoilWeight), mod_reduc=currentNonLinearityG,
                                             damping=currentNonLinearityD)
        soilDict[currentSoilName] = currentSoilObj

    # Defining bedrock soil
    BedrockData = analysisDict['Bedrock']
    BedrockSoil = pysra.site.SoilType('Bedrock', float(BedrockData[0]), None, float(BedrockData[2]) / 100)

    if waitBar is not None:
        waitBar.setLabelText('Building profile...')
        waitBar.setValue(2)
        App.processEvents()

    # Creating profile
    profileLayer = list()
    for riga in profileList:
        currentDepth, currentThickness, currentSoilName, currentVelocity = riga
        profileLayer.append(pysra.site.Layer(soilDict[currentSoilName], currentThickness, currentVelocity))

    # Adding bedrock layer
    profileLayer.append(pysra.site.Layer(BedrockSoil, 0, float(BedrockData[1])))
    finalProfile = pysra.site.Profile(profileLayer)

    discretizationParam = analysisDict['Discretization']
    if discretizationParam[0] > 0:
        finalProfile = finalProfile.auto_discretize(*discretizationParam)

    # Running analysis
    if waitBar is not None:
        waitBar.setLabelText('Running analysis...')
        waitBar.setValue(3)
        App.processEvents()

    strainRatio, errorTolerance, maxIterations = analysisDict['LECalcOptions']
    computationEngine = pysra.propagation.EquivalentLinearCalculator(strainRatio, errorTolerance, maxIterations)
    freqRange = np.logspace(-1, 2, num=91)  # Periods between 0.01 and 10 seconds

    # Building output object
    outputList = analysisDict['OutputList']
    outputParam = analysisDict['OutputParam']

    outputObject = list()
    if outputList[0]:  # Response spectrum
        outputObject.append(pysra.output.ResponseSpectrumOutput(
            freqRange, pysra.output.OutputLocation('outcrop', depth=outputParam[0]),
            osc_damping=0.05))
    if outputList[1]:  # Acceleration
        outputObject.append(pysra.output.AccelerationTSOutput(
            pysra.output.OutputLocation('outcrop', depth=outputParam[1])))
    if outputList[2]:  # Strains
        outputObject.append(pysra.output.StrainTSOutput(
            pysra.output.OutputLocation('within', depth=outputParam[2]), in_percent=True))

    finalOutput = pysra.output.OutputCollection(outputObject)

    computationEngine(inputMotion, finalProfile, finalProfile.location('outcrop', index=-1))
    finalOutput(computationEngine)

    if waitBar is not None:
        waitBar.setLabelText('Running analysis...')
        waitBar.setValue(4)
        App.processEvents()

    return finalOutput
