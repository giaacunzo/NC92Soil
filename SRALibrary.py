import numpy as np
import pandas as pd
import re
import pysra
import os
import SRAClasses
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


def loadSpectra(fileNames, Damping, Duration, Units):
    conversionFactorList = {'cm/s^2': 1 / 980.665, 'm/s^2': 1 / 9.80665, 'g': 1}
    SpectraDict = dict()

    for filename in fileNames:
        with open(filename) as spectrumFile:
            Contenuto = spectrumFile.read().splitlines()

        eventID = [re.findall(r'NAME:\s*(.*)', linea) for linea in Contenuto[:50] if linea.startswith('NAME')]
        DampingTxt = [re.findall(r'DAMPING:\s*(\d*\.*\d*)', linea) for linea in Contenuto[:50] if linea.startswith('DAMPING')]
        DurationTxt = [re.findall(r'DURATION:\s*(\d*\.*\d*)', linea) for linea in Contenuto[:50] if linea.startswith('DURATION')]
        measureUnits = [re.findall(r'UNITS:\s*(.*)', linea) for linea in Contenuto[:50] if linea.startswith('UNITS')]

        eventID = 'Input Spectrum' if len(eventID) == 0 else eventID[0][0]
        Duration = Duration if len(DurationTxt) == 0 else float(DurationTxt[0][0])
        Damping = Damping if len(DampingTxt) == 0 else float(DampingTxt[0][0])
        measureUnits = Units if len(measureUnits) == 0 else measureUnits[0][0]

        onlyName = os.path.basename(filename)

        conversionFactor = conversionFactorList[measureUnits]

        onlyValuesLines = [line for line in Contenuto if "\t" in line]
        periodVect = [float(line.split('\t')[0]) for line in onlyValuesLines]
        PSAVect = [conversionFactor * float(line.split('\t')[1]) for line in onlyValuesLines]

        origSpectrum = [periodVect, PSAVect]

        periodVect = [value if value != 0 else 1e-5 for value in periodVect]

        # # Interpolating values under 0.02 (50Hz) to prevent numeric errors
        # underThreshold = [(period, value)
        #                   for period, value in zip(periodVect, PSAVect)
        #                   if period < 0.02]
        # firstOverThreshold = [(period, value)
        #                       for period, value in zip(periodVect, PSAVect)
        #                       if period >= 0.02][0]
        #
        # if len(underThreshold) > 0:
        #     newFirstPSA = np.interp(0.02,
        #                             [underThreshold[0][0], firstOverThreshold[0]],
        #                             [underThreshold[0][1], firstOverThreshold[1]])
        #     for period, value in underThreshold:
        #         periodVect.remove(period)
        #         PSAVect.remove(value)
        #
        #     periodVect.insert(0, 0.02)
        #     PSAVect.insert(0, newFirstPSA)

        # Switching to frequencies
        # freqVect = np.array(periodVect[::-1]) ** -1
        # PSAVect = np.array(PSAVect[::-1])
        freqVect = np.array(periodVect) ** -1
        PSAVect = np.array(PSAVect)

        inputMotion = pysra.motion.CompatibleRvtMotion(freqVect, PSAVect,
                                                       duration=Duration, osc_damping=Damping,
                                                       window_len=None)

        # Cutting high frequencies
        minFreq = 0.05 / 2
        maxFreq = 50 * 2

        toTake = [minFreq <= valore <= maxFreq for valore in inputMotion.freqs]
        inputMotion._fourier_amps = inputMotion.fourier_amps[toTake]
        inputMotion._freqs = inputMotion.freqs[toTake]

        SpectraDict[onlyName] = [inputMotion, origSpectrum, eventID, Damping, measureUnits]
    return SpectraDict


def drawTH(inputMotion, asseGrafico):
    asseGrafico.clear()
    asseGrafico.plot(inputMotion.times, inputMotion.accels)
    asseGrafico.set_xlabel('Time [s]')
    asseGrafico.set_ylabel('Acc [g]')
    asseGrafico.grid('on')
    asseGrafico.set_title(inputMotion.description)


def drawSpectrum(inputSpectrum, eventID, asseGrafico, xlog=False, ylog=False):
    asseGrafico.clear()
    asseGrafico.plot(inputSpectrum[0], inputSpectrum[1])
    asseGrafico.set_xlabel('Time [s]')
    asseGrafico.set_ylabel('Acc [g]')
    asseGrafico.grid('on')
    asseGrafico.set_title(eventID)

    # Applying scale to X axis
    if xlog:
        asseGrafico.set_xscale('log')
    else:
        asseGrafico.set_xscale('linear')

    # Applying scale to Y axis
    if ylog:
        asseGrafico.set_yscale('log')
    else:
        asseGrafico.set_yscale('linear')


def drawFAS(inputMotion, eventID, asseGrafico, xlog=False, ylog=False):
    freqs = inputMotion.freqs
    amplitudes = inputMotion.fourier_amps
    asseGrafico.clear()
    asseGrafico.plot(freqs, amplitudes)
    asseGrafico.set_xlabel('Freq [Hz]')
    asseGrafico.set_ylabel('Ampl [g-s]')

    # Applying scale to X axis
    if xlog:
        asseGrafico.set_xscale('log')
    else:
        asseGrafico.set_xscale('linear')

    # Applying scale to Y axis
    if ylog:
        asseGrafico.set_yscale('log')
    else:
        asseGrafico.set_yscale('linear')

    asseGrafico.grid('on')
    asseGrafico.set_title(eventID)
    asseGrafico.set_xlim([0, 100])
    asseGrafico.set_ylim(auto=True)
    

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


def addVariableProperties(soilList, profileList):
    # Expanding soil list
    newSoilList = list()
    for indice, layer in enumerate(profileList):
        currentCentroid = layer[0] + layer[1] / 2
        currentSoilName = layer[2]
        currentSoilDef = [row for row in soilList
                          if (row[0] == currentSoilName and row[2] <= currentCentroid < row[3])][0]
        newSoilName = '{}[{}]'.format(currentSoilDef[0], indice)
        newSoilRow = [newSoilName, currentSoilDef[1], currentSoilDef[-1]]
        newSoilList.append(newSoilRow)

        # Adding velocity to current profile table and changing soil name
        layer[-1] = newSoilName
        layer.append(currentSoilDef[4])

    return newSoilList, profileList


def table2list(soilTable, profileTable):
    soilList = list()
    for riga in range(soilTable.rowCount()):
        currentSoilName = soilTable.item(riga, 0).text() if soilTable.item(riga, 0) is not None else ''
        currentSoilWeight = soilTable.item(riga, 1).text() if soilTable.item(riga, 1) is not None else ''
        currentSoilVs_From = soilTable.item(riga, 2).text() if soilTable.item(riga, 2) is not None else 0
        currentSoilVs_To = soilTable.item(riga, 3).text() if soilTable.item(riga, 3) is not None else 1e4
        currentVs = soilTable.item(riga, 4).text() if soilTable.item(riga, 4) is not None else ''
        currentCurve = soilTable.cellWidget(riga, 5).currentText()

        try:
            soilList.append([currentSoilName, float(currentSoilWeight), float(currentSoilVs_From),
                             float(currentSoilVs_To), float(currentVs), currentCurve])
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
    if outputList[3]:  # Brief report
        outputObject.append(SRAClasses.BriefReportOutput(outputDepth=outputParam[3]))
        # Computing RS at bedrock level
        outputObject.append(pysra.output.ResponseSpectrumOutput(
            freqRange, pysra.output.OutputLocation('outcrop', index=-1),
            osc_damping=0.05))

    finalOutput = pysra.output.OutputCollection(outputObject)

    computationEngine(inputMotion, finalProfile, finalProfile.location('outcrop', index=-1))
    finalOutput(computationEngine)

    # If brief output is checked adds amplification factors using computed RS
    if outputList[3]:
        BriefObject = [briefOutput for briefOutput in finalOutput.outputs
                       if type(briefOutput) == SRAClasses.BriefReportOutput][0]
        RSObjects = [RSObj for RSObj in finalOutput.outputs if type(RSObj) == pysra.output.ResponseSpectrumOutput]
        BriefObject.computeAF(RSObjects)

        # Deleting spectrum at bedrock level to prevent export
        inputSpecIndex = [index for index, value in enumerate(finalOutput.outputs)
                          if type(value) == pysra.output.ResponseSpectrumOutput and
                          value.location.index == -1][0]
        finalOutput.outputs.pop(inputSpecIndex)

    if waitBar is not None:
        waitBar.setLabelText('Running analysis...')
        waitBar.setValue(4)
        App.processEvents()

    return finalOutput


def getBriefValues(computationObject, depth):
    """
    This function computes the fourier antitransform of the wave at a given depth and extract the desired parameters

    :param computationObject:
    :param depth:
    :return:
    """

    motionType = 'outcrop' if depth == 0 else 'within'
    sourceLocation = computationObject.profile.location('outcrop', index=-1)
    outputLocation = computationObject.profile.location('within', index=0)
    currentWaveAccTF = computationObject.calc_accel_tf(sourceLocation, outputLocation)
    maxAcc = computationObject.motion.calc_peak(currentWaveAccTF)

    # Computing fourier amplitude of velocities
    currentWaveVelTF = np.multiply(currentWaveAccTF, computationObject.motion.angular_freqs ** -1)
    maxVel = computationObject.motion.calc_peak(currentWaveVelTF)

    # CONTINUARE DA QUI

