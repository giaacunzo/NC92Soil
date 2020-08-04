import pysra
import numpy as np
import pandas as pd


class BriefReportOutput(pysra.output.Output):
    """
    New class for handling NC92Soil brief report as an output
    """

    def __init__(self, outputDepth, sourceDepth=-1):
        super().__init__()

        self._outputDepth = outputDepth
        self._sourceDepth = sourceDepth
        self._outputLocation = self._sourceLocation = None
        self.maxValuesDict = dict()

    def __call__(self, calc, name=None):
        if self._sourceDepth == -1:
            self._sourceLocation = calc.profile.location('outcrop', index=-1)
        else:
            self._sourceLocation = calc.profile.location('outcrop', depth=self._sourceDepth)

        self._outputLocation = calc.profile.location('within', depth=self._outputDepth)

        currentWaveAccTF = calc.calc_accel_tf(self._sourceLocation, self._outputLocation)
        maxAcc = calc.motion.calc_peak(currentWaveAccTF)

        # Computing fourier amplitude of velocities
        currentWaveVelTF = np.multiply(currentWaveAccTF, calc.motion.angular_freqs ** -1)
        maxVel = calc.motion.calc_peak(currentWaveVelTF) * 981

        # Getting shallow soil name
        shallowSoilName = calc.profile.layers[0].soil_type.name.split('[')[0]

        # Getting input PGA and PGV
        inputPGA = calc.motion.pga  # PGA in g
        inputPGV = calc.motion.pgv * 100  # PGV in cm/s

        vs30 = self.calcVs30(calc)
        self.maxValuesDict = {'PGA ref [g]': inputPGA, 'PGV ref [cm/s]': inputPGV,
                              'Shallow soil name': shallowSoilName, 'VS 30 [m/s]': vs30,
                              'AF_PGA': maxAcc/inputPGA, 'AF_PGV': maxVel/inputPGV}

    def computeAF(self, spectraObj):
        periodRange = [(0.1, 0.5), (0.4, 0.8), (0.7, 1.1)]

        inputSpectrum = [spectrum for spectrum in spectraObj if spectrum.location.index == -1][0]
        outputSpetrum = [spectrum for spectrum in spectraObj if spectrum.location.depth is not None][0]

        inputPSA = inputSpectrum.values[::-1]
        outputPSA = outputSpetrum.values[::-1]
        periods = inputSpectrum.periods[::-1]

        AFDict = dict()
        for bounds in periodRange:
            refName = "AF [{} - {}s]".format(*bounds)
            inRange = [(period, inputValue, outputValue) for period, inputValue, outputValue
                       in zip(periods, inputPSA, outputPSA) if bounds[0] <= period <= bounds[1]]

            if inRange[0][0] != bounds[0]:
                inRange.insert(0, (bounds[0], np.interp(bounds[0], periods, inputPSA),
                                   np.interp(bounds[0], periods, outputPSA)))
            if inRange[-1][0] != bounds[-1]:
                inRange.append((bounds[-1], np.interp(bounds[-1], periods, inputPSA),
                                np.interp(bounds[-1], periods, outputPSA)))

            inRange = np.array(inRange)
            currentAF = np.trapz(inRange[:, 2], inRange[:, 0]) / np.trapz(inRange[:, 1], inRange[:, 0])
            AFDict[refName] = currentAF

        self.maxValuesDict.update(AFDict)
        self.updateValues()

    def calcVs30(self, calc):
        currentThickness = list(calc.profile.thickness).copy()
        currentVelocities = list(calc.profile.initial_shear_vel).copy()
        if max(calc.profile.depth) < 30:
            lastLayerThick = 30 - max(calc.profile.depth)
            currentThickness[-1] = lastLayerThick

        elif max(calc.profile.depth) > 30:
            firstOutIndex = [index for index, depth in enumerate(calc.profile.depth) if depth > 30][0]
            currentThickness = currentThickness[:firstOutIndex-1]
            currentVelocities = currentVelocities[:firstOutIndex]
            lastLayerThick = 30 - sum(currentThickness)
            currentThickness.append(lastLayerThick)

        vs30Value = 30/sum([thickness/Vs for thickness, Vs in
                            zip(currentThickness, currentVelocities)])
        return vs30Value

    def updateValues(self):
        currentDict = self.maxValuesDict
        refs = list()
        values = list()

        for key, value in currentDict.items():
            refs.append(key)
            values.append(value)

        self._refs = refs
        self._values = values


class BatchAnalyzer:
    """
    Class for handling batch analysis
    """
    def __init__(self, filename):
        currentSoils = pd.read_excel(filename, sheet_name='Soils')
        currentClusters = pd.read_excel(filename, sheet_name='Clusters')

        self._rawSoils = currentSoils
        self._rawClusters = currentClusters
        self.vsNumber = currentSoils['Vs\n[m/s]'][0].count(';') + 1
        self.profileNumber = len(currentClusters)

        self.getInputNames(0)

    def getSoilTable(self, vsIndex):
        soilTable = list()
        for index, row in self._rawSoils.iterrows():
            rowList = list(row)
            rowList[4] = float(rowList[4].split(';')[vsIndex].strip())
            rowList[2] = rowList[2] if not np.isnan(rowList[2]) else None  # From [m]
            rowList[3] = rowList[3] if not np.isnan(rowList[3]) else None  # To [m]
            soilTable.append(rowList)
        return soilTable

    def getProfileInfo(self, profileIndex):
        profileTable = list()
        currentProfile = self._rawClusters.iloc[profileIndex, :]
        bedrockDepth = currentProfile['Bedrock depth\n[m]']
        brickSize = currentProfile['Brick thickness\n[m]']
        soilNames = currentProfile.index[5:]

        for soil in soilNames:
            profileTable.append([soil, currentProfile[soil]])

        return profileTable, bedrockDepth, brickSize

    def getInputNames(self, profileIndex):
        return [inputName.strip() for inputName in self._rawClusters.iloc[profileIndex][4].split(';')]

    def getProfileName(self, profileIndex):
        currentProfile = self._rawClusters.iloc[profileIndex, :2]
        return "{}-{}".format(*currentProfile)
