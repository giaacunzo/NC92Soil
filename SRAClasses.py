import pysra
import numpy as np
import pandas as pd
import sympy


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
        currentProfiles = pd.read_excel(filename, sheet_name='Profiles')

        # Converting Vs column to string to prevent further incompatibility
        currentSoils['Vs\n[m/s]'] = currentSoils['Vs\n[m/s]'].astype(str)

        self._rawSoils = currentSoils
        self._rawClusters = currentClusters
        self._rawProfiles = currentProfiles
        self.vsNumber = currentSoils['Vs\n[m/s]'][0].count(';') + 1
        self.clusterNumber = len(currentClusters)
        self.profileNumber = currentProfiles.shape[1]

    def getSoilTable(self, vsIndex):
        soilTable = list()
        for index, row in self._rawSoils.iterrows():
            rowList = list(row)
            rowList[4] = float(rowList[4].split(';')[vsIndex].strip())
            rowList[2] = rowList[2] if not np.isnan(rowList[2]) else None  # From [m]
            rowList[3] = rowList[3] if not np.isnan(rowList[3]) else None  # To [m]
            soilTable.append(rowList)
        return soilTable

    def getClusterInfo(self, clusterIndex):
        permutationTable = list()
        currentCluster = self._rawClusters.iloc[clusterIndex, :]
        bedrockDepth = currentCluster['Bedrock depth\n[m]']
        brickSize = currentCluster['Brick thickness\n[m]']
        soilNames = currentCluster.index[5:]

        for soil in soilNames:
            permutationTable.append([soil, currentCluster[soil]])

        return permutationTable, bedrockDepth, brickSize

    def getInputNames(self, elementIndex, element_type='clusters'):
        if element_type == 'clusters':
            return [inputName.strip() for inputName in self._rawClusters.iloc[elementIndex][4].split(';')]
        else:
            return [inputName.strip() for inputName in self._rawProfiles.iloc[0, elementIndex].split(';')]

    def getElementName(self, elementIndex, element_type='clusters'):
        if element_type == 'clusters':
            currentCluster = self._rawClusters.iloc[elementIndex, :2]
            return "{}-{}".format(*currentCluster)
        else:
            return self._rawProfiles.iloc[:, elementIndex].name

    def getProfileInfo(self, profileIndex):
        profileTable = list()
        VsList = list()
        currentProfile = self._rawProfiles.iloc[:, profileIndex]

        for layer in currentProfile[1:]:
            # Nan check, skip cell if empty
            if isinstance(layer, float) and np.isnan(layer):
                continue

            if layer.count(';') == 2:  # Vs has been specified
                layerName, layerThickness, layerVs = layer.split(';')
            else:
                layerName, layerThickness = layer.split(';')
                layerVs = -1
            profileTable.append([float(layerThickness), layerName])
            VsList.append(float(layerVs))

        return profileTable, VsList


class StochasticAnalyzer:
    """
    Class for handling Stochastic analysis
    """

    def __init__(self, filename):
        currentStochastic = pd.read_excel(filename, sheet_name='Stochastic')
        self.numberIterations = int(currentStochastic['Number of iterations'][0])
        self.correlationMode = currentStochastic['Correlation mode'][0]

        # Check of random seed
        if np.isnan(currentStochastic['Random seed'][0]) or currentStochastic['Random seed'].strip() == "":
            self.randomSeed = None
        else:
            self.randomSeed = currentStochastic['Random seed'][0]

        for index, vsLaw in enumerate(currentStochastic['Vs Law']):
            vsLawObject = sympy.sympify(vsLaw.replace('^', '**'))
            # currentStochastic['Vs Law'][index] = vsLawObject
            currentStochastic.loc[index, 'Vs Law'] = vsLawObject

        self._rawGroups = currentStochastic.iloc[:, :9]
        self._profileDF = pd.DataFrame()
        self._correlationsDF = pd.DataFrame()

    def parseLaw(self, lawIndex, depth):
        """

        :param lawIndex: index of the considered group
        :param depth: the value of depth at which vs mean value must be computed

        :return: the value of the mean Vs for normal distribution
        """

        currentVsLaw = self._rawGroups['Vs Law'][lawIndex]
        return currentVsLaw.subs('x', depth).evalf()

    def generateRndProfile(self):
        np.random.seed(self.randomSeed)

        # Creating basic layered profile as [centroid, thickness, name]
        currentDepth = 0
        layeredProfile = []
        layeredSplitted = []
        for index, group in self._rawGroups.iterrows():
            currentLayeredProfile = []
            groupName = group['Group name']
            groupThickness = np.random.randint(group['Min thickness\n[m]'], group['Max thickness\n[m]'] + 1)

            for counter in range(groupThickness):
                currentCentroid = currentDepth + 1/2
                currentDepth += 1

                # Evaluating mean Vs value from the given relation
                meanVsValue = self.parseLaw(index, currentCentroid)
                stdVsValue = group['Vs Std']
                currentLayeredProfile.append([currentCentroid, 1, groupName, meanVsValue, stdVsValue])

            layeredProfile.extend(currentLayeredProfile)  # Extending final profile list
            layeredSplitted.append(currentLayeredProfile)  # Adding current subprofile to the splitted list

        # Generation of the correlation coefficients list
        correlationCoeffList = []
        if self.correlationMode == 'Single groups':
            for index, group in self._rawGroups.iterrows():
                currentCorrelationName = group['Inter-layer correlation']

                currentCoeff = self.getCorrelationVector(layeredSplitted[index], currentCorrelationName)
                correlationCoeffList.extend(currentCoeff)
        else:
            firstCorrelationName = self._rawGroups['Inter-layer correlation'][0]
            correlationCoeffList = self.getCorrelationVector(layeredProfile, firstCorrelationName)

        # Generating random Vs profile
        finalLayers = []
        lastNormValue = 0
        for index, layer, correlation in zip(range(len(layeredProfile)), layeredProfile, correlationCoeffList):

            # Computing lognormal parameters
            layerMean = float(layer[3])
            layerStd = float(layer[4])

            muLogn = np.log(layerMean**2/(layerStd**2 + layerMean**2)**0.5)
            sigmaLogn = (np.log(layerStd**2/layerMean**2 + 1))**0.5

            # Getting standard normal random variable for i-th layer (epsilon_i)
            standardNormValue = np.random.standard_normal(1)

            if index == 0:  # First layer of entire profile, no correlation needed
                currentNormValue = standardNormValue
                currentVs = np.exp(sigmaLogn*currentNormValue + muLogn)
            else:
                currentNormValue = correlation*lastNormValue + standardNormValue *\
                                   (1 - correlation**2)**0.5
                currentVs = np.exp(sigmaLogn*currentNormValue + muLogn)

            lastNormValue = currentNormValue
            # Appending generated layer in list
            finalLayers.append([layer[1], layer[2], float(currentVs)])

        self.makeProfileColumn(finalLayers)
        self.makeCorrelationColumn(correlationCoeffList)

    def makeProfileColumn(self, finalLayers):
        existingDFSize = len(self._profileDF.columns)
        currentProfileName = "P{}".format(existingDFSize + 1)
        currentProfile = [" "]
        for element in finalLayers:
            currentProfile.append("{};{};{}".format(element[1], element[0], round(element[2], 1)))

        self._profileDF[currentProfileName] = pd.Series(currentProfile)

    def makeCorrelationColumn(self, correlationCoeffList):
        existingDFSize = len(self._correlationsDF.columns)
        currentProfileName = "P{}".format(existingDFSize + 1)
        self._correlationsDF[currentProfileName] = pd.Series(correlationCoeffList)

    def getCorrelationVector(self, currentLayeredProfile, currentLaw):

        if currentLaw.lower().startswith('toro:'):  # Using Toro correlation model
            genericModelName = currentLaw.lower().split('toro:')[1].strip().upper()

            # Creating a copy of layered profile with repeated last layer to simulate perfectly
            # correlated bedrock
            layeredCopy = currentLayeredProfile.copy()
            layeredCopy.append(layeredCopy[-1])
            profileLikeObj = [LayerLikeObj(layer) for layer in layeredCopy]
            toroVelModelObj = pysra.variation.ToroVelocityVariation.generic_model(genericModelName)
            currentCoeff = toroVelModelObj._calc_corr(profileLikeObj)[:-1]

        else:  # Custom equation specified
            currentCoeff = []
            symbolicParse = sympy.sympify(currentLaw)
            for layer in currentLayeredProfile[1:]:
                currentValue = symbolicParse.subs('x', layer[0]).subs('y', layer[1])
                currentCoeff.append(currentValue)

        # Adding uncorrelated first layer at the beginning of the list
        currentCoeff = np.insert(currentCoeff, 0, 0)

        return currentCoeff

    def exportExcel(self, filename):
        """
        Export current profiles DataFrame into a new batch input file

        :param filename: name of the Excel file to save
        :return:
        """
        # Generating soil sheet
        soilDF = pd.DataFrame(columns=['Soil name', 'Unit weight\n[KN/m3]', 'From\n[m]', 'To\n[m]',
                                       'Vs\n[m/s]', 'Degradation curve'])
        for index, group in self._rawGroups.iterrows():
            currentRow = [group['Group name'], group['Unit weight\n[KN/m3]'], "", "", "",
                          group['Degradation curve\nMean']]
            soilDF.loc[index] = currentRow

        with pd.ExcelWriter(filename, mode='w') as writer:
            soilDF.to_excel(writer, sheet_name='Soils', index=False)

        # Generating profiles, correlations and empty clusters sheets
        with pd.ExcelWriter(filename, mode='a') as writer:
            self._profileDF.to_excel(writer, sheet_name='Profiles', index=False)
            self._correlationsDF.to_excel(writer, sheet_name='Correlations', index=False)
            emptyClusterSheet = pd.DataFrame(columns=['Cluster', 'Sub-cluster', 'Bedrock depth\n[m]',
                                             'Brick thickness\n[m]', 'Input files'])
            emptyClusterSheet.to_excel(writer, sheet_name='Clusters', index=False)


class LayerLikeObj:
    """
    Simple class with depth_mid attribute to be compatible with ToroVelocity class in Pysra
    """

    def __init__(self, layerList):
        self.depth_mid = layerList[0]


