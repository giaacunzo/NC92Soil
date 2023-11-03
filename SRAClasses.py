import pysra
import numpy as np
import pandas as pd
import sympy
import SRALibrary as NCLib
import os
from scipy.signal import find_peaks
from PySide6 import QtCore

# DECLARING CONSTANTS
INF_DEPTH = 10000


class BriefReportOutput(pysra.output.Output):
    """
    New class for handling NC92Soil brief report as an output
    """

    def __init__(self, outputDepth, sourceDepth=-1, source_interface='outcrop', output_interface='outcrop'):
        super().__init__()

        self._outputDepth = outputDepth
        self._sourceDepth = sourceDepth
        self._source_interface = source_interface
        self._output_interface = output_interface
        self._outputLocation = self._sourceLocation = None
        self.maxValuesDict = dict()

    def __call__(self, calc, name=None):
        if self._sourceDepth == -1:
            self._sourceLocation = calc.profile.location(self._source_interface, index=-1)
        else:
            self._sourceLocation = calc.profile.location(self._source_interface, depth=self._sourceDepth)

        self._outputLocation = calc.profile.location(self._output_interface, depth=self._outputDepth)

        currentWaveAccTF = calc.calc_accel_tf(self._sourceLocation, self._outputLocation)
        maxAcc = calc.motion.calc_peak(currentWaveAccTF)

        # if outcrop is requested for AF computation, within is computed for F0
        if self._source_interface == 'outcrop':
            bedrock_within = calc.profile.location('within', index=-1)
            currentWaveAccTF_F0 = calc.calc_accel_tf(bedrock_within, self._outputLocation)
        else:
            currentWaveAccTF_F0 = currentWaveAccTF

        # Computinf F0
        upper_bound = 60
        freq_vect = calc.motion.freqs[calc.motion.freqs <= upper_bound]
        # peaks_index, _ = find_peaks(
        #     pysra.output.ko_smooth(freq_vect, calc.motion.freqs, abs(currentWaveAccTF), upper_bound)
        # )
        amplitude_vect = abs(currentWaveAccTF_F0[calc.motion.freqs <= upper_bound])
        peaks_index, _ = find_peaks(amplitude_vect)
        if len(peaks_index) > 0:
            F0_first = freq_vect[peaks_index[0]]
            F0_max_pos = peaks_index[amplitude_vect[peaks_index].argmax()]
            F0_max = freq_vect[F0_max_pos]
        else:
            F0_max = ''
            F0_first = ''

        # Computing fourier amplitude of velocities
        currentWaveVelTF = np.multiply(currentWaveAccTF, calc.motion.angular_freqs ** -1)
        maxVel = calc.motion.calc_peak(currentWaveVelTF) * 981

        # Getting shallow soil name
        shallowSoilName = calc.profile.layers[0].soil_type.name.split('[')[0]

        # Getting input PGA and PGV
        inputPGA = calc.motion.pga  # PGA in g
        inputPGV = calc.motion.pgv * 100  # PGV in cm/s

        # Computing H, VsH and Vs30
        H, vsH, vs30, vsEq = self.calcVs30(calc)

        # Computing max final error
        final_error = max([layer.max_error for layer in calc.profile.layers])

        self.maxValuesDict = {'PGA ref [g]': inputPGA, 'PGV ref [cm/s]': inputPGV,
                              'Final error [%]': final_error,
                              'H [m]': H, 'Vs H [m/s]': vsH,
                              'VS 30 [m/s]': vs30, 'VS Eq [m/s]': vsEq, 'Shallow soil name': shallowSoilName,
                              'F0 - First [Hz]': F0_first, 'F0 - Max [Hz]': F0_max,
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

    @staticmethod
    def calcVs30(calc):
        currentThickness = list(calc.profile.thickness).copy()
        currentVelocities = list(calc.profile.initial_shear_vel).copy()

        # Computing VsH
        H = sum(currentThickness)
        vsHValue = H / sum([thickness / Vs for thickness, Vs in
                            zip(currentThickness, currentVelocities)])

        # Computing Vs30
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

        if H >= 30:
            vsEqValue = vs30Value
        else:
            vsEqValue = vsHValue

        return H, vsHValue, vs30Value, vsEqValue

    def updateValues(self):
        currentDict = self.maxValuesDict
        refs = list()
        values = list()

        for key, value in currentDict.items():
            refs.append(key)
            values.append(value)

        self._refs = refs
        self._values = values


class TransferFunctionOutput(pysra.output.Output):

    def __init__(self, outputDepth, sourceDepth=-1, source_interface='within', output_interface='outcrop',
                 upper_bound=60, tag=None, only_FFT=False):
        super().__init__()

        self._outputDepth = outputDepth
        self._sourceDepth = sourceDepth
        self._upper_bound = upper_bound
        self._source_interface = source_interface
        self._output_interface = output_interface
        self._outputLocation = self._sourceLocation = None
        self.tag = tag
        self._only_FFT = only_FFT

    def __call__(self, calc, name=None):
        if self._only_FFT:
            # Only signal FFT is required
            freq_vect = np.array([freq for freq in calc.motion.freqs if freq <= self._upper_bound])
            amplitude_vect = np.array([value for freq, value in zip(calc.motion.freqs, calc.motion.fourier_amps)
                                       if freq <= self._upper_bound])
            self._refs = np.real(freq_vect)
            self._values = np.abs(amplitude_vect)
            return None

        if self._sourceDepth == -1:
            self._sourceLocation = calc.profile.location(self._source_interface, index=-1)
        else:
            self._sourceLocation = calc.profile.location(self._source_interface, depth=self._sourceDepth)

        self._outputLocation = calc.profile.location(self._output_interface, depth=self._outputDepth)

        currentTF = calc.calc_accel_tf(lin=self._sourceLocation, lout=self._outputLocation)
        freq_vect = np.array([freq for freq in calc.motion.freqs if freq <= self._upper_bound])
        amplitude_vect = np.array([value for freq, value in zip(calc.motion.freqs, currentTF)
                                   if freq <= self._upper_bound])

        self._refs = np.real(freq_vect)
        self._values = np.abs(amplitude_vect)


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
        self.filename = filename

        # Check for Std column
        if 'Curve Std' not in self._rawSoils.columns:
            self._rawSoils['Curve Std'] = np.nan

    def getSoilTable(self, vsIndex):
        soilTable = list()
        for index, row in self._rawSoils.iterrows():
            rowList = list(row)
            rowList[4] = float(rowList[4].split(';')[vsIndex].strip())
            rowList[2] = rowList[2] if not np.isnan(rowList[2]) else None  # From [m]
            rowList[3] = rowList[3] if not np.isnan(rowList[3]) else None  # To [m]
            soilTable.append(rowList)
        return soilTable

    def getDegradationCurveStd(self):
        subSection = self._rawSoils[['Soil name', 'Curve Std']]

        return [list(row) for index, row in subSection.iterrows()]

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
        currentBedrockVs = currentProfile[1].split(':')[-1].strip()
        for layer in currentProfile[2:]:
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

        return profileTable, VsList, currentBedrockVs


class StochasticAnalyzer:
    """
    Class for handling Stochastic analysis
    """

    def __init__(self, filename):
        currentStochastic = pd.read_excel(filename, sheet_name='Stochastic', dtype={'ID CODE': str}).\
            dropna(subset=['ID CODE'])
        currentGroupsDef = pd.read_excel(filename, sheet_name='Group definitions').\
            dropna(subset=['Group name'])

        # Stripping ID CODE strings
        currentStochastic['ID CODE'] = [orig_name.strip() for orig_name in currentStochastic['ID CODE']]
        self.numberIterations = int(currentStochastic['Number of iterations'][0])
        self.correlationMode = currentStochastic['Correlation mode'][0]
        self.vsLimit = currentStochastic['Bedrock Vs\n[m/s]'][0]
        self.inputFiles = currentStochastic['Input files'][0]
        self.maxDepth = currentStochastic['Max depth\n[m]'][0]
        self.vsPolicy = currentStochastic['Bedrock Vs policy'][0]
        self.harmonicMeanDepth = int(currentStochastic['Harmonic mean depth\n[m]'][0])

        # Check of random seed
        if np.isnan(currentStochastic['Random seed'][0]) or not isinstance(currentStochastic['Random seed'][0], float):
            self.randomSeed = None
        else:
            self.randomSeed = int(currentStochastic['Random seed'][0])

        # Merging with "Group definition" sheet
        # currentStochastic = self.merge_sheets(currentStochastic, currentGroupsDef)

        # Parsing the vs law expression
        # for index, vsLaw in enumerate(currentStochastic['Vs Law']):
        #     vsLawObject = sympy.sympify(vsLaw.replace('^', '**'))
        #     currentStochastic.loc[index, 'Vs Law'] = vsLawObject

        self.idList = self.make_unique_list(currentStochastic['ID CODE'])
        self._allData = currentStochastic
        self._allSoils = self.makeSoils(currentGroupsDef)
        self._rawGroups = pd.DataFrame()
        self._profileDF = pd.DataFrame()
        self._correlationsDF = pd.DataFrame()

        np.random.seed(self.randomSeed)

    def check_input_file(self):
        # Checking for all soils to be defined in Soils sheet
        all_groups = set(self._allData['Group name'])
        unique_soils = set([element.split('[')[0] for element in self._allSoils['Group name']])
        not_found_soils = list()
        for group in all_groups:
            current_soils = [element.strip() for element in group.split(',')]
            not_found_soils.extend([element for element in current_soils
                                    if element != "" and element not in unique_soils])
        not_found_soils = list(set(not_found_soils))

        # Checking for all soils to have minimum and maximum thickness
        not_defined_thickness = self._allSoils[(self._allSoils['Min thickness\n[m]'].astype(str).str.strip() == '') |
                                               (self._allSoils['Max thickness\n[m]'].astype(str).str.strip() == '')]\
            ['Orig name']

        not_defined_thickness = list(not_defined_thickness.values)

        # Checking for all soils to be continuously defined
        grouped_soils = self._allSoils.groupby(by='Orig name')
        discontinuous_soils = []
        for soil, frame in grouped_soils:
            expected_lower_bound = 0
            for index, row in frame.reset_index().iterrows():
                if row['From\n[m]'] == expected_lower_bound:
                    expected_lower_bound = row['To\n[m]']
                else:
                    discontinuous_soils.append(row['Orig name'])
                    break

                if index == len(frame) - 1 and row['To\n[m]'] != INF_DEPTH:
                    discontinuous_soils.append(row['Orig name'])
                    break

        return not_found_soils, not_defined_thickness, discontinuous_soils

    @staticmethod
    def make_unique_list(orig_list):
        unique_list = list()

        for value in orig_list:
            if value not in unique_list:
                unique_list.append(value)

        return unique_list

    @staticmethod
    def makeSoils(soil_sheet):

        for index, row in soil_sheet.iterrows():
            if np.isnan(row['From\n[m]']):
                soil_sheet.loc[index, 'From\n[m]'] = 0
            if np.isnan(row['To\n[m]']):
                soil_sheet.loc[index, 'To\n[m]'] = INF_DEPTH

        # Parse the Vs laws
        for index, vsLaw in enumerate(soil_sheet['Vs Law']):
            vsLaw = vsLaw.replace('^', '**') if isinstance(vsLaw, str) else vsLaw
            # vsLawObject = sympy.sympify(vsLaw.replace('^', '**'))
            vsLawObject = sympy.sympify(vsLaw)
            soil_sheet.loc[index, 'Vs Law'] = vsLawObject

        # new_sheet = pd.DataFrame(columns=soil_sheet.columns)
        new_sheet = pd.DataFrame()
        soils_set = set()
        unique_soils = [soil for soil in soil_sheet['Group name']
                        if soil not in soils_set and soils_set.add(soil) is None]

        for soil in unique_soils:
            current_defs = soil_sheet[soil_sheet['Group name'] == soil]
            current_defs.reset_index(inplace=True)
            for index, row in current_defs.iterrows():
                row['Orig name'] = soil
                row['Group name'] = "{}[{}]".format(soil, index) if len(current_defs) > 1 else soil
                # new_sheet = new_sheet.append(row, ignore_index=True)
                new_sheet = pd.concat([new_sheet, row.to_frame().T], ignore_index=True)

        return new_sheet

    @staticmethod
    def merge_sheets(currentStochastic, currentGroupsDef):
        for index, row in currentStochastic.iterrows():
            # Checking fields
            fields_list = currentGroupsDef.columns[1:]
            for field in fields_list:
                if isinstance(row[field], float) and np.isnan(row[field]):
                    currentGroupName = row['Group name']
                    currentGroupRow = currentGroupsDef[currentGroupsDef['Group name'] == currentGroupName]
                    currentStochastic.loc[index, field] = currentGroupRow[field].values
        return currentStochastic

    def resetDataFrames(self):
        self._profileDF = pd.DataFrame()
        self._correlationsDF = pd.DataFrame()
        self._rawGroups = pd.DataFrame()

    def updateByID(self, id_code):
        self.resetDataFrames()
        self._rawGroups = self._allData[self._allData['ID CODE'] == id_code]
        self._rawGroups.reset_index(inplace=True)

    def parseLaw(self, group_name, depth, current_law=None):
        """

        :param group_name: name of current soil
        :param depth: the value of depth at which vs mean value must be computed

        :return: the value of the mean Vs for normal distribution
        """
        current_def = self._allSoils[(self._allSoils['Orig name'] == group_name) &
                                     (self._allSoils['From\n[m]'] <= depth) &
                                     (depth < self._allSoils['To\n[m]'])]
        current_soil = current_def['Group name'].values[0]
        if not current_law:
            currentVsLaw = current_def['Vs Law'].values[0]
            current_sigma = current_def['Sigma logn'].values[0]
            currentMean = currentVsLaw.subs('H', depth).evalf()
        else:  # A new law tuple is given (law, std)
            currentMean = sympy.sympify(current_law[0]).subs('H', depth)
            if isinstance(current_law[1], (float, int)) and np.isnan(current_law[1]):
                current_sigma = current_def['Sigma logn'].values[0]
            else:
                current_sigma = current_law[1]

        return current_soil, currentMean, current_sigma

    def parseCorrelation(self, group_name):
        return self._allSoils[self._allSoils['Group name'] == group_name]['Inter-layer correlation'].values[0], \
               self._allSoils[self._allSoils['Group name'] == group_name]['Maximum depth\n[m]'].values[0]

    @staticmethod
    def randomGroup(groupNames, separator):
        random_index = np.random.randint(0, len(groupNames.split(separator)))
        return random_index, groupNames.split(separator)[random_index].strip()

    def generateRndProfile(self):
        # Creating basic layered profile as [centroid, thickness, name]
        separator = ','
        currentDepth = 0
        layeredProfile = []
        layeredSplitted = []
        totalDepth = 0
        bedrock_extended = False  # Set as default, will be changed further if required
        for index, group in self._rawGroups.iterrows():
            currentLayeredProfile = []
            groupName = group['Group name']

            # Choosing a random group if separator is found
            if separator in groupName:
                # Cleaning string if given in tuple format
                groupName = self.randomGroup(groupName, separator)
                group_orig_name = groupName[1]
            else:
                group_orig_name = groupName

            # Setting default minimum thickness for the group if not specified
            if np.isnan(group['Min thickness\n[m]']) or int(group['Min thickness\n[m]']) == -1:
                default_group = self._allSoils[self._allSoils['Orig name'] == group_orig_name]['Min thickness\n[m]']
                minThickness = default_group.values[0]
            else:
                minThickness = group['Min thickness\n[m]']

            # Setting default maximum thickness for the group if not specified
            if np.isnan(group['Max thickness\n[m]']) or int(group['Max thickness\n[m]']) == -1:
                default_group = self._allSoils[self._allSoils['Orig name'] == group_orig_name]['Max thickness\n[m]']
                maxThickness = default_group.values[0]
            else:
                maxThickness = group['Max thickness\n[m]']

            # Check for minimum and maximum order
            if minThickness > maxThickness:
                old_min = minThickness
                minThickness = maxThickness
                maxThickness = old_min

            # If last group and bedrock extension is Y, depth is extended until defined limit
            if index == len(self._rawGroups) - 1:
                if isinstance(group['Bedrock extension'], str) and \
                        group['Bedrock extension'].lower() == 'y':  # Bedrock extension

                    groupThickness = self.maxDepth - totalDepth
                    # VsMin = np.nan
                    # VsMax = np.nan

                    # Considering defined max and minimum values in bedrock extension
                    VsMin = group['Vs min\n[m/s]']
                    VsMax = group['Vs max\n[m/s]']

                    bedrock_extended = True
                else:  # Extract a random thickness as usual
                    groupThickness = np.random.randint(minThickness, maxThickness + 1)
                    VsMin = group['Vs min\n[m/s]']
                    VsMax = group['Vs max\n[m/s]']
            else:
                groupThickness = np.random.randint(minThickness, maxThickness + 1)
                totalDepth += groupThickness
                VsMin = group['Vs min\n[m/s]']
                VsMax = group['Vs max\n[m/s]']

            for counter in range(int(groupThickness)):
                currentCentroid = currentDepth + 1/2
                currentDepth += 1

                # Evaluating mean Vs value from the given relation
                if isinstance(groupName, tuple):  # Randomly chosen GT group, associating corresponding law
                    try:
                        lawValue = [law.strip() for law in group['Vs Law'].split(separator)][groupName[0]]
                    except AttributeError:  # Vs law has not been specified for each group
                        lawValue = str(group['Vs Law']) if not np.isnan(group['Vs Law']) else '-1'

                    group_name = groupName[1]
                    if lawValue == "-1":  # Using Vs law and Sigma logn from main soil sheet
                        currentLaw = None
                    else:  # Using current Vs law and Sigma logn
                        currentLaw = (lawValue, group['Sigma logn'])
                elif isinstance(group['Vs Law'], (float, int)) and not np.isnan(group['Vs Law'])\
                        and group['Vs Law'] != -1:  # A numeric value for Vs mean is specified
                    currentLaw = (str(group['Vs Law']), group['Sigma logn'])
                    group_name = groupName
                elif isinstance(group['Vs Law'], str):  # A custom law for Vs mean is specified
                    currentLaw = (group['Vs Law'], group['Sigma logn'])
                    group_name = groupName
                else:  # No law specified, using main soil sheet values
                    currentLaw = None
                    group_name = groupName

                soil_name, meanVsValue, stdVsValue = self.parseLaw(group_name, currentCentroid, currentLaw)
                currentLayeredProfile.append([currentCentroid, 1, soil_name, meanVsValue, stdVsValue, VsMin, VsMax])

            layeredProfile.extend(currentLayeredProfile)  # Extending final profile list
            layeredSplitted.append(currentLayeredProfile)  # Adding current subprofile to the splitted list

        # Generation of the correlation coefficients list
        correlationCoeffList = []
        if self.correlationMode == 'Single groups':
            for index, group in self._rawGroups.iterrows():
                currentCorrelationName, currentLimit = self.parseCorrelation(group['Group name'])

                currentCoeff = self.getCorrelationVector(layeredSplitted[index], currentCorrelationName,
                                                         currentLimit)
                correlationCoeffList.extend(currentCoeff)
        else:
            firstGroup = layeredProfile[0][2]
            firstCorrelationName, firstCorrelationLimit = self.parseCorrelation(firstGroup)
            correlationCoeffList = self.getCorrelationVector(layeredProfile, firstCorrelationName,
                                                             firstCorrelationLimit)

        # Generating random Vs profile
        finalLayers = []
        lastNormValue = 0
        for index, layer, correlation in zip(range(len(layeredProfile)), layeredProfile, correlationCoeffList):

            # Computing lognormal parameters
            # layerMean = float(layer[3])
            # layerStd = float(layer[4])
            #
            # muLogn = np.log(layerMean**2/(layerStd**2 + layerMean**2)**0.5)
            # sigmaLogn = (np.log(layerStd**2/layerMean**2 + 1))**0.5
            muLogn = np.log(float(layer[3]))
            sigmaLogn = float(layer[4])

            if np.isnan(sigmaLogn):
                currentVs = float(layer[3])
            else:
                # Getting standard normal random variable for i-th layer (epsilon_i)
                standardNormValue = np.random.standard_normal(1)

                if index == 0:  # First layer of entire profile, no correlation needed
                    currentNormValue = standardNormValue
                    currentVs = np.exp(sigmaLogn*currentNormValue + muLogn)
                else:
                    currentNormValue = correlation*lastNormValue + standardNormValue *\
                                       (1 - correlation**2)**0.5
                    currentVs = np.exp(float(sigmaLogn*currentNormValue + muLogn))

                lastNormValue = currentNormValue

            # Checking Vs limit
            VsMin = layer[5]
            VsMax = layer[6]

            if not (np.isnan(VsMin) or VsMin == -1):
                currentVs = max(currentVs, VsMin)

            if not(np.isnan(VsMax) or VsMax == -1):
                currentVs = min(currentVs, VsMax)

            # Appending generated layer in list
            finalLayers.append([layer[1], layer[2], float(currentVs)])

        if bedrock_extended:
            finalLayers, correlationCoeffList = self.cutOnThreshold(finalLayers, correlationCoeffList)

        self.makeProfileColumn(finalLayers)
        self.makeCorrelationColumn(correlationCoeffList)

    def cutOnThreshold(self, finalLayers, correlationCoeffList):
        vsValues = [[index, float(current_vs[0]), float(current_vs[2])]
                    for index, current_vs in enumerate(finalLayers)]
        over_threshold = [index for index, _, value in vsValues if value >= self.vsLimit]

        if len(over_threshold) == 0:
            return finalLayers, correlationCoeffList

        current_index = over_threshold[0]
        current_remaining = finalLayers[current_index + 1:]

        remaining_vs = 0
        while len(current_remaining) > self.harmonicMeanDepth and remaining_vs < self.vsLimit:
            current_num = sum([value[0] for value in current_remaining[:self.harmonicMeanDepth]])
            current_denom = sum([thickness/value for thickness, _, value in current_remaining[:self.harmonicMeanDepth]])
            remaining_vs = current_num / current_denom

            current_index += 1
            current_remaining.pop(0)

        if len(current_remaining) <= self.harmonicMeanDepth:
            return finalLayers, correlationCoeffList
        else:
            return finalLayers[:current_index], correlationCoeffList[:current_index]

    def cutOnThreshold_old(self, finalLayers, correlationCoeffList):
        """
        SUPERSEDED BY NEW BEDROCK EXTENSION METHOD BASED ON LIMITED HARMONIC MEAN DEPTH
        """
        vsValues = [[index, float(current_vs[0]), float(current_vs[2])]
                    for index, current_vs in enumerate(finalLayers)]
        over_threshold = [index for index, _, value in vsValues if value >= self.vsLimit]

        if len(over_threshold) == 0:
            return finalLayers, correlationCoeffList

        current_index = over_threshold[0]
        current_remaining = finalLayers[current_index + 1:]

        remaining_vs = 0
        while len(current_remaining) > 0 and remaining_vs < self.vsLimit:
            current_num = sum([value[0] for value in current_remaining])
            current_denom = sum([thickness/value for thickness, _, value in current_remaining])
            remaining_vs = current_num / current_denom

            current_index += 1
            current_remaining.pop(0)

        if len(current_remaining) == 0:
            return finalLayers, correlationCoeffList
        else:
            return finalLayers[:current_index], correlationCoeffList[:current_index]

    def makeProfileColumn(self, finalLayers):
        existingDFSize = len(self._profileDF.columns)
        currentProfileName = "P{}".format(existingDFSize + 1)

        # Writing input motions
        if self.inputFiles == 'Assign by ID':
            currentProfile = ["{}.txt".format(self._rawGroups['ID CODE'][0])]
        else:
            currentProfile = ["ALL"]

        # Specifying bedrock velocity
        if self.vsPolicy == 'Assign last velocity':
            currentProfile.append(f"Bedrock Vs: {round(finalLayers[-1][-1], 1)}")
        else:
            currentProfile.append(f"Bedrock Vs: {round(self.vsLimit, 1)}")

        for element in finalLayers:
            currentProfile.append("{};{};{}".format(element[1], element[0], round(element[2], 1)))

        new_column = pd.DataFrame(currentProfile, columns=[currentProfileName])
        self._profileDF = pd.concat([self._profileDF, new_column], axis=1)

    def makeCorrelationColumn(self, correlationCoeffList):
        existingDFSize = len(self._correlationsDF.columns)
        currentProfileName = "P{}".format(existingDFSize + 1)

        new_column = pd.DataFrame(correlationCoeffList, columns=[currentProfileName])
        self._correlationsDF = pd.concat([self._correlationsDF, new_column], axis=1)
        # self._correlationsDF[currentProfileName] = pd.Series(correlationCoeffList)

    @staticmethod
    def getCorrelationVector(currentLayeredProfile, currentLaw, currentLimit=np.nan,
                             var_name_depth='H', var_name_thick='T'):

        if not isinstance(currentLaw, str) and np.isnan(currentLaw):
            return np.zeros(len(currentLayeredProfile))
        elif currentLaw.lower().startswith('toro:'):  # Using Toro correlation model
            genericModelName = currentLaw.lower().split('toro:')[1].strip().upper()

            # Creating a copy of layered profile with repeated last layer to simulate perfectly
            # correlated bedrock
            layeredCopy = currentLayeredProfile.copy()
            layeredCopy.append(layeredCopy[-1])
            profileLikeObj = ProfileLikeObj([LayerLikeObj(layer) for layer in layeredCopy])
            toroVelModelObj = pysra.variation.ToroVelocityVariation.generic_model(genericModelName)
            currentCoeff = toroVelModelObj._calc_corr(profileLikeObj)[:-1]

        else:  # Custom equation specified
            currentCoeff = []
            symbolicParse = sympy.sympify(currentLaw.replace('EXP', 'exp'))
            for layer in currentLayeredProfile[1:]:
                if np.isnan(currentLimit) or layer[0] < currentLimit:  # Computing value for current depth
                    currentValue = symbolicParse.subs(var_name_depth, layer[0]).subs(var_name_thick, layer[1])
                else:
                    currentValue = symbolicParse.subs(var_name_depth, currentLimit).subs(var_name_thick, layer[1])
                currentCoeff.append(currentValue)

        # Adding uncorrelated first layer at the beginning of the list
        currentCoeff = np.insert(currentCoeff, 0, 0)

        return currentCoeff

    def get_current_soils(self):
        soil_set = set()
        # Expanding soil alternatives
        splitted_soils = [soil.split(',') for soil in self._rawGroups['Group name']]
        extended_soils = list()
        for splitted in splitted_soils:
            extended_soils.extend(splitted)

        ordered_unique = [soil for soil in extended_soils
                          if soil not in soil_set and soil_set.add(soil) is None]

        return self._allSoils[self._allSoils['Orig name'].isin(ordered_unique)]

    def exportExcel(self, filename):
        """
        Export current profiles DataFrame into a new batch input file

        :param filename: name of the Excel file to save
        :return:
        """
        # Generating soil sheet
        new_columns = ['Soil name', 'Unit weight\n[KN/m3]', 'From\n[m]', 'To\n[m]',
                       'Vs\n[m/s]', 'Degradation curve', 'Curve Std']
        # for index, group in self._rawGroups.iterrows():
        #     currentRow = [group['Group name'], group['Unit weight\n[KN/m3]'], "", "", "",
        #                   group['Degradation curve\nMean'], group['Degradation curve\nStd']]
        #     soilDF.loc[index] = currentRow
        current_soils = self.get_current_soils()
        # for index, soil in current_soils.iterrows():
        #     currentRow = [soil['Group name'], soil['Unit weight\n[KN/m3]'], soil['From\n[m]'],
        #                   soil['To\n[m]'], str(soil['Vs Law']), soil['Degradation curve\nMean'],
        #                   soil['Degradation curve\nStd']]
        soilDF = current_soils[['Group name', 'Unit weight\n[KN/m3]', 'From\n[m]', 'To\n[m]',
                                'Degradation curve\nMean', 'Degradation curve\nStd']]
        soilDF.insert(loc=4, column='Vs\n[m]', value="")
        rename_mapper = {old: new for old, new in zip(soilDF.columns, new_columns)}

        with pd.ExcelWriter(filename, mode='w') as writer:
            soilDF.rename(columns=rename_mapper).to_excel(writer, sheet_name='Soils', index=False)

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


class ProfileLikeObj:
    """
    Simple class with depth mid attribute to be compatible with ToroVelocity class in Pysra
    """
    def __init__(self, layer_like_obj):
        self.depth_mid = np.array([layer.depth_mid for layer in layer_like_obj])


class GenericSoilVariator(pysra.variation.SoilTypeVariation):
    """
    Generic soil variator with custom equation
    """
    def __init__(self, correlation, GFunc, DFunc):
        super().__init__(correlation=correlation)
        self._GFunc = sympy.sympify(GFunc.replace('^', '**'))
        self._DFunc = sympy.sympify(DFunc.replace('^', '**'))

    def _get_varied(self, randvar, mod_reduc, damping):
        mod_reduc_means = mod_reduc
        mod_reduc_stds = self.calc_std_mod_reduc(mod_reduc_means)
        varied_mod_reduc = mod_reduc_means + randvar[0] * mod_reduc_stds

        damping_means = damping
        damping_stds = self.calc_std_damping(damping_means)
        varied_damping = damping_means + randvar[1] * damping_stds

        return varied_mod_reduc, varied_damping

    def calc_std_mod_reduc(self, mod_reduc, var_name='G'):
        mod_reduc = np.asarray(mod_reduc).astype(float)
        std = [self._GFunc.subs(var_name, mod_value).evalf()
               for mod_value in mod_reduc]
        return np.array(std)

    def calc_std_damping(self, damping, var_name='D'):
        damping = np.asarray(damping).astype(float)
        std = [self._DFunc.subs(var_name, damp_value).evalf()
               for damp_value in damping]
        return np.array(std)


class ClusterPermutator:
    """
    Class for handling the generation of batch input files for cluster permutations
    """
    def __init__(self, filename, output_folder):
        soil_definition = pd.read_excel(filename, sheet_name='Soils')
        clusters_definition = pd.read_excel(filename, sheet_name='Clusters')
        bedrock_definition = pd.read_excel(filename, sheet_name='Bedrocks')

        self.soils = self.makeSoils(soil_definition)
        self.bedrocks = self.makeSoils(bedrock_definition, variable_prop=False)
        self.number_clusters = len(clusters_definition)

        name_list = ["{}-{}".format(row['Cluster'], row['Sub-cluster'])
                     for _, row in clusters_definition.iterrows()]
        self.name_list = name_list

        self._original_soil_names = set(soil_definition['Soil name'])
        self._rawSoils = soil_definition
        self._rawClusters = clusters_definition
        self._rawBedrocks = bedrock_definition
        self._output_folder = output_folder

    @staticmethod
    def makeSoils(soil_definition, variable_prop=True):
        if variable_prop:
            for index, row in soil_definition.iterrows():
                if np.isnan(row['From\n[m]']):
                    soil_definition.loc[index, 'From\n[m]'] = 0
                if np.isnan(row['To \n[m]']):
                    soil_definition.loc[index, 'To \n[m]'] = INF_DEPTH

        soil_names = set(soil_definition['Soil name'])

        soil_list = pd.DataFrame(columns=soil_definition.columns)
        for soil in soil_names:
            current_features = soil_definition[soil_definition['Soil name'] == soil]
            current_features.reset_index(inplace=True)
            for index, row in current_features.iterrows():
                if variable_prop:
                    row['Orig name'] = soil
                    row['Soil name'] = "{}[{}]".format(soil, index)

                soil_list = soil_list.append(row, ignore_index=True)

        # Parsing Vs laws
        for index, row in soil_list.iterrows():
            soil_list.loc[index, 'Vs law'] = sympy.sympify(row['Vs law'])

        return soil_list

    def makeClusterProfiles(self, row_index):
        current_cluster = self._rawClusters.iloc[row_index]
        current_input = current_cluster['Input files']

        brickedProfile = self.makeBricks(current_cluster)

        bricked_permutations = NCLib.calcPermutations(brickedProfile, True)

        if current_cluster['Add geological bedrock'] == 'Y':
            final_permutations = list()
            bricked_permutations_tuple = [self.addBedrock(profile) for profile in bricked_permutations]
            for profile_package in bricked_permutations_tuple:
                for profile in profile_package:
                    final_permutations.append(profile)
        else:  # Extending last layer type until bedrock speed is met
            final_permutations = self.extendUntilVelocity(bricked_permutations)

        complete_profiles = [self.addVs(NCLib.addDepths(profile)) for profile in final_permutations]
        cluster_info_dict = {'Subcluster': current_cluster['Sub-cluster'],
                             'Cluster name': current_cluster['Cluster'],
                             'Input list': current_input}

        self.writeProfile(complete_profiles, cluster_info_dict)

    def makeFinalSoilSheet(self):
        # Creating soil sheet
        soil_sheet = self.soils.drop(columns=['index', 'Orig name']).rename(columns={'Vs law': 'Vs\n[m/s]'})
        soil_sheet['Curve Std'] = ""
        soil_sheet['Vs\n[m/s]'] = ""

        # Adding bedrock definitions
        bedrock_sheet = self.bedrocks.drop(columns=['index']).rename(columns={'Vs law': 'Vs\n[m/s]'})
        bedrock_sheet.insert(2, 'From\n[m]', 0)
        bedrock_sheet.insert(3, 'To \n[m]', INF_DEPTH)
        bedrock_sheet['Vs\n[m/s]'] = ""
        bedrock_sheet['Curve Std'] = ""

        return pd.concat([soil_sheet, bedrock_sheet])

    def writeProfile(self, complete_profiles, cluster_info_dict):
        onlyname = "{}-{}.xlsx".format(cluster_info_dict['Cluster name'],
                                       cluster_info_dict['Subcluster'])
        filename = os.path.join(self._output_folder, onlyname)

        self.makeFinalSoilSheet().to_excel(filename, index=False, sheet_name='Soils')

        # Creating profile sheet
        profileDF = pd.DataFrame()

        for index, profile in enumerate(complete_profiles):
            current_name = "P{}".format(index + 1)
            current_input = cluster_info_dict['Input list']

            profile_list = list()

            for layer in profile:
                profile_list.append("{};{};{}".format(*layer))
            profile_list.insert(0, current_input)

            new_column = pd.DataFrame(profile_list, columns=[current_name])
            profileDF = pd.concat([profileDF, new_column], axis=1)
            # profileDF[current_name] = pd.Series(profile_list)

        # Generating profiles, correlations and empty clusters sheets
        with pd.ExcelWriter(filename, mode='a') as writer:
            profileDF.to_excel(writer, sheet_name='Profiles', index=False)
            emptyClusterSheet = pd.DataFrame(columns=['Cluster', 'Sub-cluster', 'Bedrock depth\n[m]',
                                                      'Brick thickness\n[m]', 'Input files'])
            emptyClusterSheet.to_excel(writer, sheet_name='Clusters', index=False)

    @staticmethod
    def makeBricks(cluster_info):
        bedrock_depth = cluster_info['Bedrock depth\n[m]']
        brick_size = cluster_info['Brick thickness\n[m]']
        soil_info = cluster_info.iloc[6:]

        new_profile = list()
        for soil, percentage in soil_info.iteritems():
            currentThickness = round(bedrock_depth * percentage / 100, 2)
            numberBricks = int(np.floor(currentThickness / brick_size))

            for strato in range(numberBricks):
                if strato != numberBricks - 1:
                    stratoThickness = brick_size
                else:
                    stratoThickness = currentThickness - brick_size * (numberBricks - 1)
                new_profile.append([stratoThickness, soil])

        return new_profile

    def addVs(self, profile, max_vs=800):

        new_profile = list()
        for depth, thickness, name in profile:
            current_centroid = depth + thickness / 2
            if name in self._original_soil_names:
                # Associating name based on the depth
                current_soil = self.soils[(self.soils['Orig name'] == name) &
                                          (self.soils['From\n[m]'] <= current_centroid) &
                                          (current_centroid <= self.soils['To \n[m]'])]
            else:
                current_soil = self.bedrocks[self.bedrocks['Soil name'] == name]

            current_vs = min(max_vs, current_soil['Vs law'].values[0].subs('H', current_centroid).evalf())
            new_profile.append([current_soil['Soil name'].values[0], thickness, current_vs])

        return new_profile

    def addBedrock(self, profile, brick_size=3, max_depth=100, max_vs=800):
        profile_depth = sum([thickness for thickness, _ in profile])

        profiles_list = list()
        for index, bedrock in self.bedrocks.iterrows():
            current_depth = profile_depth
            current_vs = 0
            bedrock_layers = list()
            while current_depth < max_depth and current_vs < max_vs:
                current_centroid = current_depth + brick_size / 2
                current_vs = bedrock['Vs law'].subs('H', current_centroid).evalf()
                current_depth += brick_size
                bedrock_layers.append((brick_size, bedrock['Soil name']))
            profiles_list.append(profile + tuple(bedrock_layers))

        return profiles_list

    def extendUntilVelocity(self, profile_list, max_vs=800, max_depth=100):

        profile_list = list(profile_list)
        new_profile_list = list()
        for profile in profile_list:
            current_profile = list(profile)
            profile_w_depth = NCLib.addDepths(current_profile)
            profile_vs = self.addVs(profile_w_depth)
            base_depth = profile_w_depth[-1][0] + profile_w_depth[-1][1]

            while profile_vs[-1][2] < max_vs and base_depth < max_depth:
                current_profile.append(profile[-1])
                profile_w_depth = NCLib.addDepths(current_profile)
                profile_vs = self.addVs(profile_w_depth)
                base_depth = profile_w_depth[-1][0] + profile_w_depth[-1][1]
            new_profile_list.append(current_profile)

        return new_profile_list


class ClusterToMOPS(ClusterPermutator):

    def __init__(self, filename, output_folder, iterations=100):
        super().__init__(filename, output_folder)
        self.iterations = iterations

    def addBedrock(self, profile, brick_size=3, max_depth=100, max_vs=800):
        profiles_list = list()
        unique_soil_names = set(self.bedrocks['Soil name'])
        for bedrock in unique_soil_names:
            current_profile = list(profile)
            current_profile.append((brick_size, bedrock))
            profiles_list.append(current_profile)
        return profiles_list

    def makeClusterProfiles(self, row_index):
        current_cluster = self._rawClusters.iloc[row_index]
        current_input = current_cluster['Input files']

        brickedProfile = self.makeBricks(current_cluster)
        bricked_permutations = NCLib.calcPermutations(brickedProfile, True)

        if current_cluster['Add geological bedrock'] == 'Y':
            bricked_permutations_tuple = [self.addBedrock(profile) for profile in bricked_permutations]
            final_permutations = list()
            for profile_package in bricked_permutations_tuple:
                for profile in profile_package:
                    final_permutations.append(profile)
        else:
            final_permutations = bricked_permutations

        cluster_info_dict = {'Subcluster': current_cluster['Sub-cluster'],
                             'Cluster name': current_cluster['Cluster'],
                             'Input list': current_input}

        self.writeProfile(final_permutations, cluster_info_dict)

    def writeProfile(self, complete_profile, cluster_info_dict):
        onlyname = "{}-{}.xlsx".format(cluster_info_dict['Cluster name'],
                                       cluster_info_dict['Subcluster'])
        filename = os.path.join(self._output_folder, onlyname)

        merged_soilSheet = self.makeFinalSoilSheet()
        merged_soilSheet.to_excel(filename, index=False, sheet_name='Group definitions')

        # Creating the Stochastic sheet

        stochastic_sheet = pd.DataFrame(columns=['ID CODE', 'Lat', 'Lon', 'Group name', 'Min thickness\n[m]',
                                                 'Max thickness\n[m]', 'Vs Law', 'Sigma logn', 'Vs min\n[m/s]',
                                                 'Vs max\n[m/s]', 'Bedrock extension', ' ', '  ',
                                                 'Number of iterations', 'Input files', 'Random seed',
                                                 'Correlation mode', 'Bedrock Vs\n[m/s]', 'Max depth\n[m]'])

        stochastic_sheet['Bedrock extension'] = 'N'  # Bedrock not extended by default

        for perm_index, permutation in enumerate(complete_profile):
            current_ID = "{}-{}-Perm{}".format(cluster_info_dict['Cluster name'], cluster_info_dict['Subcluster'],
                                               perm_index + 1)

            for soil_index, soil_data in enumerate(permutation):
                currentRow = pd.Series(index=stochastic_sheet.columns)

                if perm_index == 0 and soil_index == 0:
                    currentRow['Number of iterations'] = self.iterations
                    currentRow['Input files'] = self._rawClusters['Input files'][0]
                    currentRow['Correlation mode'] = 'All profile'
                    currentRow['Bedrock Vs\n[m/s]'] = 800
                    currentRow['Max depth\n[m]'] = 100
                currentRow['ID CODE'] = current_ID
                currentRow['Group name'] = soil_data[1]
                currentRow['Min thickness\n[m]'] = soil_data[0]
                currentRow['Max thickness\n[m]'] = soil_data[0]

                stochastic_sheet = stochastic_sheet.append(currentRow, ignore_index=True)

        with pd.ExcelWriter(filename, mode='a') as writer:
            stochastic_sheet.to_excel(writer, index=False, sheet_name='Stochastic')

    def makeFinalSoilSheet(self):
        # Creating soil sheet
        soil_sheet = self._rawSoils.rename(columns={'Vs law': 'Vs Law', 'Soil name': 'Group name',
                                                    'To \n[m]': 'To\n[m]'})

        # Adding bedrock definitions
        bedrock_sheet = self._rawBedrocks.rename(columns={'Vs law': 'Vs Law', 'Soil name': 'Group name',
                                                          'To \n[m]': 'To\n[m]'})

        return pd.concat([soil_sheet, bedrock_sheet])


class NTCCalculator:
    """
    Class for handling NTC spectra generation
    """
    def __init__(self, csvname):
        self.NTCDatabase = pd.read_csv(csvname, sep='\t')

    def agNTC(self, lon, lat, tr=475):
        """
        Calcola il valore di accelerazione attesa al sito note le coordinate e il tempo di ritorno

        lon: Longitudine del sito
        lat: Latitudine del sito
        tr:  Tempo di ritorno dell'azione sismica
        """

        NTCDatabase = self.NTCDatabase
        distanceVect = [((lon - float(row['Lon'])) ** 2 + (lat - float(row['Lat'])) ** 2) ** 0.5
                        for _, row in NTCDatabase.iterrows()]
        NTCDatabase['Distance'] = distanceVect
        currentPoints = NTCDatabase.sort_values('Distance').iloc[:4, :]

        # Replacing zero distance if any points lies on the grid
        currentPoints['Distance'] = currentPoints['Distance'].apply(lambda x: x if x != 0 else 1e-9)

        close_ag = currentPoints["{}_ag".format(tr)].values
        close_F0 = currentPoints["{}_F0".format(tr)].values
        close_Tc = currentPoints["{}_Tc".format(tr)].values

        inv_distance = currentPoints['Distance'].values ** -1

        final_ag = sum(np.multiply(close_ag, inv_distance)) / sum(inv_distance)
        final_F0 = sum(np.multiply(close_F0, inv_distance)) / sum(inv_distance)
        final_Tc = sum(np.multiply(close_Tc, inv_distance)) / sum(inv_distance)

        return final_ag / 10, final_F0, final_Tc  # Restituisce l'accelerazione in g, il fattore F0 e il periodo Tc*

    @staticmethod
    def soilCoefCalc(ag, F0, Tcstar, Cat):
        """
        Calcola il coefficiente di sottosuolo a partire dalla categoria e dal valore del fattore di amplificazione

        INPUT
        Cat: Categoria di sottosuolo (da 'A' a 'E')
        ag: Valore dell'accelerazione attesa al sito (in g)
        F0: Fattore di amplificazione relativo al sito e al Tr
        Tcstar: Periodo Tc* di normativa

        OUTPUT
        Ss: Coefficiente di sottosuolo 1
        Cc: Coefficiente di sottosuolo 2
        """

        if Cat == 'A':
            Ss = 1
            Cc = 1
        elif Cat == 'B':
            SsCalc = 1.4 - 0.4 * F0 * ag
            Ss = min(SsCalc, 1.2)
            Ss = Ss if Ss > 1 else 1

            Cc = 1.1 * Tcstar ** (-0.2)
        elif Cat == 'C':
            SsCalc = 1.7 - 0.6 * F0 * ag
            Ss = min(SsCalc, 1.5)
            Ss = Ss if Ss > 1 else 1

            Cc = 1.05 * Tcstar ** (-0.33)
        elif Cat == 'D':
            SsCalc = 2.4 - 1.5 * F0 * ag
            Ss = min(SsCalc, 1.8)
            Ss = Ss if Ss > 0.9 else 0.9

            Cc = 1.25 * Tcstar ** (-0.5)
        elif Cat == 'E':
            SsCalc = 2 - 1.1 * F0 * ag
            Ss = min(SsCalc, 1.6)
            Ss = Ss if Ss > 1 else 1

            Cc = 1.15 * Tcstar ** (-0.4)
        else:
            Ss = None
            Cc = None
        return Ss, Cc

    @staticmethod
    def computeNTCSpectrum(ag, F0, Tcstar, Ss=1, Cc=1, Topog='T1', qq=1, passo=0.01):
        """
        Calcola lo spettro di risposta di normativa secondo NTC2008

        INPUT
        ag:     Accelerazione di picco attesa al sito per categoria A
        F0:     Fattore di amplificazione
        Tcstar: Valore del periodo Tc*
        Ss:     Coefficiente di sottosuolo
        Cc:     Coefficiente di sottosuolo 2
        Topog:  Categoria topografica (da T1 a T4)
        qq:     Fattore di struttura
        passo:  Passo dei periodi per il calcolo dello spettro di risposta
        asseGrafico:    Se specificato, plotta lo spettro nell'asse desiderato

        OUTPUT
        TT:     Vettore dei periodi in cui è stato calcolato lo spettro
        RS:     Valori di pseudoaccelerazione calcolati
        """

        # Calcolo del coefficiente di categoria topografica
        if Topog.upper() == 'T4':
            St = 1.4
        elif Topog.upper() == 'T2' or Topog.upper() == 'T3':
            St = 1.2
        else:  # Categoria topografica T1
            St = 1

        S = Ss * St

        # Calcolo dello spettro di risposta
        Tc = Cc * Tcstar
        Tb = Tc / 3
        Td = 4 * ag + 1.6
        eta = 1 / qq

        # TT = np.arange(0, 4 + passo, passo)
        TTlog = np.logspace(-1, np.log10(4), 91, endpoint=True)
        TTlin = np.arange(0, 0.1, passo)

        TT = np.hstack([TTlin, TTlog])
        RS = list()

        for periodo in TT:

            if periodo < Tb:
                RS.append(ag * S * eta * F0 * (periodo / Tb + 1 / (eta * F0) * (1 - periodo / Tb)))
            elif periodo < Tc:
                RS.append(ag * S * eta * F0)
            elif periodo < Td:
                RS.append(ag * S * eta * F0 * (Tc / periodo))
            else:
                RS.append(ag * S * eta * F0 * (Tc * Td / periodo ** 2))

        return np.vstack((TT, RS)).transpose()


class UHSCalculator:

    def __init__(self, database_name, percentile='50'):
        current_sheet = f"SA_475_{percentile}percentile"
        self.UHSdatabase = pd.read_excel(database_name, sheet_name=current_sheet)
        self.NTCCalculator = NTCCalculator('Resources/NTC2008.csv')

    def get_values(self, lon, lat):
        UHSDatabase = self.UHSdatabase
        distanceVect = [((lon - float(row['Lon'])) ** 2 + (lat - float(row['Lat'])) ** 2) ** 0.5
                        for _, row in UHSDatabase.iterrows()]
        UHSDatabase['Distance'] = distanceVect
        currentPoints = UHSDatabase.sort_values('Distance').iloc[:4, :]

        # Replacing zero distance if any points lies on the grid
        currentPoints['Distance'] = currentPoints['Distance'].apply(lambda x: x if x != 0 else 1e-9)

        inv_distance = currentPoints['Distance'].values ** -1

        numeric_data = currentPoints.iloc[:, 3:-1]
        numeric_data = numeric_data.apply(lambda column: sum(np.multiply(column, inv_distance)) / sum(inv_distance))

        TT = [float(name.split('_')[-1]) for name in numeric_data.index]

        # Adding pga value to 0.01 period to prevent further numeric errors

        SA_001, _, _ = self.NTCCalculator.agNTC(lon=lon, lat=lat)

        TT.insert(0, 0.01)
        numeric_data = np.append(SA_001, numeric_data)

        return np.column_stack((TT, numeric_data))
