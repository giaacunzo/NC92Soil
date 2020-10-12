import pysra
import numpy as np
import pandas as pd
import sympy
import SRALibrary as NCLib
import os


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

    @staticmethod
    def calcVs30(calc):
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
        currentGroupsDef = pd.read_excel(filename, sheet_name='Group definitions')

        self.numberIterations = int(currentStochastic['Number of iterations'][0])
        self.correlationMode = currentStochastic['Correlation mode'][0]
        self.vsLimit = currentStochastic['Bedrock Vs\n[m/s]'][0]

        # Check of random seed
        if np.isnan(currentStochastic['Random seed'][0]) or not isinstance(currentStochastic['Random seed'], float):
            self.randomSeed = None
        else:
            self.randomSeed = int(currentStochastic['Random seed'][0])

        # Merging with "Group definition" sheet
        currentStochastic = self.merge_sheets(currentStochastic, currentGroupsDef)

        # Parsing the vs law expression
        for index, vsLaw in enumerate(currentStochastic['Vs Law']):
            vsLawObject = sympy.sympify(vsLaw.replace('^', '**'))
            currentStochastic.loc[index, 'Vs Law'] = vsLawObject

        self.idList = sorted(set(currentStochastic['ID CODE']))
        self._allData = currentStochastic
        self._rawGroups = pd.DataFrame()
        self._profileDF = pd.DataFrame()
        self._correlationsDF = pd.DataFrame()

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

    def parseLaw(self, lawIndex, depth):
        """

        :param lawIndex: index of the considered group
        :param depth: the value of depth at which vs mean value must be computed

        :return: the value of the mean Vs for normal distribution
        """

        currentVsLaw = self._rawGroups['Vs Law'][lawIndex]
        currentLimit = self._rawGroups['Maximum depth\n[m]'][lawIndex]

        return currentVsLaw.subs('H', currentLimit).evalf()

    def generateRndProfile(self):
        np.random.seed(self.randomSeed)

        # Creating basic layered profile as [centroid, thickness, name]
        currentDepth = 0
        layeredProfile = []
        layeredSplitted = []
        totalDepth = 0
        for index, group in self._rawGroups.iterrows():
            currentLayeredProfile = []
            groupName = group['Group name']
            if index == len(self._rawGroups) - 1:
                groupThickness = 100 - totalDepth
            else:
                groupThickness = np.random.randint(group['Min thickness\n[m]'], group['Max thickness\n[m]'] + 1)
                totalDepth += groupThickness

            for counter in range(groupThickness):
                currentCentroid = currentDepth + 1/2
                currentDepth += 1

                # Evaluating mean Vs value from the given relation
                meanVsValue = self.parseLaw(index, currentCentroid)
                stdVsValue = group['Sigma logn']
                currentLayeredProfile.append([currentCentroid, 1, groupName, meanVsValue, stdVsValue])

            layeredProfile.extend(currentLayeredProfile)  # Extending final profile list
            layeredSplitted.append(currentLayeredProfile)  # Adding current subprofile to the splitted list

        # Generation of the correlation coefficients list
        correlationCoeffList = []
        if self.correlationMode == 'Single groups':
            for index, group in self._rawGroups.iterrows():
                currentCorrelationName = group['Inter-layer correlation']
                currentLimit = group['Maximum depth\n[m]']

                currentCoeff = self.getCorrelationVector(layeredSplitted[index], currentCorrelationName, currentLimit)
                correlationCoeffList.extend(currentCoeff)
        else:
            firstCorrelationName = self._rawGroups['Inter-layer correlation'][0]
            firstCorrelationLimit = self._rawGroups['Maximum depth\n[m]'][0]
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

        finalLayers = self.cutOnThreshold(finalLayers)

        self.makeProfileColumn(finalLayers)
        self.makeCorrelationColumn(correlationCoeffList)

    def cutOnThreshold(self, finalLayers):
        vsValues = [[index, float(current_vs[0]), float(current_vs[2])]
                    for index, current_vs in enumerate(finalLayers)]
        over_threshold = [index for index, _, value in vsValues if value >= self.vsLimit]

        if len(over_threshold) == 0:
            return finalLayers

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
            return finalLayers
        else:
            return finalLayers[:current_index]

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

    @staticmethod
    def getCorrelationVector(currentLayeredProfile, currentLaw, currentLimit=np.nan,
                             var_name_depth='H', var_name_thick='T'):

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
                if np.isnan(currentLimit):  # No limit depth for correlation specified
                    currentValue = symbolicParse.subs(var_name_depth, layer[0]).subs(var_name_thick, layer[1])
                else:
                    currentValue = symbolicParse.subs(var_name_depth, currentLimit).subs(var_name_thick, layer[1])
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
                                       'Vs\n[m/s]', 'Degradation curve', 'Curve Std'])
        for index, group in self._rawGroups.iterrows():
            currentRow = [group['Group name'], group['Unit weight\n[KN/m3]'], "", "", "",
                          group['Degradation curve\nMean'], group['Degradation curve\nStd']]
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
                    soil_definition.loc[index, 'To \n[m]'] = 10000

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
        else:
            final_permutations = bricked_permutations

        complete_profiles = [self.addVs(NCLib.addDepths(profile)) for profile in final_permutations]
        cluster_info_dict = {'Subcluster': current_cluster['Sub-cluster'],
                             'Cluster name': current_cluster['Cluster'],
                             'Input list': current_input}

        self.writeProfile(complete_profiles, cluster_info_dict)

    def writeProfile(self, complete_profiles, cluster_info_dict):
        onlyname = "{}-{}.xlsx".format(cluster_info_dict['Cluster name'],
                                       cluster_info_dict['Subcluster'])
        filename = os.path.join(self._output_folder, onlyname)

        # Creating soil sheet
        soil_sheet = self.soils.drop(columns=['index', 'Orig name']).rename(columns={'Vs law': 'Vs\n[m/s]'})
        soil_sheet['Curve Std'] = ""
        soil_sheet['Vs\n[m/s]'] = ""
        soil_sheet.to_excel(filename, index=False, sheet_name='Soils')

        # Creating profile sheet
        profileDF = pd.DataFrame()

        for index, profile in enumerate(complete_profiles):
            current_name = "P{}".format(index + 1)
            current_input = cluster_info_dict['Input list']

            profile_list = list()

            for layer in profile:
                profile_list.append("{};{};{}".format(*layer))
            profile_list.insert(0, current_input)

            profileDF[current_name] = pd.Series(profile_list)

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

    def addVs(self, profile):

        new_profile = list()
        for depth, thickness, name in profile:
            current_centroid = depth + thickness / 2
            if name in self._original_soil_names:
                # Associating name based on the depth
                current_soil = self.soils[(self.soils['Orig name'] == name) &
                                          (self.soils['From\n[m]'] <= current_centroid) &
                                          (current_centroid <= self.soils['To \n[m]'])]
                current_vs = current_soil['Vs law'].values[0].subs('H', current_centroid).evalf()
                new_profile.append([name, thickness, current_vs])
            else:
                current_soil = self.bedrocks[self.bedrocks['Soil name'] == name]
                current_vs = current_soil['Vs law'].values[0].subs('H', current_centroid).evalf()
                new_profile.append([name, thickness, current_vs])

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

# class SoilPropertyVariatorOLD:
#     """
#     Class for handling randomly variated soil property curves
#     """
#     def __init__(self, meanCurves, curveStds, GLimits=(0.05, 1), DLimits=(0.001, 0.25),
#                  truncation=2, correlation=-0.5):
#         self._GLimits = GLimits
#         self._DLimits = DLimits
#         self._meanCurves = meanCurves
#         self._curveStds = curveStds
#         self.randGenerator = pysra.variation.TruncatedNorm(truncation)
#         self._correlation = correlation
#
#     def getGenericVariated(self):
#         """
#         Generates a variated curve given mean value and std deviation
#         :return: nx3 array with the generated curve
#         """
#         # Generates a pair of correlated random variables
#         randomVars = self.randGenerator.correlated(self._correlation)
#
#         # Computing variated G curve
#         GVariated = list()
#         for value in self._meanCurves[:, 1]:
#             currentValue = value + randomVars[0] * self._curveStds[0]
#             if currentValue < self._GLimits[0]:
#                 currentValue = self._GLimits[0]
#             elif currentValue > self._GLimits[1]:
#                 currentValue = self._GLimits[1]
#             GVariated.append(currentValue)
#
#         GVariated = np.array(GVariated)
#
#         # Computing variated D curve
#         DVariated = list()
#         for value in self._meanCurves[:, 2]:
#             currentValue = value + randomVars[1] * self._curveStds[1]
#             if currentValue < self._DLimits[0]:
#                 currentValue = self._DLimits[0]
#             elif currentValue > self._DLimits[1]:
#                 currentValue = self._DLimits[1]
#             DVariated.append(currentValue)
#
#         DVariated = np.array(DVariated)
#
#         return np.stack([self._meanCurves[:, 0], GVariated, DVariated], axis=1)
