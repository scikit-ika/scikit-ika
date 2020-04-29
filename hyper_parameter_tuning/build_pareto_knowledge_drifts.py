# -*- coding: utf-8 -*-
#import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
import warnings
from kneed import KneeLocator


class buildDriftKnowledge():
    """
    Description
    Class to build the pareto knowledge from hyper-parameters configurations evaluated on differents datasets for the drift detector tuning.
    The knowledge consists in the best configuration of hyper-parameters for each dataset. 
    
    The datasets are characterised by meta-features and a knowledge base can be then be built to link these features to the best configurations. 
    
    Parameters :
        results_directory : str 
            Path to the directory containing the knowledge files (results of the evaluation of the configurations on example streams)
            
        namesDetectors : list of str
            List of the names of the detectors
            
        namesStreams : list of str
            list of the names of the streams
            
        nMetaFeatures : int, default = 15 ((severity, magnitude, interval) * (med, kurto, skew, per10, per90))
            Number of meta-features extracted from the stream 
            NOT USED FOR THE MOMENT as we use theoritical meta-features and not measured ones
        
        knowledge_type : str
            String indicating what knowledge is being calculated (for arf tree tuning or drift detectors)
            NOT USED FOR THE MOMENT, need further implementing to bring the two applications together
            
        output : str
            Directory path where to save output file
            
        verbose : bool, default = False
            Print pareto figures if True 
        
        Output :
            Csv file containing the configurations selected for each example stream (each row = 1 stream)
            
            
        Example
        --------  
            >>> namesStm = ['BernouW1ME0010','BernouW1ME005095','BernouW1ME00509','BernouW1ME0109','BernouW1ME0108','BernouW1ME0208','BernouW1ME0207','BernouW1ME0307','BernouW1ME0306','BernouW1ME0406','BernouW1ME0506','BernouW1ME05506',
                            'BernouW100ME0010','BernouW100ME005095','BernouW100ME00509','BernouW100ME0109','BernouW100ME0108','BernouW100ME0208','BernouW100ME0207','BernouW100ME0307','BernouW100ME0306','BernouW100ME0406','BernouW100ME0506','BernouW100ME05506',
                            'BernouW500ME0010','BernouW500ME005095','BernouW500ME00509','BernouW500ME0109','BernouW500ME0108','BernouW500ME0208','BernouW500ME0207','BernouW500ME0307','BernouW500ME0306','BernouW500ME0406','BernouW500ME0506','BernouW500ME05506']
        
            >>> namesDetect = [['PH1','PH2','PH3','PH4','PH5','PH6','PH7','PH8','PH9','PH10','PH11','PH12','PH13','PH14','PH15','PH16'],
                                  ['ADWIN1','ADWIN2','ADWIN3','ADWIN4','ADWIN5','ADWIN6','ADWIN7','ADWIN8','ADWIN9'],
                                  ['DDM1','DDM2','DDM3','DDM4','DDM5','DDM6','DDM7','DDM8','DDM9','DDM10'],
                                  ['SeqDrift21','SeqDrift22','SeqDrift23','SeqDrift24','SeqDrift25','SeqDrift26','SeqDrift27','SeqDrift28','SeqDrift29','SeqDrift210',
                                   'SeqDrift211','SeqDrift212','SeqDrift213','SeqDrift214','SeqDrift215','SeqDrift216','SeqDrift217','SeqDrift218']]
            
            >>> output_dir = os.getcwd()
            >>> directoryPathFiles = 'examples/pareto_knowledge/ExampleDriftKnowledge'
            
            >>> paretoBuild = buildDriftKnowledge(results_directory = directoryPathFiles, namesDetectors = namesDetect, namesStreams = namesStm, output = output_dir, verbose =True)
            >>> paretoBuild.load_drift_data()
            >>> paretoBuild.calculatePareto()
            >>> paretoBuild.bestConfig
        
        
        
    """
    
    def __init__(self,
                 results_directory,
                 namesDetectors,
                 namesStreams,
                 output,
#                 nMetaFeatures = 15,
#                 knowledge_type = 'Drift',
                 verbose = False):
        
        if results_directory != None and namesDetectors != None and namesStreams != None and output != None:
            self.results_directory = results_directory
            self.namesDetectors = namesDetectors
            self.namesStreams = namesStreams
            self.output = output
        else :
            raise ValueError('Directory paths or list of detectors names or list of streams missing.')
        
        self.verbose = verbose
        
#        self.knowledge_type = knowledge_type
        
        self.n_detectors = 4
        
#        self.nMetaFeatures = nMetaFeatures
        
        self.nStreams = len(self.namesStreams)
        
        self.bestConfig = [[] for indS in range(self.nStreams)]
        
        warnings.filterwarnings("ignore")
    
    @property
    def bestConfig(self):
        """ Retrieve the length of the stream.
        Returns
        -------
        int
            The length of the stream.
        """
        return self._bestConfig
    
    @bestConfig.setter
    def bestConfig(self, bestConfig):
        """ Set the length of the stream
        Parameters
        ----------
        length of the stream : int
        """
        self._bestConfig = bestConfig   
        
    def load_drift_data(self) :
        """
        Function to load the performance data from the csv files
    
    
        """
        
        # Variables for performances of the detectors
        self.scoresPerf = []
        self.listNameDetecOK = []
        for indS in range(self.nStreams) :
            self.scoresPerf.append([[] for indD in range(self.n_detectors)])
            self.listNameDetecOK.append([[] for indD in range(self.n_detectors)])
        
        # Variables for the meat-features
        self.meanMetaFeatures = []
        self.stdMetaFeatures = []
        for indS in range(self.nStreams) :
            self.meanMetaFeatures.append([[] for indD in range(self.n_detectors)])
            self.stdMetaFeatures.append([[] for indD in range(self.n_detectors)])
        
        indS = 0
        # Loop through the streams
        for streamName in self.namesStreams :
            
            # Open the performance file for the given stream
            streamPerfFile = self.results_directory+'\\'+streamName+'PerfDetectors.csv'
            with open(streamPerfFile) as csvDataFile:
                dataP = [row for row in csv.reader(csvDataFile)]
                
            indD = 0
            # Loop through the detectors
            for nameDetector in self.namesDetectors :
                # Loop through the detectors configurations
                for nameIndDetector in nameDetector :
                    # Get the row of performances for the given configuration
                    rowindDetector = next((x for x in dataP if x[0] == nameIndDetector), None)
                    # Only if no Nan value in the row (which mean that the detector detected no drifts at some point)
                    if 'nan' not in rowindDetector :
                        # Store the TP number and the FP number for the given detector
                        self.scoresPerf[indS][indD].append([float(rowindDetector[2]),float(rowindDetector[3])])
                        self.listNameDetecOK[indS][indD].append(nameIndDetector)
                indD += 1       
             
                
#            # Open the meta-features file for the given stream
#            streamMetaFeatFile = self.results_directory+'\\'+streamName+'metaFeatDetectors.csv'
#            with open(streamMetaFeatFile) as csvDataFile:
#                dataMF = [row for row in csv.reader(csvDataFile)]
#                
#            indD = 0
#            # Loop through the detectors
#            for nameDetector in self.namesDetectors :
#                listMetaFeatValues = [[] for i in range(self.nMetaFeatures)] # list to store values of each of the meta-features for each detector type
#                # Loop through the detectors configurations
#                for nameIndDetector in nameDetector :
#                    indDetec = [i for i in range(len(dataMF)) if dataMF[i][0] == nameIndDetector][0]
#                    # Loop for each meta-feature
#                    for indMetaFeat in range(self.nMetaFeatures) :
#                        listMetaFeatValues[indMetaFeat].append(float(dataMF[indDetec+indMetaFeat+1][1]))
#                    
#                self.meanMetaFeatures[indS][indD] = np.nanmean(listMetaFeatValues, axis = 1)
#                self.stdMetaFeatures[indS][indD] = np.nanstd(listMetaFeatValues, axis = 1)    
#                
#                indD += 1  
                
            indS += 1 
            
        print('end')
    
    def processMetaFeatures(self):
        print('Start process meta-features')
        
        # TODO : to come
        
    def calculatePareto(self) :
        """
        Function to calculate the Pareto front and detect the knee point
        
        """
        print('Start Pareto calculation')
        for indS in range(self.nStreams) :
            for indD in range(self.n_detectors) :
                names = self.listNameDetecOK[indS][indD]
                
                score = np.array(self.scoresPerf[indS][indD])
            
                # Calculate pareto front
                pareto = self.identify_pareto(score)
#                print ('Pareto front index vales')
#                print ('Points on Pareto front: \n',pareto)
                
                pareto_front = score[pareto]
#                print ('\nPareto front scores')
#                print (pareto_front)
                
                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df.sort_values(0, inplace=True)
                pareto_front = pareto_front_df.values
                
                scorepd = pd.DataFrame(score,columns = ['X' , 'Y'])
                
                x_all = score[:, 0]
                y_all = score[:, 1]
                x_pareto = pareto_front[:, 0]
                y_pareto = pareto_front[:, 1]
                
                # Detect Knee point on the pareto
                try :
                    kn = KneeLocator(x_pareto, y_pareto, curve='convex', direction='increasing',S=0)
                    # Knee variable is used 
                    kneeX = kn.knee
                    kneeY = y_pareto[np.where(x_pareto == kneeX)[0][0]]
                    
                    # Get the index of the selected configuration 
                    idName = scorepd.loc[(scorepd['X'] == kneeX) & (scorepd['Y'] == kneeY)].index[0]
                    
                    
                except (IndexError, ValueError)  :
                    try :
                        kn = KneeLocator(x_pareto, y_pareto, curve='concave', direction='increasing',S=0)
                        kneeX = kn.knee
                        kneeY = y_pareto[np.where(x_pareto == kneeX)[0][0]]
                        
                        # Get the index of the selected configuration 
                        idName = scorepd.loc[(scorepd['X'] == kneeX) & (scorepd['Y'] == kneeY)].index[0]
                        
                    except (IndexError, ValueError) :
                        kneeX = pareto_front[len(pareto_front)-1][0]
                        if all(x == x_pareto[0] for x in x_pareto) :
                            kneeY = pareto_front[np.argmin(pareto_front.T[1][:])][1]
                        else :
                            kneeY = scorepd.loc[(scorepd['X'] == kneeX)].iloc[0]['Y']
                        # Get the index of the selected configuration 
                        idName = scorepd.loc[(scorepd['X'] == kneeX) & (scorepd['Y'] == kneeY)].index[0]
                        
                
                if self.verbose == True :
                    print('Knee point : '+str(names[idName]))
                    # Plot Pareto front and knee
                    plt.scatter(x_all, y_all)
                    
                    for i, txt in enumerate(names):
                        plt.annotate(txt, (x_all[i], y_all[i]))
                    
                    plt.title('Pareto front '+str(self.namesStreams[indS])+'. Knee : '+str(names[idName])+' '+str(kneeX)+' '+str(kneeY))
                    plt.plot(x_pareto, y_pareto, color='r')
                    plt.xlabel('n_TP')
                    plt.ylabel('n_FP')
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.vlines(kneeX, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
                    plt.show()
            
                
                self.bestConfig[indS].append(names[idName])
                
        with open(self.output+'/bestConfigsDrift.csv','w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.bestConfig)    
            
        print('End Pareto calculation')
#        print(self.bestConfig)
        
    
    def identify_pareto(self, scores):
        """
        From https://github.com/MichaelAllen1966
        """
        
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item. This will then be compared with all other items
        for i in range(population_size):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if (scores[j][0] >= scores[i][0]) and (scores[j][1] <= scores[i][1]) and (scores[j][0] > scores[i][0]) and (scores[j][1] < scores[i][1]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    def calculate_crowding(self, scores):
        """
        From https://github.com/MichaelAllen1966
        Crowding is based on a vector for each individual
        All dimension is normalised between low and high. For any one dimension, all
        solutions are sorted in order low to high. Crowding for chromsome x
        for that score is the difference between the next highest and next
        lowest score. Total crowding value sums all crowding for all scores
         
        """
        
        population_size = len(scores[:, 0])
        number_of_scores = len(scores[0, :])
    
        # create crowding matrix of population (row) and score (column)
        crowding_matrix = np.zeros((population_size, number_of_scores))
    
        # normalise scores (ptp is max-min)
        normed_scores = (scores - scores.min(0)) / scores.ptp(0)
    
        # calculate crowding distance for each score in turn
        for col in range(number_of_scores):
            crowding = np.zeros(population_size)
    
            # end points have maximum crowding
            crowding[0] = 1
            crowding[population_size - 1] = 1
    
            # Sort each score (to calculate crowding between adjacent scores)
            sorted_scores = np.sort(normed_scores[:, col])
    
            sorted_scores_index = np.argsort(
                normed_scores[:, col])
    
            # Calculate crowding distance for each individual
            crowding[1:population_size - 1] = \
                (sorted_scores[2:population_size] -
                 sorted_scores[0:population_size - 2])
    
            # resort to orginal order (two steps)
            re_sort_order = np.argsort(sorted_scores_index)
            sorted_crowding = crowding[re_sort_order]
    
            # Record crowding distances
            crowding_matrix[:, col] = sorted_crowding
    
        # Sum crowding distances of each score
        crowding_distances = np.sum(crowding_matrix, axis=1)
    
        return crowding_distances
    
    def reduce_by_crowding(self, scores, number_to_select):
        """
        From https://github.com/MichaelAllen1966
        This function selects a number of solutions based on tournament of
        crowding distances. Two members of the population are picked at
        random. The one with the higher croding dostance is always picked
         
        """
        
        population_ids = np.arange(scores.shape[0])
    
        crowding_distances = self.calculate_crowding(scores)
    
        picked_population_ids = np.zeros((number_to_select))
    
        picked_scores = np.zeros((number_to_select, len(scores[0, :])))
    
        for i in range(number_to_select):
    
            population_size = population_ids.shape[0]
    
            fighter1ID = rn.randint(0, population_size - 1)
    
            fighter2ID = rn.randint(0, population_size - 1)
    
            # If fighter # 1 is better
            if crowding_distances[fighter1ID] >= crowding_distances[
                fighter2ID]:
    
                # add solution to picked solutions array
                picked_population_ids[i] = population_ids[
                    fighter1ID]
    
                # Add score to picked scores array
                picked_scores[i, :] = scores[fighter1ID, :]
    
                # remove selected solution from available solutions
                population_ids = np.delete(population_ids, 
                                           (fighter1ID),
                                           axis=0)
    
                scores = np.delete(scores, (fighter1ID), axis=0)
    
                crowding_distances = np.delete(crowding_distances, (fighter1ID),
                                               axis=0)
            else:
                picked_population_ids[i] = population_ids[fighter2ID]
    
                picked_scores[i, :] = scores[fighter2ID, :]
    
                population_ids = np.delete(population_ids, (fighter2ID), axis=0)
    
                scores = np.delete(scores, (fighter2ID), axis=0)
    
                crowding_distances = np.delete(
                    crowding_distances, (fighter2ID), axis=0)
                
        # Convert to integer
        picked_population_ids = np.asarray(picked_population_ids, dtype=int)
        return (picked_population_ids)


