# -*- coding: utf-8 -*-

#import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
from kneed import KneeLocator
import warnings


class buildTreesKnowledge():
    """
    Description
    Class to build the pareto knowledge from hyper-parameters configurations evaluated on differents datasets for tuning the number of trees in ARF. 
    The knowledge consists in the best configuration of hyper-parameters for each dataset. 
    
    The datasets are characterised by meta-features and a knowledge base can be then be built to link these features to the best configurations. 
    
    Parameters :
        results_file: str 
            Path to the file containing the knowledge files (results of the evaluation of the configurations on example streams)
            See example in hyper_param_tuning_examples to format the file. 
        
        list_perc_redund: list of float
            List of percentages of redundance used in the example streams
        
        list_models: list of str
            List of the names of the ARF configurations tested on the streams
        
        output: str
            Directory path where to save output file
            
        verbose: bool, default = False
            Print pareto figures if True 
    
    Output :
            Csv file containing the configurations selected for each example stream (each row = 1 stream)
     
    Example
        -------- 
        >>> names = ['ARF10','ARF30','ARF60','ARF70','ARF90','ARF100','ARF120','ARF150','ARF200']    
        >>> perc_redund = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        
        >>> output_dir = os.getcwd()
        >>> name_file =' /examples/pareto_knowledge/ExamplesTreesKnowledge/Results10-200.csv'
        
        >>> paretoBuild = buildTreesKnowledge(results_file = name_file, list_perc_redund = perc_redund, list_models = names, output = output_dir, verbose = True)
        >>> paretoBuild.load_drift_data()
        >>> paretoBuild.calculatePareto()
        >>> paretoBuild.bestConfig
    """
    
    def __init__(self,
             results_file,
             list_perc_redund,
             list_models,
             output,
             verbose = False):
        
        if results_file != None and list_perc_redund != None and list_models != None and output != None :
            self.results_file = results_file
            self.list_perc_redund = list_perc_redund
            self.list_models = list_models
            self.output = output
        else :
            raise ValueError('Directory path or list of detectors names or list of streams missing.')
        
        self.n_perc_redund = len(list_perc_redund)
        self.n_models = len(list_models)
        
        self.verbose = verbose
        
        self.bestConfig = [[self.list_perc_redund[indS],np.nan] for indS in range(self.n_perc_redund)]
        
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
        Function to load the performance data from the csv file
    
        """
        # IMPORTATION OF DATA FROM CSV
        with open(self.results_file) as csvDataFile:
            data = [row for row in csv.reader(csvDataFile)]
            
        self.scores = []
        
        for perc in range(self.n_perc_redund):
            score = []
            kappa_mean = np.array([float(data[i+2][perc+1]) for i in range(self.n_models)])
            ramh_mean = np.array([float(data[i+2][perc+13]) for i in range(self.n_models)])
                
            score.append(kappa_mean)
            score.append(ramh_mean)
                
            self.scores.append(np.array(score).T)
        
    def calculatePareto(self) :
        """
        Function to calculate the Pareto front and detect the knee point
        
        """
        
        if self.verbose == True :
            fig = plt.figure()
            fig.suptitle('Pareto fronts for ARF meta-knowledge building')
        
           
        indr = 0
        for score in self.scores :
            print("########################## Perc redun = "+str(perc_redund[indr])+" ##########################")
                  
            # Calculate pareto front
            pareto = self.identify_pareto(score)
            print ('Pareto front index vales')
            print ('Points on Pareto front: \n',pareto)
            
            pareto_front = score[pareto]
            print ('\nPareto front scores')
            print (pareto_front)
            
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values
            
            x_all = score[:, 0]
            y_all = score[:, 1]
            x_pareto = pareto_front[:, 0]
            y_pareto = pareto_front[:, 1]
            
            scorepd = pd.DataFrame(score,columns = ['X' , 'Y'])
            
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
            
            print('Knee point : '+str(self.list_models[idName]))
            self.bestConfig[indr][1] = str(self.list_models[idName])
            
            if self.verbose == True :
#                # Print individual figures
#                print('Knee point : '+str(self.list_models[idName]))
#                plt.scatter(x_all, y_all)
#                
#                for i, txt in enumerate(names):
#                    plt.annotate(txt, (x_all[i], y_all[i]))
#                
#                plt.title('Pareto front')
#                plt.plot(x_pareto, y_pareto, color='r')
#                plt.xlabel('Kappa')
#                plt.ylabel('RAM hours')
#                xmin, xmax, ymin, ymax = plt.axis()
#                plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
#                plt.show()
                
                # Print global figure
                ax = fig.add_subplot(5,2,indr+1)
                ax.scatter(x_all, y_all)
                for i, txt in enumerate(self.list_models):
                    ax.annotate(txt, (x_all[i], y_all[i]))
                ax.plot(x_pareto, y_pareto, color='r')
                ax.set_title('Perc redun = '+str(self.list_perc_redund[indr]))
                ax.set_xlabel('Kappa')
                ax.set_ylabel('RAM hours')
                xmin, xmax, ymin, ymax = ax.axis()
                ax.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
                
            indr = indr+1
                
        if self.verbose == True :    
            plt.show()
        
        
        with open(self.output+'/bestConfigsTreesARF.csv','w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.bestConfig) 
        
    ###########################################################################
    # FUNCTIONS FOR PARETO FRONT
    ###########################################################################
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
    
