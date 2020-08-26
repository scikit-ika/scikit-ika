import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
import warnings
from kneed import KneeLocator


class BuildDriftKnowledge():
    """
    Description :
        Class to build the pareto knowledge from hyper-parameters configurations evaluated on differents datasets for the drift detector tuning.
        The knowledge consists in the best configuration of hyper-parameters for each dataset.

        The datasets are characterised by meta-features and a knowledge base can be then be built to link these features to the best configurations.

    Parameters :
        results_directory: str
            Path to the directory containing the knowledge files (results of the evaluation of the configurations on example streams)

        names_detectors: list of str
            List of the names of the detectors

        names_streams: list of str
            list of the names of the streams

        n_meta_features: int, default = 15 ((severity, magnitude, interval) * (med, kurto, skew, per10, per90))
            Number of meta-features extracted from the stream
            NOT USED FOR THE MOMENT as we use theoritical meta-features and not measured ones

        knowledge_type: str
            String indicating what knowledge is being calculated (for arf tree tuning or drift detectors)
            NOT USED FOR THE MOMENT, need further implementing to bring the two applications together

        output: str
            Directory path where to save output file

        verbose: bool, default = False
            Print pareto figures if True

        Output:
            Csv file containing the configurations selected for each example stream (each row = 1 stream)


    Example:

        >>> names_stm = ['BernouW1ME0010','BernouW1ME005095','BernouW1ME00509','BernouW1ME0109','BernouW1ME0108','BernouW1ME0208','BernouW1ME0207','BernouW1ME0307','BernouW1ME0306','BernouW1ME0406','BernouW1ME0506','BernouW1ME05506',
        >>>             'BernouW100ME0010','BernouW100ME005095','BernouW100ME00509','BernouW100ME0109','BernouW100ME0108','BernouW100ME0208','BernouW100ME0207','BernouW100ME0307','BernouW100ME0306','BernouW100ME0406','BernouW100ME0506','BernouW100ME05506',
        >>>             'BernouW500ME0010','BernouW500ME005095','BernouW500ME00509','BernouW500ME0109','BernouW500ME0108','BernouW500ME0208','BernouW500ME0207','BernouW500ME0307','BernouW500ME0306','BernouW500ME0406','BernouW500ME0506','BernouW500ME05506']
        >>>
        >>> names_detect = [['PH1','PH2','PH3','PH4','PH5','PH6','PH7','PH8','PH9','PH10','PH11','PH12','PH13','PH14','PH15','PH16'],
        >>>                   ['ADWIN1','ADWIN2','ADWIN3','ADWIN4','ADWIN5','ADWIN6','ADWIN7','ADWIN8','ADWIN9'],
        >>>                   ['DDM1','DDM2','DDM3','DDM4','DDM5','DDM6','DDM7','DDM8','DDM9','DDM10'],
        >>>                   ['SeqDrift21','SeqDrift22','SeqDrift23','SeqDrift24','SeqDrift25','SeqDrift26','SeqDrift27','SeqDrift28','SeqDrift29','SeqDrift210',
        >>>                    'SeqDrift211','SeqDrift212','SeqDrift213','SeqDrift214','SeqDrift215','SeqDrift216','SeqDrift217','SeqDrift218']]
        >>>
        >>> output_dir = os.getcwd()
        >>> directory_path_files = 'examples/pareto_knowledge/ExampleDriftKnowledge' # Available in hyper-param-tuning-examples repository
        >>>
        >>> pareto_build = BuildDriftKnowledge(results_directory=directory_path_files, names_detectors=names_detect, names_streams=names_stm, output=output_dir, verbose=True)
        >>> pareto_build.load_drift_data()
        >>> pareto_build.calculate_pareto()
        >>> pareto_build.best_config



    """

    def __init__(self,
                 results_directory,
                 names_detectors,
                 names_streams,
                 output,
                 # n_meta_features = 15,
                 # knowledge_type = 'Drift',
                 verbose = False):

        if results_directory != None and names_detectors != None and names_streams != None and output != None:
            self.results_directory = results_directory
            self.names_detectors = names_detectors
            self.names_streams = names_streams
            self.output = output
        else :
            raise ValueError('Directory paths or list of detectors names or list of streams missing.')

        self.verbose = verbose

        # self.knowledge_type = knowledge_type

        self.n_detectors = 4

        # self.n_meta_features = n_meta_features

        self.n_streams = len(self.names_streams)

        self.best_config = [[] for ind_s in range(self.n_streams)]

        warnings.filterwarnings("ignore")

    @property
    def best_config(self):
        """ Retrieve the length of the stream.
        Returns
        -------
        int
            The length of the stream.
        """
        return self._best_config

    @best_config.setter
    def best_config(self, best_config):
        """ Set the length of the stream
        Parameters
        ----------
        length of the stream : int
        """
        self._best_config = best_config

    def load_drift_data(self) :
        """
        Function to load the performance data from the csv files


        """

        # Variables for performances of the detectors
        self.scores_perf = []
        self.list_name_detec_ok = []
        for ind_s in range(self.n_streams) :
            self.scores_perf.append([[] for ind_d in range(self.n_detectors)])
            self.list_name_detec_ok.append([[] for ind_d in range(self.n_detectors)])

        # Variables for the meat-features
        self.mean_meta_features = []
        self.std_meta_features = []
        for ind_s in range(self.n_streams) :
            self.mean_meta_features.append([[] for ind_d in range(self.n_detectors)])
            self.std_meta_features.append([[] for ind_d in range(self.n_detectors)])

        ind_s = 0
        # Loop through the streams
        for stream_name in self.names_streams :

            # Open the performance file for the given stream
            stream_perf_file = os.sep.join([self.results_directory, os.sep, stream_name + 'PerfDetectors.csv'])
            with open(stream_perf_file) as csv_data_file:
                data_p = [row for row in csv.reader(csv_data_file)]

            ind_d = 0
            # Loop through the detectors
            for name_detector in self.names_detectors :
                # Loop through the detectors configurations
                for name_ind_detector in name_detector :
                    # Get the row of performances for the given configuration
                    rowind_detector = next((x for x in data_p if x[0] == name_ind_detector), None)
                    # Only if no Nan value in the row (which mean that the detector detected no drifts at some point)
                    if 'nan' not in rowind_detector :
                        # Store the TP number and the FP number for the given detector
                        self.scores_perf[ind_s][ind_d].append([float(rowind_detector[2]),float(rowind_detector[3])])
                        self.list_name_detec_ok[ind_s][ind_d].append(name_ind_detector)
                ind_d += 1


            # # Open the meta-features file for the given stream
            # stream_meta_feat_file = self.results_directory+'\\'+stream_name+'metaFeatDetectors.csv'
            # with open(stream_meta_feat_file) as csv_data_file:
            #     data_mf = [row for row in csv.reader(csv_data_file)]

            # ind_d = 0
            # # Loop through the detectors
            # for name_detector in self.names_detectors :
            #     list_meta_feat_values = [[] for i in range(self.n_meta_features)] # list to store values of each of the meta-features for each detector type
            #     # Loop through the detectors configurations
            #     for name_ind_detector in name_detector :
            #         ind_detec = [i for i in range(len(data_mf)) if data_mf[i][0] == name_ind_detector][0]
            #         # Loop for each meta-feature
            #         for ind_meta_feat in range(self.n_meta_features) :
            #             list_meta_feat_values[ind_meta_feat].append(float(data_mf[ind_detec+ind_meta_feat+1][1]))

            #     self.mean_meta_features[ind_s][ind_d] = np.nanmean(list_meta_feat_values, axis = 1)
            #     self.std_meta_features[ind_s][ind_d] = np.nanstd(list_meta_feat_values, axis = 1)

            #     ind_d += 1

            ind_s += 1

        # print('end')

    def process_meta_features(self):
        print('Start process meta-features')

        # TODO : to come

    def calculate_pareto(self) :
        """
        Function to calculate the Pareto front and detect the knee point

        """
        if self.verbose == True :
            print('Start Pareto calculation')
        for ind_s in range(self.n_streams) :
            for ind_d in range(self.n_detectors) :
                names = self.list_name_detec_ok[ind_s][ind_d]

                score = np.array(self.scores_perf[ind_s][ind_d])

                # Calculate pareto front
                pareto = self.identify_pareto(score)
                # print ('Pareto front index values')
                # print ('Points on Pareto front: \n',pareto)

                pareto_front = score[pareto]
                # print ('\nPareto front scores')
                # print (pareto_front)

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
                    knee_x = kn.knee
                    knee_y = y_pareto[np.where(x_pareto == knee_x)[0][0]]

                    # Get the index of the selected configuration
                    id_name = scorepd.loc[(scorepd['X'] == knee_x) & (scorepd['Y'] == knee_y)].index[0]


                except (IndexError, ValueError)  :
                    try :
                        kn = KneeLocator(x_pareto, y_pareto, curve='concave', direction='increasing',S=0)
                        knee_x = kn.knee
                        knee_y = y_pareto[np.where(x_pareto == knee_x)[0][0]]

                        # Get the index of the selected configuration
                        id_name = scorepd.loc[(scorepd['X'] == knee_x) & (scorepd['Y'] == knee_y)].index[0]

                    except (IndexError, ValueError) :
                        knee_x = pareto_front[len(pareto_front)-1][0]
                        if all(x == x_pareto[0] for x in x_pareto) :
                            knee_y = pareto_front[np.argmin(pareto_front.T[1][:])][1]
                        else :
                            knee_y = scorepd.loc[(scorepd['X'] == knee_x)].iloc[0]['Y']
                        # Get the index of the selected configuration
                        id_name = scorepd.loc[(scorepd['X'] == knee_x) & (scorepd['Y'] == knee_y)].index[0]


                if self.verbose == True :
                    print('Knee point : '+str(names[id_name]))
                    # Plot Pareto front and knee
                    plt.scatter(x_all, y_all)

                    for i, txt in enumerate(names):
                        plt.annotate(txt, (x_all[i], y_all[i]))

                    plt.title('Pareto front '+ str(self.names_streams[ind_s]) +'. Knee : '+ str(names[id_name]) + ' ' + str(knee_x) + ' ' + str(knee_y))
                    plt.plot(x_pareto, y_pareto, color='r')
                    plt.xlabel('n_TP')
                    plt.ylabel('n_FP')
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.vlines(knee_x, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
                    plt.show()


                self.best_config[ind_s].append(names[id_name])

        with open(self.output+'/bestConfigsDrift.csv.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.best_config)

        if self.verbose == True :
            print('End Pareto calculation')
#        print(self.best_config)


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

            sorted_scores_index = np.argsort(normed_scores[:, col])

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
            if crowding_distances[fighter1ID] >= crowding_distances[fighter2ID]:

                # add solution to picked solutions array
                picked_population_ids[i] = population_ids[fighter1ID]

                # Add score to picked scores array
                picked_scores[i, :] = scores[fighter1ID, :]

                # remove selected solution from available solutions
                population_ids = np.delete(population_ids,
                                           (fighter1ID),
                                           axis=0)

                scores = np.delete(scores, (fighter1ID), axis=0)

                crowding_distances = np.delete(crowding_distances,
                                               (fighter1ID),
                                               axis=0)
            else:
                picked_population_ids[i] = population_ids[fighter2ID]
                picked_scores[i, :] = scores[fighter2ID, :]
                population_ids = np.delete(population_ids, (fighter2ID), axis=0)
                scores = np.delete(scores, (fighter2ID), axis=0)
                crowding_distances = np.delete(crowding_distances, (fighter2ID), axis=0)

        # Convert to integer
        picked_population_ids = np.asarray(picked_population_ids, dtype=int)
        return (picked_population_ids)
