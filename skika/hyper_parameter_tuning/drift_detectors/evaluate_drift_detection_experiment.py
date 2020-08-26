import os
import numpy as np
from copy import deepcopy
import csv

from scipy.spatial import distance

dirpath = os.getcwd()

class EvaluateDriftDetection():
    """ Prequential evaluation method with adaptive tuning of hyper-parameters for drift detector tuning.

    Description :
        Prequential evaluation method with adaptive tuning of hyper-parameters for drift detector tuning.
        This class enable to evaluate the performance of an adaptive tuning of drift detectors based on a meta-knowledge base built from results from class
        evaluate_drift_detection_knowledge.


    Parameters :
        list_drifts_detectors: list of drift detector object
            List of drift detectors to evaluate. Each detector is used for warning and drift detection.
            If the detector doesn't give both directly, one should pass two detectors, the first one for the warning detection and the second one for the drift detection.

        list_names_drifts_detectors: list of drift detectors names to evaluate.

        kBase:
            Knowledge base containing meta-features values matched with drift detectors best configurations.

        dict_k_base: dict
            Dict linking the name of the configurations with the warning and drift detectors

        adapt: list of int
            List of int to indicate if the drift detector configuration should be :
                - 0 : Not adapted
                - 1 : Adapted given knowledge with meta-learning
            Must be the same size as list_drifts_detectors.

        win_adapt_size: list
            Length of the sliding window to store and compare meta-features with knowledge.

        stream: stream object
            Stream on which the detectors are evaluated (Bernoulli stream).

        n_runs: int
            Number of runs to process
            The results will be given as mean and std over n_runs.

        name_file: str
            Name of the file to save the results.

        Output:
            Csv files containing the performance results.

    Example:

        See https://github.com/scikit-ika/hyper-param-tuning-examples


    """

    def __init__(self,
                 list_drifts_detectors,
                 list_names_drifts_detectors,
                 adapt,
                 k_base,
                 dict_k_base,
                 win_adapt_size,
                 stream,
                 n_runs,
                 name_file):

        self.name_file = name_file
        if (self.name_file == None) or not(isinstance(self.name_file, str)) :
            raise ValueError("Attribute 'name_file' must be specified and must be a string, passed {}".format(type(self.name_file)))

        self.list_drifts_detectors = list_drifts_detectors
        self.list_names_drifts_detectors = list_names_drifts_detectors
        self.adapt = adapt
        self.k_base = k_base
        self.dict_k_base = dict_k_base
        self.win_adapt_size = win_adapt_size
        self.stream = stream
        self.n_runs = n_runs

        self.lenght_stream = stream.n_samples
        self.true_positions = stream.list_positions

        self.n_drift_detectors = len(self.list_drifts_detectors)

        self.init_list_drifts_detectors = deepcopy(self.list_drifts_detectors)
#        print(self.init_list_drifts_detectors)

        # Results variables
        self.mean_n_TP = 0
        self.mean_n_FP = 0

    @property
    def mean_n_TP(self):
        """ Retrieve the mean number of true Positives
        Returns
        -------
        float
            The number of True Positives.
        """
        return self._mean_n_TP

    @mean_n_TP.setter
    def mean_n_TP(self, mean_n_TP):
        """ Set the mean number of true Positives
        Parameters
        ----------
        mean number of true Positives : float
        """
        self._mean_n_TP = mean_n_TP

    @property
    def mean_n_FP(self):
        """ Retrieve the mean number of False Positives
        Returns
        -------
        float
            The number of False Positives.
        """
        return self._mean_n_FP

    @mean_n_FP.setter
    def mean_n_FP(self, mean_n_FP):
        """ Set the mean number of False Positives
        Parameters
        ----------
        mean number of False Positives : float
        """
        self._mean_n_FP = mean_n_FP



    def run(self):
        self.prepare_evaluation()
        self.evaluate()

        return self.mean_n_detected, self.mean_n_TP, self.mean_n_FP, self.mean_delays

    def prepare_evaluation(self) :
        """
         Prepare variables and stream for drift detection and meta-features extraction

        """

        ## Variables to store perf of all runs
        self.list_n_detected = []
        self.list_n_TP = []
        self.list_n_FP = []
        self.list_delays = []

        for i_detector in range(self.n_drift_detectors) :
            self.list_n_detected.append([])
            self.list_n_TP.append([])
            self.list_n_FP.append([])
            self.list_delays.append([])

        ## Variables to store perf of each run
        self.list_instances = []

        self.detector_warning_detected = []
        self.detector_drift_detected = []

        self.n_detected_drifts = []
        self.n_detected_warning = []
        self.n_drift = []

        self.detected_positions = []
        self.warning_positions = []

        self.measured_meta_feat = []

        self.n_TP = []
        self.n_FP = []
        self.list_TP = []
        self.list_FP = []
        self.delays= []

        for i_detector in range(self.n_drift_detectors) :
            self.detector_warning_detected.append(False)
            self.detector_drift_detected.append(False)

            self.n_detected_drifts.append(0)
            self.n_detected_warning.append(0)
            self.n_drift.append(0)

            self.detected_positions.append([])
            self.warning_positions.append([])

            self.n_TP.append(0)
            self.list_TP.append([])
            self.n_FP.append(0)
            self.list_FP.append([])
            self.delays.append([])

            self.measured_meta_feat.append([[],[]])

        self.current_config = deepcopy(self.list_names_drifts_detectors)

        self.true_positions = self.stream.list_positions
        self.current_global_count = 0

    def reset_run(self):

        # Variables to store perf of each run
        self.list_instances = []

        self.detector_warning_detected = []
        self.detector_drift_detected = []

        self.n_detected_drifts = []
        self.n_detected_warning = []
        self.n_drift = []

        self.detected_positions = []
        self.warning_positions = []

        self.n_TP = []
        self.list_TP = []
        self.n_FP = []
        self.list_FP = []
        self.delays = []

        self.measured_meta_feat = []


        for i_detector in range(self.n_drift_detectors) :
            self.detector_warning_detected.append(False)
            self.detector_drift_detected.append(False)

            self.n_detected_drifts.append(0)
            self.n_detected_warning.append(0)
            self.n_drift.append(0)

            self.detected_positions.append([])
            self.warning_positions.append([])

            self.n_TP.append(0)
            self.list_TP.append([])
            self.n_FP.append(0)
            self.list_FP.append([])
            self.delays.append([])

            self.measured_meta_feat.append([[],[]])


        # Reset detectors
        ind = 0
        for drift_detector in self.list_drifts_detectors :
            if len(drift_detector) == 2 :

                drift_detector[0] = deepcopy(self.init_list_drifts_detectors[ind][0])
                drift_detector[1] = deepcopy(self.init_list_drifts_detectors[ind][1])

            elif len(drift_detector) == 1 :
                drift_detector[0] = deepcopy(self.init_list_drifts_detectors[ind][0])

            ind += 1

        print(drift_detector[0])


    def evaluate(self) :
        """
         Evaluate the detectors on the stream

        """
        for run in range(self.n_runs) :

            print('Run '+str(run)+' start')

            self.stream.prepare_for_use()

            self.current_global_count = 0


            for i in range(self.lenght_stream):

                current_sample = self.stream.next_sample(batch_size = 1)

                self.list_instances.append(current_sample[0])

                i_detector = 0
                for drift_detector in self.list_drifts_detectors :

                    if len(drift_detector) == 2 :
                        # Two different ways to add element and check for change depending if using scikit_multiflow or tornado detectors
                        try :
                            # Scikit_multiflow framework
                            drift_detector[0].add_element(int(not(current_sample[0])))
                            drift_detector[1].add_element(int(not(current_sample[0])))

                            self.detector_warning_detected[i_detector] = drift_detector[0].detected_change()
                            self.detector_drift_detected[i_detector] = drift_detector[1].detected_change()

                        except AttributeError :
                            # Tornado framework
                            f, self.detector_warning_detected[i_detector] = drift_detector[0].run(bool(int(current_sample[0])))
                            f, self.detector_drift_detected[i_detector] = drift_detector[1].run(bool(int(current_sample[0])))

                        # Warning detection
                        if self.detector_warning_detected[i_detector]:
                            self.n_detected_warning[i_detector] += 1
                            self.warning_positions[i_detector].append(i)

                            drift_detector[0].reset()
                            self.detector_warning_detected[i_detector] = False

                        # Drift detection
                        if self.detector_drift_detected[i_detector]:

                            self.n_detected_drifts[i_detector] += 1
                            self.detected_positions[i_detector].append(i)

                            drift_detector[1].reset()
                            self.detector_drift_detected[i_detector] = False

                    elif len(drift_detector) == 1 :
                        # Two different ways to add element and check for change depending if using scikit_multiflow or tornado detectors
                        try :
                            # Scikit_multiflow framework
                            drift_detector[0].add_element(int(not(current_sample[0])))

                            self.detector_warning_detected[i_detector] = drift_detector[0].detected_warning_zone()
                            self.detector_drift_detected[i_detector] = drift_detector[0].detected_change()

                        except AttributeError :
                            # Tornado framework
                            self.detector_warning_detected[i_detector], self.detector_drift_detected[i_detector] = drift_detector[0].run(bool(int(current_sample[0])))

                        # Warning detection
                        if self.detector_warning_detected[i_detector] :
                            self.n_detected_warning[i_detector] += 1
                            self.warning_positions[i_detector].append(i)

                            drift_detector[0].reset()
                            self.detector_warning_detected[i_detector] = False

                        # Drift detection
                        if self.detector_drift_detected[i_detector] :

                            self.n_detected_drifts[i_detector] += 1
                            self.detected_positions[i_detector].append(i)

                            drift_detector[0].reset()
                            self.detector_drift_detected[i_detector] = False

                    ########### Adaptive Part ###############

                    # If adapt == 1 : adapt drift detector based on meta-knowledge
                    if self.adapt[i_detector] == 1 :

                        # If we are on a true drift position -> To be replaced later with actuel drift detection
                        if self.current_global_count in self.true_positions :
                            self.n_drift[i_detector] += 1

                            # Store meta-feat for win_adapt_size first drifts
                            if self.n_drift[i_detector] < self.win_adapt_size :
                                self.measured_meta_feat[i_detector][0].append(self.stream.current_drift_severity)
                                self.measured_meta_feat[i_detector][1].append(self.stream.current_drift_magnitude)
                            # Store meta-feat with window slidding (pop last element and add new one)
                            else :
                                self.measured_meta_feat[i_detector][0].pop(0)
                                self.measured_meta_feat[i_detector][0].append(self.stream.current_drift_severity)

                                self.measured_meta_feat[i_detector][1].pop(0)
                                self.measured_meta_feat[i_detector][1].append(self.stream.current_drift_magnitude)

                            # If we have seen more drift than the initialisation requires, compare to metaK and adapt if needed
                            if self.n_drift[i_detector] > self.win_adapt_size :
                                mean_measured_meta_feat = np.mean(self.measured_meta_feat[i_detector],axis=1)

                                # Calculate distances between extracted meta-features set and knowledge base and get the index of the minimum distance
                                distances = distance.cdist(self.k_base[0],np.array([mean_measured_meta_feat]),'euclidean')
                                ind_min_dist = list(distances).index(min(distances))

                                new_config = deepcopy(self.k_base[1][ind_min_dist])

                                if new_config != self.current_config[i_detector] :

                                    if len(drift_detector) == 2 :
                                        drift_detector[0] = deepcopy(self.dict_k_base[new_config][0])
                                        drift_detector[1] = deepcopy(self.dict_k_base[new_config][1])

                                        drift_detector[0].reset()
                                        drift_detector[1].reset()
                                    elif len(drift_detector) == 1 :
                                        drift_detector[0] = deepcopy(self.dict_k_base[new_config][0])

                                        drift_detector[0].reset()

                                    self.current_config[i_detector] = deepcopy(new_config)
#                                    print(self.current_config)


                    i_detector += 1


                self.current_global_count += 1
#            print(self.list_drifts_detectors[2][0])



            self.process_results_run()

        self.process_results_global()

    def process_results_run(self):

        #########
        # Prepare variables for results processing
        #########

        true_detected_positions = []
        temp_list = []
        for i_detector in range(self.n_drift_detectors) :
            true_detected_positions.append([])
            temp_list.append(deepcopy(self.true_positions))

        #########
        # Process results from drift detection
        #########

        for i_detector in range(self.n_drift_detectors) :
            # Get a list of TP and FP detected drifts
            for j in range(len(self.detected_positions[i_detector])) :
                try :
                    true_position = min([num for num in temp_list[i_detector] if num<self.detected_positions[i_detector][j]], key=lambda x:abs(x-self.detected_positions[i_detector][j]))
                    true_detected_positions[i_detector].append(true_position)

                    # Consider TP is delay < larger width of drift in stream
                    if self.detected_positions[i_detector][j]-true_position <= 500 :
                        self.list_TP[i_detector].append(self.detected_positions[i_detector][j])
                        self.delays[i_detector].append(self.detected_positions[i_detector][j]-true_position)

                    else :
                        self.list_FP[i_detector].append(self.detected_positions[i_detector][j])

                    ind = temp_list[i_detector].index(true_position)

                    if ind > 0:
                        del temp_list[i_detector][0:ind+1]
                    else :
                        del temp_list[i_detector][ind]

                except ValueError :
                    self.list_FP[i_detector].append(self.detected_positions[i_detector][j])

            # Calculate the number of TP and FP detected drifts
            self.n_TP[i_detector] = len(self.list_TP[i_detector])
            self.n_FP[i_detector] = len(self.list_FP[i_detector])

            self.list_n_detected[i_detector].append(self.n_detected_drifts[i_detector])
            self.list_n_TP[i_detector].append(self.n_TP[i_detector])
            self.list_n_FP[i_detector].append(self.n_FP[i_detector])
            self.list_delays[i_detector].append(np.nanmean(self.delays[i_detector]))


#        try :
#            new_dirpath = dirpath+'\\ResultsDriftExperiment\\'
#            os.mkdir(new_dirpath)
#        except :
#            pass
#
#        for i_detector in range(self.n_drift_detectors) :
#            # Second file with list tot
#            list_tot2 = [[self.n_detected_drifts[i_detector]],self.list_TP[i_detector],self.list_FP[i_detector],self.delays[i_detector]]
#
#            with open(new_dirpath+self.list_names_drifts_detectors[i_detector]+'_PerfDetectors.csv', 'w', newline='') as f:
#                writer = csv.writer(f)
#                for values in zip_longest(*list_tot2):
#                    writer.writerow(values)


        # Reset for next run
        self.reset_run()


    def process_results_global(self):
        #########
        #
        # Process mean results + save csv
        #
        #########

        self.mean_n_detected = [np.mean(np.array(elem)) for elem in self.list_n_detected]
        self.mean_n_TP = [np.mean(np.array(elem)) for elem in self.list_n_TP]
        self.mean_n_FP = [np.mean(np.array(elem)) for elem in self.list_n_FP]
        self.mean_delays = [np.mean(np.array(elem)) for elem in self.list_delays]


        self.std_n_detected = [np.std(np.array(elem)) for elem in self.list_n_detected]
        self.std_n_TP = [np.std(np.array(elem)) for elem in self.list_n_TP]
        self.std_n_FP = [np.std(np.array(elem)) for elem in self.list_n_FP]
        self.std_delays = [np.std(np.array(elem)) for elem in self.list_delays]

        ind = 0
        for name in self.list_names_drifts_detectors :
            if self.adapt[ind] == True :
                self.list_names_drifts_detectors[ind] = self.list_names_drifts_detectors[ind]+'Adapt'
            ind += 1

        # Create csv file
        head = ['detector',
                'mean_n_detect','mean_n_TP','mean_n_FP','mean_delay',
                'std_n_detect','std_n_TP','std_n_FP','std_delay']
        list_tot = [self.list_names_drifts_detectors,
                    self.mean_n_detected,self.mean_n_TP,self.mean_n_FP,self.mean_delays,
                    self.std_n_detected,self.std_n_TP,self.std_n_FP,self.std_delays]

        list_tot = list(map(list, zip(*list_tot)))
        list_tot.insert(0,head)

        try :
            new_dirpath = dirpath + os.sep + 'ResultsDriftExperiment' + os.sep
            os.mkdir(new_dirpath)
        except :
            pass

        with open(new_dirpath+self.name_file+'_PerfDetectors.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list_tot)
