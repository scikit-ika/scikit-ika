import os
import numpy as np
from copy import copy
import csv

from scipy.stats import kurtosis, skew
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.page_hinkley import PageHinkley
from skika.data.bernoulli_stream import BernoulliStream

dirpath = os.getcwd()

class EvaluateDriftDetection():
    """ Prequential evaluation method to collect knowledge for drift detector tuning.

    Description :
        Class to evaluate the performance of the drift detection for the knowledge computation.
        Performance is evaluate with the numbers of TP and FP detections.

    Parameters :
        list_drifts_detectors : list of drift detector object
            List of drift detectors to evaluate.
            Each detector is used for warning and drift detection.
            If the detector doesn't handle both directly, one should pass two detectors, the first one for the warning detection and the second one for the drift detection.

        list_names_drifts_detectors: list of str.
            List of drift detectors names to evaluate.

        stream: stream object
            Stream on which the detectors are evaluated.

        n_runs: int
            Number of runs to process.
            The results will be given as mean and std over n_runs.

        name_file: str
            Name of the file to save the results.

        Output:
            Csv files containing the performance results to be exploited to build the knowledge base.

    Example:

        See https://github.com/scikit-ika/hyper-param-tuning-examples

    """

    def __init__(self,
                 list_drifts_detectors = [[ADWIN(delta=0.5), ADWIN(delta=0.05)],
                                          [DDM(min_num_instances=100, warning_level=2.0, out_control_level=3.0)],
                                          [PageHinkley(min_instances=100, delta=0.05, threshold=50, alpha=0.9999), PageHinkley(min_instances=100, delta=0.05, threshold=100, alpha=0.9999)]],
                                          # [SeqDrift2ChangeDetector(delta=0.5, block_size=100), SeqDrift2ChangeDetector(delta=0.05, block_size=100)]],
                 list_names_drifts_detectors = ['ADWIN','DDM','PH'],
                 stream = BernoulliStream(drift_period = 1000, n_drifts = 50, widths_drifts = [1], mean_errors = [0.1,0.9]),
                 n_runs = 1,
                 name_file = None):

        self.name_file = name_file
        if (self.name_file == None) or not(isinstance(self.name_file, str)) :
            raise ValueError("Attribute 'name_file' must be specified and must be a string, passed {}".format(type(self.name_file)))

        self.list_drifts_detectors = list_drifts_detectors
        self.list_names_drifts_detectors = list_names_drifts_detectors
        self.stream = stream
        self.n_runs = n_runs

        self.lenght_stream = stream.n_samples
        self.true_positions = stream.list_positions

        self.n_drift_detectors = len(self.list_drifts_detectors)

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

        # Variables for stats of meta_features measures of all runs
        self.stats_severity_list = []
        self.stats_magnitude_list = []
        self.stats_interval_list = []

        for i_detector in range(self.n_drift_detectors) :
            self.stats_severity_list.append([])
            self.stats_magnitude_list.append([])
            self.stats_interval_list.append([])

        ## Variables to store perf of each run
        self.list_instances = []

        self.detector_warning_detected = []
        self.detector_drift_detected = []

        self.n_detected_drifts = []
        self.n_detected_warning = []

        self.detected_positions = []
        self.warning_positions = []

        self.n_TP = []
        self.list_TP = []
        self.n_FP = []
        self.list_FP = []
        self.delays = []

        # Variables for meta_features measures
        self.warning_in_progress = []
        self.interval_in_progress = []
        self.n_instances_warning = []
        self.n_missclass_warning = []
        self.severity_list = []
        self.magnitude_list = []
        self.sample_drift = []
        self.instances_before_drift = []
        self.instances_after_drift = []
        self.interval = []
        self.interval_list = []

        for i_detector in range(self.n_drift_detectors) :
            self.detector_warning_detected.append(False)
            self.detector_drift_detected.append(False)

            self.n_detected_drifts.append(0)
            self.n_detected_warning.append(0)

            self.detected_positions.append([])
            self.warning_positions.append([])

            self.n_TP.append(0)
            self.list_TP.append([])
            self.n_FP.append(0)
            self.list_FP.append([])
            self.delays.append([])

            self.warning_in_progress.append(False)
            self.interval_in_progress.append(False)
            self.n_instances_warning.append(0)
            self.n_missclass_warning.append(0)
            self.severity_list.append([])
            self.magnitude_list.append([])
            self.sample_drift.append([])
            self.instances_before_drift.append([])
            self.instances_after_drift.append([])
            self.interval.append(0)
            self.interval_list.append([])


        self.true_positions = self.stream.list_positions

        # Magnitude measure
        self.window_size = 50
        self.safety_interval = 100

    def reset_run(self):

        # Variables to store perf of each run
        self.list_instances = []

        self.n_detected_drifts = []
        self.n_detected_warning = []

        self.detected_positions = []
        self.warning_positions = []

        self.n_TP = []
        self.list_TP = []
        self.n_FP = []
        self.list_FP = []
        self.delays = []

        # Variables for meta_features measures
        self.warning_in_progress = []
        self.interval_in_progress = []
        self.n_instances_warning = []
        self.n_missclass_warning = []
        self.severity_list = []
        self.magnitude_list = []
        self.sample_drift = []
        self.instances_before_drift = []
        self.instances_after_drift = []
        self.interval = []
        self.interval_list = []


        for i_detector in range(self.n_drift_detectors) :
            self.detector_warning_detected.append(False)
            self.detector_drift_detected.append(False)

            self.n_detected_drifts.append(0)
            self.n_detected_warning.append(0)

            self.detected_positions.append([])
            self.warning_positions.append([])

            self.n_TP.append(0)
            self.list_TP.append([])
            self.n_FP.append(0)
            self.list_FP.append([])
            self.delays.append([])

            self.warning_in_progress.append(False)
            self.interval_in_progress.append(False)
            self.n_instances_warning.append(0)
            self.n_missclass_warning.append(0)
            self.severity_list.append([])
            self.magnitude_list.append([])
            self.sample_drift.append([])
            self.instances_before_drift.append([])
            self.instances_after_drift.append([])
            self.interval.append(0)
            self.interval_list.append([])

        # Reset detectors
        for drift_detector in self.list_drifts_detectors :
                if len(drift_detector) == 2 :
                    drift_detector[0].reset()
                    drift_detector[1].reset()

                elif len(drift_detector) == 1 :
                    drift_detector[0].reset()

    def evaluate(self) :
        """
         Evaluate the detectors on the stream

        """
        for run in range(self.n_runs) :

            print('Run '+str(run)+' start')

            self.stream.prepare_for_use()

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
                            self.warning_in_progress[i_detector] = True
                            self.n_instances_warning[i_detector] = 0

                            drift_detector[0].reset()
                            self.detector_warning_detected[i_detector] = False


                        # Drift detection
                        if self.detector_drift_detected[i_detector]:

                            self.interval_measure(i_detector)

                            self.n_detected_drifts[i_detector] += 1
                            self.detected_positions[i_detector].append(i)


                            # Magnitude measure
                            self.sample_drift[i_detector].append(True)
                            self.instances_before_drift[i_detector].append(self.list_instances[i-self.window_size-self.safety_interval:i-self.safety_interval])
                            self.instances_after_drift[i_detector].append([])


                            # Severity measure
                            self.severity_measure(i_detector)

                            self.warning_in_progress[i_detector] = False

                            drift_detector[1].reset()
                            self.detector_drift_detected[i_detector] = False


                        if self.warning_in_progress[i_detector]:
                            self.n_instances_warning[i_detector] += 1

                        if self.interval_in_progress[i_detector] == True :
                            self.interval[i_detector] += 1

                        # Magnitude measure
                        for ind in range(len(self.sample_drift[i_detector])) :
                            if self.sample_drift[i_detector][ind] == True :
                                if (i>self.detected_positions[i_detector][ind]+self.safety_interval) and (len(self.instances_after_drift[i_detector][ind]) < self.window_size):
                                    self.instances_after_drift[i_detector][ind].append(current_sample[0])
                                elif len(self.instances_after_drift[i_detector][ind]) == self.window_size :
                                    self.magnitude_measure(i_detector, self.instances_before_drift[i_detector][ind], self.instances_after_drift[i_detector][ind])
                                    self.sample_drift[i_detector][ind] = False

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
                            self.warning_in_progress[i_detector] = True
                            self.n_instances_warning[i_detector] = 0

                            drift_detector[0].reset()
                            self.detector_warning_detected[i_detector] = False


                        # Drift detection
                        if self.detector_drift_detected[i_detector] :

                            self.interval_measure(i_detector)

                            self.n_detected_drifts[i_detector] += 1
                            self.detected_positions[i_detector].append(i)

                            # Magnitude measure
                            self.sample_drift[i_detector].append(True)
                            self.instances_before_drift[i_detector].append(self.list_instances[i-self.window_size-self.safety_interval:i-self.safety_interval])
                            self.instances_after_drift[i_detector].append([])

                            # Severity measure
                            self.severity_measure(i_detector)

                            self.warning_in_progress[i_detector] = False

                            drift_detector[0].reset()
                            self.detector_drift_detected[i_detector] = False


                        if self.warning_in_progress[i_detector]:
                            self.n_instances_warning[i_detector] += 1

                        if self.interval_in_progress[i_detector] == True :
                            self.interval[i_detector] += 1

                        # Magnitude measure
                        for ind in range(len(self.sample_drift[i_detector])) :
                            if self.sample_drift[i_detector][ind] == True :
                                if (i>self.detected_positions[i_detector][ind]+self.safety_interval) and (len(self.instances_after_drift[i_detector][ind]) < self.window_size):
                                    self.instances_after_drift[i_detector][ind].append(current_sample[0])
                                elif len(self.instances_after_drift[i_detector][ind]) == self.window_size :
                                    self.magnitude_measure(i_detector, self.instances_before_drift[i_detector][ind], self.instances_after_drift[i_detector][ind])
                                    self.sample_drift[i_detector][ind] = False

                    i_detector += 1

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
            temp_list.append(copy(self.true_positions))

        #########
        # Process results from drift detection
        #########

        for i_detector in range(self.n_drift_detectors) :
            # Get a list of TP detected drifts
            for j in range(len(self.detected_positions[i_detector])) :
                try :
                    true_position = min([num for num in temp_list[i_detector] if num<self.detected_positions[i_detector][j]], key=lambda x:abs(x-self.detected_positions[i_detector][j]))
                    true_detected_positions[i_detector].append(true_position)
                    self.list_TP[i_detector].append(self.detected_positions[i_detector][j])

                    self.delays[i_detector].append(self.detected_positions[i_detector][j]-true_position)

                    ind = temp_list[i_detector].index(true_position)

                    if ind > 0:
                        del temp_list[i_detector][0:ind+1]
                    else :
                        del temp_list[i_detector][ind]

                except ValueError :
                    pass


            self.n_TP[i_detector] = len(self.list_TP[i_detector])
            self.n_FP[i_detector] = len(self.detected_positions[i_detector]) - self.n_TP[i_detector]

            self.list_n_detected[i_detector].append(self.n_detected_drifts[i_detector])
            self.list_n_TP[i_detector].append(self.n_TP[i_detector])
            self.list_n_FP[i_detector].append(self.n_FP[i_detector])
            self.list_delays[i_detector].append(np.mean(self.delays[i_detector]))

            # Exceptions raised if not enought drift detected to calculate stats, we decide to add Nan to the results then
            try :
                # Stats of meta-features : median, kurtosis, skewness, perc10, perc90
                self.stats_severity_list[i_detector].append([np.median(self.severity_list[i_detector]),
                                                            kurtosis(self.severity_list[i_detector]),
                                                            skew(self.severity_list[i_detector]),
                                                            np.percentile(self.severity_list[i_detector],10),
                                                            np.percentile(self.severity_list[i_detector],90)])
            except:
                #Debug
#                print('Severity')
#                print('Detector : '+str(self.list_names_drifts_detectors[i_detector]))
#                print('Stream : '+str(self.name_file))
#                print('Nombre de drifts detected : '+str(len(self.detected_positions[i_detector])))

                self.stats_severity_list[i_detector].append([np.nan,
                                                            np.nan,
                                                            np.nan,
                                                            np.nan,
                                                            np.nan])

            try :
                self.stats_magnitude_list[i_detector].append([np.median(self.magnitude_list[i_detector]),
                                                            kurtosis(self.magnitude_list[i_detector]),
                                                            skew(self.magnitude_list[i_detector]),
                                                            np.percentile(self.magnitude_list[i_detector],10),
                                                            np.percentile(self.magnitude_list[i_detector],90)])
            except:
                #Debug
#                print('Magnitude')
#                print('Detector : '+str(self.list_names_drifts_detectors[i_detector]))
#                print('Stream : '+str(self.name_file))
#                print('Nombre de drifts detected : '+str(len(self.detected_positions[i_detector])))

                self.stats_magnitude_list[i_detector].append([np.nan,
                                                            np.nan,
                                                            np.nan,
                                                            np.nan,
                                                            np.nan])
            try :
                self.stats_interval_list[i_detector].append([np.median(self.interval_list[i_detector]),
                                                            kurtosis(self.interval_list[i_detector]),
                                                            skew(self.interval_list[i_detector]),
                                                            np.percentile(self.interval_list[i_detector],10),
                                                            np.percentile(self.interval_list[i_detector],90)])
            except:
                #Debug
#                print('Interval')
#                print('Detector : '+str(self.list_names_drifts_detectors[i_detector]))
#                print('Stream : '+str(self.name_file))
#                print('Nombre de drifts detected : '+str(len(self.detected_positions[i_detector])))

                self.stats_interval_list[i_detector].append([np.nan,
                                                            np.nan,
                                                            np.nan,
                                                            np.nan,
                                                            np.nan])


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

        #If len(self.detected_positions[i_detector]) != 0 :
#        try :
        self.mean_stats_severity = np.nanmean(np.array(self.stats_severity_list),axis = 1)
        self.std_stats_severity = np.nanstd(np.array(self.stats_severity_list),axis = 1)

        self.mean_stats_magnitude = np.nanmean(np.array(self.stats_magnitude_list),axis = 1)
        self.std_stats_magnitude = np.nanstd(np.array(self.stats_magnitude_list),axis = 1)

        self.mean_stats_interval = np.nanmean(np.array(self.stats_interval_list),axis = 1)
        self.std_stats_interval = np.nanstd(np.array(self.stats_interval_list),axis = 1)
        #else :
#        except IndexError as err:
#            print('\n')
#            print('\n')
#            print(err)
#            for stat in self.stats_severity_list :
#                print('\n')
#                print(len(stat))
#                print(stat)


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
            new_dirpath = dirpath + os.sep + 'ResultsDriftKnowledge'
            os.mkdir(new_dirpath)
        except :
            pass

        with open(new_dirpath+self.name_file+'PerfDetectors.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list_tot)

#        # Create csv file for meta-features measures
##        head = []
#        list_tot = []
#        list_sev_names = ['med_sev','kurto_sev','skew_sev','perc10_sev','perc90_sev']
#        list_mag_names = ['med_mag','kurto_mag','skew_mag','perc10_mag','perc90_mag']
#        list_interv_names = ['med_interv','kurto_interv','skew_interv','perc10_interv','perc90_interv']
#
#
#        for i_detector in range(self.n_drift_detectors):
#            list_tot.append([self.list_names_drifts_detectors[i_detector],'mean','std'])
#
#            for i_stat in range(len(self.mean_stats_severity[i_detector])):
#                list_tot.append([list_sev_names[i_stat],self.mean_stats_severity[i_detector][i_stat],self.std_stats_severity[i_detector][i_stat]])
#            for i_stat in range(len(self.mean_stats_magnitude[i_detector])):
#                list_tot.append([list_mag_names[i_stat],self.mean_stats_magnitude[i_detector][i_stat],self.std_stats_magnitude[i_detector][i_stat]])
#            for i_stat in range(len(self.mean_stats_interval[i_detector])):
#                list_tot.append([list_interv_names[i_stat],self.mean_stats_interval[i_detector][i_stat],self.std_stats_interval[i_detector][i_stat]])
#            list_tot.append(['','',''])
#
##        head = np.concatenate(head)
##        list_tot = list(np.concatenate(list_tot))
##        list_tot = list(map(list, zip(*list_tot)))
##        export_data = zip_longest(*list_tot, fillvalue = '')
##        list_tot.insert(0,head)
#        with open(new_dirpath+self.name_file+'metaFeatDetectors.csv', 'w', newline='') as file:
#            writer = csv.writer(file)
##            writer.writerow(head)
#            writer.writerows(list_tot)


    # General functions
    def severity_measure(self, detecId):
        """
         Measure severity of the drift. The severity is defined as the number of instances in the warning interval.

        """
        self.severity_list[detecId].append(self.n_instances_warning[detecId])
        self.n_instances_warning[detecId] = 0


    def interval_measure(self, detecId):
        """
         Measure the interval between 2 drifts.

        """
        if self.n_detected_drifts[detecId] == 0 :
            self.interval_in_progress[detecId] = True
        else :
            self.interval_list[detecId].append(self.interval[detecId])
            self.interval[detecId] = 0

    def magnitude_measure(self,detecId, p, q):
        """ Extraction of the magnitude of a single drift

            Attributes :
            detecId : ID of the drift detector (int)
            p : distribution before drift (array)
            q : distribution after drift (array)

        """

#        P = p/p.sum()
#        Q = q/q.sum()

        # If drift detected before enought instances passed (window size + safety window), it is raising a Value error/ same if detected too clos eto the end
        # We decide to ignore this case as it is very marginal
        try :
            # Magnitude = hellinger distance
            self.magnitude_list[detecId].append(np.sqrt(sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
        except ValueError as e:
            pass
