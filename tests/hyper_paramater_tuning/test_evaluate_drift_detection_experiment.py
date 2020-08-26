import os
import numpy as np
import itertools
import csv

from skmultiflow.drift_detection.page_hinkley import PageHinkley
from skika.data.bernoulli_stream import BernoulliStream
from skika.hyper_parameter_tuning.drift_detectors.evaluate_drift_detection_experiment import EvaluateDriftDetection

def test_evaluate_drift_detection_experiment():

    expected_n_detec = [16767.0, 7379.0, 6059.0]
    expected_n_TP = [300.0, 259.0, 294.0]
    expected_n_FP = [16467.0, 7120.0, 5765.0]

    #######################################################################################################################################
    #### Construction of the meta-Knowledge base ####

    # Meta-features in the knowledge base
    sev = [1,100,500]
    mag = [1.0,0.751,0.725,0.632,0.578,0.447,0.389,0.289,0.227,0.142,0.067,0.033]

    meta_features = list(itertools.product(sev,mag))

    # Load best configurations from Pareto
    with open(os.sep.join(['.','recurrent-data','hyper-param-tuning','bestConfigs.csv'])) as csvDataFile:
        best_configs = [row for row in csv.reader(csvDataFile)]

    # Build dict to link configs names and drift detectors
    list_detect = [[PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.005, threshold=2.5, alpha=0.999)],
                   [PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.005, threshold=2.5, alpha=0.9)],
                   [PageHinkley(min_instances=15, delta=0.005, threshold=0.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.999)],
                   [PageHinkley(min_instances=15, delta=0.005, threshold=0.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.9)],
                   [PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.05, threshold=2.5, alpha=0.999)],
                   [PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.05, threshold=2.5, alpha=0.9)],
                   [PageHinkley(min_instances=15, delta=0.05, threshold=0.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.999)],
                   [PageHinkley(min_instances=15, delta=0.05, threshold=0.5, alpha=0.9), PageHinkley(min_instances=15, delta=0.05, threshold=1.5, alpha=0.9)],
                   [PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.999)],
                   [PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.9)],
                   [PageHinkley(min_instances=30, delta=0.005, threshold=0.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.999)],
                   [PageHinkley(min_instances=30, delta=0.005, threshold=0.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9)],
                   [PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.05, threshold=2.5, alpha=0.999)],
                   [PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.05, threshold=2.5, alpha=0.9)],
                   [PageHinkley(min_instances=30, delta=0.05, threshold=0.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.999)],
                   [PageHinkley(min_instances=30, delta=0.05, threshold=0.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.05, threshold=1.5, alpha=0.9)]]


    names_detect = ['PH1','PH2','PH3','PH4','PH5','PH6','PH7','PH8','PH9','PH10','PH11','PH12','PH13','PH14','PH15','PH16']

    dictionary = dict(zip(names_detect, list_detect))
    #######################################################################################################################################

    # Detectors configurations for evaluation
    drift_detect_eval = [[[PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=30, delta=0.005, threshold=0.5, alpha=0.999)],
                         [PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9)],
                         [PageHinkley(min_instances=30, delta=0.005, threshold=2.5, alpha=0.9), PageHinkley(min_instances=30, delta=0.005, threshold=1.5, alpha=0.9)]]]

    names_drift_detect_eval = [['PH9','PH10','PH10Adapt']]


    # Build metaK bases : Select best configurations depending on what detector is used
    metaK_bases = []
    list_ind_detec = [0] # TODO : MODIFY depending what detector is used -> 0 : PH, 1 : ADWIN, 2: DDM, 3: SeqDrift2
    for ind_detec in list_ind_detec :
        metaK_bases.append([meta_features,list(map(list, zip(*best_configs)))[ind_detec]])

    names_files = ['testAdaptEvalPH']

    win_sizes= [2]

    streams = [[BernoulliStream(drift_period=1500, n_drifts=300, widths_drifts=[1,100,500], mean_errors=[[0.1,0.9],[0.5,0.6],[0.2,0.8],[0.4,0.5],[0.3,0.7]], n_stable_drifts=2),"Bernou2"]]


    eval = EvaluateDriftDetection(list_drifts_detectors=drift_detect_eval[0],
                                  list_names_drifts_detectors= names_drift_detect_eval[0],
                                  adapt=[0, 0, 1],
                                  k_base=metaK_bases[0],
                                  dict_k_base=dictionary,
                                  win_adapt_size=win_sizes[0],
                                  stream=streams[0][0],
                                  n_runs=1,
                                  name_file=streams[0][1]+'_'+names_files[0]+'_WinSize'+str(win_sizes[0]))

    n_detec, n_TP, n_FP, delay = eval.run()

    assert np.alltrue(n_detec == expected_n_detec)
    assert np.alltrue(n_TP == expected_n_TP)
    assert np.alltrue(n_FP == expected_n_FP)
