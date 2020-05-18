# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:31:06 2020

@author: tlac980
"""

from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.drift_detection import ADWIN
from skmultiflow.data import FileStream
import numpy as np

from skika.hyper_parameter_tuning.evaluate_prequential_and_adapt import EvaluatePrequentialAndAdaptTreesARF

def test_evaluate_and_adapt_trees():
    
    expected_accuracies = [0.86, 0.876, 0.914, 0.858, 0.77, 0.894, 0.876, 0.91, 0.898, 0.884, 0.804, 0.808]
    
    expected_trees = [30,60,30,60,30]
    
    # Load the meta-model
    dictMeta = {0.0:60 ,0.1:30, 0.2:30, 0.3:30, 0.4:60, 0.5:70, 0.6:60, 0.7:30, 0.8:30, 0.9:30} # dict = {'pourc redund feat':best nb tree}
    
    n_trees = 10
    n_samples_max = 6000
    n_samples_meas = 500
    
    stream = FileStream('./recurrent-data/real-world/elec.csv')
    
    stream.prepare_for_use()
    
    # Evaluate model (with adaptation or not)
    arf = AdaptiveRandomForest(n_estimators = n_trees, lambda_value=6, grace_period=10, split_confidence=0.1, tie_threshold=0.005, 
                               warning_detection_method= ADWIN(delta=0.01), drift_detection_method=ADWIN(delta=0.001), random_state = 0)
    
    modelsList = [arf]
    modelsNames = ['ARF']
    
    evaluator = EvaluatePrequentialAndAdaptTreesARF(metrics=['accuracy','kappa','running_time','ram_hours'],
                                    show_plot=False,
                                    n_wait=n_samples_meas,
                                    pretrain_size=200,
                                    max_samples=n_samples_max,
                                    output_file = None,
                                    metaKB=dictMeta)
    
    # Run evaluation
    model, acc, n_trees = evaluator.evaluate(stream=stream, model=modelsList, model_names=modelsNames)
    
    assert np.alltrue(acc[0] == expected_accuracies)
    
    assert np.alltrue(n_trees[0] == expected_trees)