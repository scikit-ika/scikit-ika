import numpy as np

from skmultiflow.drift_detection.page_hinkley import PageHinkley
from skika.data.bernoulli_stream import BernoulliStream
from skika.hyper_parameter_tuning.drift_detectors.evaluate_drift_detection_knowledge import EvaluateDriftDetection

def test_evaluate_drift_detection_knowledge():

    expected_n_detec = [11]
    expected_n_TP = [4]
    expected_n_FP = [7]

    list_detect = [[PageHinkley(min_instances=15, delta=0.005, threshold=1.5, alpha=0.999), PageHinkley(min_instances=15, delta=0.005, threshold=2.5, alpha=0.999)]]

    names_detect = ['PH1']

    # Knowledge
    list_streams = [[BernoulliStream(drift_period=100, n_drifts=10, widths_drifts=[1], mean_errors=[0.1,0.9]), 'BernouW1Me0109']]

    eval = EvaluateDriftDetection(list_drifts_detectors=list_detect, list_names_drifts_detectors=names_detect, stream=list_streams[0][0], n_runs=1, name_file=list_streams[0][1])
    n_detec, n_TP, n_FP, delay = eval.run()

    assert np.alltrue(n_detec == expected_n_detec)
    assert np.alltrue(n_TP == expected_n_TP)
    assert np.alltrue(n_FP == expected_n_FP)
