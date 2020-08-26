# Imports
import numpy as np
#from skmultiflow.evaluation.base_evaluator import StreamEvaluator
#from skmultiflow import data
#from skmultiflow.metrics import WindowClassificationMeasurements, ClassificationMeasurements, \
#    MultiTargetClassificationMeasurements, WindowMultiTargetClassificationMeasurements, RegressionMeasurements, \
#    WindowRegressionMeasurements, MultiTargetRegressionMeasurements, \
#    WindowMultiTargetRegressionMeasurements, RunningTimeMeasurements


class ComputeStreamMetaFeatures():
    """
    Description :
        Compute extraction of several meta-features on the stream.

    """

    # TODO : Create a file to list features to be extracted (like constants in scikit.multiflow)

    def __init__(self,
                 stream = None,
                 list_feat = None):

        if stream != None :
            self.stream = stream
        else :
            raise ValueError("A stream must be specified")

        # TODO : test here if meat-feat in list are correct
        if ((list_feat != None) and (isinstance(list_feat, list))):
            self.list_feat = list_feat
        else :
            raise ValueError("A list of features must be specified. Attibute 'list_feat' must be 'list'")

        # List of stream samples for redundancy calculation
        self.list_stream_samples = []

        # List of stream prediction and true labels for severity calculation
        self.list_predicted_y = []
        self.list_true_y = []

        # List of stream instances for magnitude calculation
        self.list_instances_after_drift = []
        self.list_instances_before_drift = []

        # List of predictions for magnitude calculation
        self.list_predictions_after_drift = []
        self.list_predictions_before_drift = []


    def run_extraction(self, list_extrac):
        list_model_meta_feats = []
        list_drift_meta_feats = []
        if 'perc_redund_feat' in self.list_feat and 'perc_redund_feat' in list_extrac :
            list_model_meta_feats.append(self.extractPercRedundFeatMeasured())

        if 'drift_severity' in self.list_feat and 'drift_severity' in list_extrac :
            list_drift_meta_feats.append(self.extractSeverityOneDrift())

        if 'drift_magnitude_inst' in self.list_feat and 'drift_magnitude_inst' in list_extrac :
            w1 = np.array(self.list_instances_before_drift)
            w2 = np.array(self.list_instances_after_drift)
            list_drift_meta_feats.append(self.extractMagnitudeOneDrift(w1,w2))

        if 'drift_magnitude_att' in self.list_feat and 'drift_magnitude_att' in list_extrac :
            w1 = np.array(self.list_instances_before_drift).T
            w2 = np.array(self.list_instances_after_drift).T
            list_drift_meta_feats.append(self.extractMagnitudeOneDrift(w1,w2))

        if 'drift_magnitude_pred' in self.list_feat and 'drift_magnitude_pred' in list_extrac :
            w1 = np.array(self.list_predictions_after_drift)
            w2 = np.array(self.list_predictions_before_drift)
            list_drift_meta_feats.append(self.extractMagnitudeOneDrift(w1,w2))

        return list_model_meta_feats, list_drift_meta_feats

    ################
    # Meta-features
    ################

    ## Model related

    # Extract percentage of redundant features by simple reading of stream attribute
    def extractPercRedundFeat(self) :
        """ Extraction of the percentage of redundant features, directly read from the stream parameters

        """
        return self.stream.perc_redund_features

    # Extract percentage of redundant features by measuring correlation between features
    def extractPercRedundFeatMeasured(self):
        """ Extraction of the percentage of redundant features, measured on the stream by correlation between features

        """

        array_samples = np.vstack(self.list_stream_samples)

        corr_matrix = np.corrcoef(array_samples,rowvar=False)

        n_feat_redund = 0
        for i in range(self.stream.n_features) :
            for j in range(i-1):
                if corr_matrix[i][j] > 0.8 :
                    n_feat_redund +=1
                    break

        perc_redund_measured = n_feat_redund/self.stream.n_features

#        print('Perc redund measured : {}'.format(perc_redund_measured))
        return perc_redund_measured


    ## Drift related

    def extractSeverityOneDrift(self):
        """ Extraction of the severity of a single drift

        """
        n_instances_in_warning = len(self.list_predicted_y)
        n_missclass_in_warning = 0

        for i in range(n_instances_in_warning) :
            if self.list_predicted_y[i] != self.list_true_y[i] :
                n_missclass_in_warning += 1

        severity = n_missclass_in_warning/n_instances_in_warning

        print('Severity measured : {}'.format(severity))
        return severity

#    def hellinger(p, q):
#        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

    def extractMagnitudeOneDrift(self,p,q):
        """ Extraction of the magnitude of a single drift

            Attributes :
            p : distribution before drift (array)
            q : distribution after drift (array)

        """

        P = p/p.sum()
        Q = q/q.sum()

        # Magnitude = hellinger distance
        magnitude = np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2)) / np.sqrt(2)

        print('Magnitude measured : {}'.format(magnitude))
        return magnitude
