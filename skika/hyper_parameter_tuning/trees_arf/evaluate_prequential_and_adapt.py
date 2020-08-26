import os

import warnings
import re
from timeit import default_timer as timer

from numpy import unique

# Include the 2 following from the third_party skmultiflow
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants

from skika.hyper_parameter_tuning.trees_arf.meta_feature_generator import ComputeStreamMetaFeatures


class EvaluatePrequentialAndAdaptTreesARF(StreamEvaluator):
    """ Prequential evaluation method with adaptive tuning of hyper-parameters to tune the number of trees in ARF.

    Description :
        This code is based on the ``scikit_multiflow`` evaluate_prequential implementation.
        Copyright (c) 2017, scikit-multiflow
        All rights reserved.

        We modified it to include adaptive tuning of hyper-parameters.

        Scikit_multiflow description:
        An alternative to the traditional holdout evaluation, inherited from
        batch setting problems.

        The prequential evaluation is designed specifically for stream settings,
        in the sense that each sample serves two purposes, and that samples are
        analysed sequentially, in order of arrival, and become immediately
        inaccessible.

        This method consists of using each sample to test the model, which means
        to make a predictions, and then the same sample is used to train the model
        (partial fit). This way the model is always tested on samples that it
        hasn't seen yet.

        Additional scikit-ika features:
        This method implements an adaptive tuning process to adapt the number of trees
        in an Adaptive Random Forest, depending on the number of redundant features in the stream.

    Parameters :
        n_wait:int (Default: 200)
            The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.
            Note that setting `n_wait` too small can significantly slow the evaluation process.

        max_samples:int (Default: 100000)
            The maximum number of samples to process during the evaluation.

        batch_size:int (Default: 1)
            The number of samples to pass at a time to the model(s).

        pretrain_size:int (Default: 200)
            The number of samples to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

        max_time:float (Default: float("inf"))
            The maximum duration of the simulation (in seconds).

        metrics:list, optional (Default: ['accuracy', 'kappa'])

            | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
              and/or logged into the output file. Valid options are
            | *Classification*
            | 'accuracy'
            | 'kappa'
            | 'kappa_t'
            | 'kappa_m'
            | 'true_vs_predicted'
            | 'precision'
            | 'recall'
            | 'f1'
            | 'gmean'
            | *Multi-target Classification*
            | 'hamming_score'
            | 'hamming_loss'
            | 'exact_match'
            | 'j_index'
            | *Regression*
            | 'mean_square_error'
            | 'mean_absolute_error'
            | 'true_vs_predicted'
            | *Multi-target Regression*
            | 'average_mean_squared_error'
            | 'average_mean_absolute_error'
            | 'average_root_mean_square_error'
            | *Experimental*
            | 'running_time'
            | 'model_size'
            | 'ram_hours'

        output_file: string, optional (Default: None)
            File name to save the summary of the evaluation.

        show_plot: bool (Default: False)
            If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
            process.

        restart_stream: bool, optional (default: True)
            If True, the stream is restarted once the evaluation is complete.

        data_points_for_classification: bool(Default: False)
            If True, the visualization used is a cloud of data points (only works for classification) and default
            performance metrics are ignored. If specific metrics are required, then they *must* be explicitly set
            using the ``metrics`` attribute.

        metaKB : dict (Default: None)
            The meta model linking the meta features to the hyper-parameters configuration.
            It is a dictionary linking the percentage of redundant features and the number of trees to choose for
            each of them. This model is built by runing multiple ARF configurations (with different number of trees)
            on multiple streams with different percentages of redundant features, and using the build_pareto_knowledge_trees
            module to choose the number of trees.
            E.g.: dictMeta = {0.0:60 ,0.1:30, 0.2:30, 0.3:30, 0.4:60, 0.5:70, 0.6:60, 0.7:30, 0.8:30, 0.9:30}
            If no metaKB, the class performs only the prequential evaluation.

    Notes
        1. If the adaptive hyper-parameter tuning is not used, this evaluator can process a single learner to track its performance;
           or multiple learners  at a time, to compare different models on the same stream.

        2. If the adaptive hyper-parameter tuning is used, this evaluator can process only a single learner at the moment.

        3. This class can be only used with the ARF as a classifier. Further developments are needed to generalise it to more tasks with
           more classifiers.

        4. The metric 'true_vs_predicted' is intended to be informative only. It corresponds to evaluations at a specific
           moment which might not represent the actual learner performance across all instances.

    Example:

        >>> from skika.data.random_rbf_generator_redund import RandomRBFGeneratorRedund
        >>> from skika.hyper_parameter_tuning.trees_arf.evaluate_prequential_and_adapt import EvaluatePrequentialAndAdaptTreesARF
        >>>
        >>> # Set the stream
        >>> stream = StreamGeneratorRedund(base_stream = RandomRBFGeneratorRedund(n_classes=2, n_features=30, n_centroids=50, noise_percentage = 0.0), random_state=None, n_drifts = 100, n_instances = 100000)
        >>> stream.prepare_for_use()
        >>>
        >>> # Set the model
        >>> arf = AdaptiveRandomForest(n_estimators = 10)
        >>>
        >>> # Set the meta knowledge
        >>> dictMeta = {0.0:60 ,0.1:30, 0.2:30, 0.3:30, 0.4:60, 0.5:70, 0.6:60, 0.7:30, 0.8:30, 0.9:30} # dict = {'pourc redund feat':best nb tree}
        >>>
        >>> # Set the evaluator
        >>>
        >>> evaluator = EvaluatePrequential(metrics=['accuracy','kappa','running_time','ram_hours'],
        >>>                                 max_samples=100000,
        >>>                                 n_wait=500,
        >>>                                 pretrain_size=200,
        >>>                                 show_plot=True)
        >>>
        >>> # Run evaluation with adative tuning
        >>> evaluator.evaluate(stream=stream, model=arf, model_names=['ARF'])

    """

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False,
                 metaKB=None):

        super().__init__()
        self._method = 'prequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.metaKB = metaKB

        if not self.data_points_for_classification:
            if metrics is None:
                self.metrics = [constants.ACCURACY, constants.KAPPA]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        else:
            if metrics is None:
                self.metrics = [constants.DATA_POINTS]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                    self.metrics.append(constants.DATA_POINTS)
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a model on samples from a stream and adapt the tuning.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.
            NOTE : Only ARF is usable with this current version of the adaptive tuning.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model, self.list_acc, new_nb_trees = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold()

            return self.model, self.list_acc, new_nb_trees

    def _train_and_test(self):
        """ Method to control the prequential evaluation and adaptive tuning.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True

        # Init meta_features extractors and list of accuracies (for test) for each model
        self.list_acc = []
        self.extractor = []
        for  i in range(self.n_models):
            self.extractor.append(ComputeStreamMetaFeatures(stream = self.stream, list_feat = ['perc_redund_feat']))
            self.list_acc.append([])

        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            ######################
            # Do model adaptation  only if a knowledge base is not None
            ######################

            if self.metaKB != None :
                ####################
                # Choose first configuration of parameters from pre_train set meta feats

                self.extractor[i].list_stream_samples.append(X)

                current_perc_redund = self.extractor[i].run_extraction(['perc_redund_feat'])[0][0]

                # ## DEBUG:
                # print('Read perc redund : '+str(self.stream.perc_redund_features))

#                print('Initial Configuration of the hyperparameters : nbtrees = {}'.format(self.metaKB[round(current_perc_redund[0],1)]))
                for i in range(self.n_models):
                    self.model[i].n_estimators = self.metaKB[round(current_perc_redund,1)] # Use round(current_perc_redund[0],1) as perc redund is the first feature in the extracted list
                    self.model[i].init_ensemble(X)
                    print('initial number of trees {}'.format(self.metaKB[round(current_perc_redund,1)]))
                last_perc_redund = current_perc_redund

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print('Evaluating...')

        # Verification variables
        compt_drift = []
        drift_detec_list = []
        pourc_redund_read = []
        new_nb_trees = []

        # Initialise RAM_hours measurements
        for i in range(self.n_models):
            self.running_RAM_H_measurements[i].compute_evaluate_start_time()

            # Verification variables
            compt_drift.append(0)          # Temp variable to simulate drift detection
            drift_detec_list.append([])
            pourc_redund_read.append([])
            new_nb_trees.append([])

        sample_stream = False

        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:

                X, y = self.stream.next_sample(self.batch_size)

                ######################
                # Do model adaptation  only if a knowledge base is not None
                ######################

                if self.metaKB != None :

                    # Update model hyper-parameters if drifts
                    # TODO : To begin we specify when the drifts happens (set in ConceptDriftStream at 5000) -->  replace with direct drift detection BaseDriftDetector
                    # if drift detected (signal from ARF) -> extract meta-features
                    # if change in meta features -> match with meta-knowledge and change parameters

                    # Drift Detection
                    # Works only with adaptiveRF modif to get drift and warnings for a StreamGeneratorRedund

                    for i in range(self.n_models):

                        # if warning detected, measure of meta-features is launched
                        if self.model[i].warning_detected :
                            print('Warning detected at {}'.format(self.global_sample_count))
                            self.extractor[i].list_stream_samples = []
                            sample_stream = True

                        if sample_stream == True :
                            # Store next samples
                            self.extractor[i].list_stream_samples.append(X)

                        if self.model[i].drift_detected and len(self.extractor[i].list_stream_samples) > 10:
                            print('Drift detected at {}'.format(self.global_sample_count))
                            current_perc_redund = self.extractor[i].run_extraction(['perc_redund_feat'])[0][0]

                            # ## DEBUG:
                            # print('Read perc redund : '+str(self.stream.perc_redund_features))

                            # Test first if meta-features really changed before updating the model
                            if last_perc_redund != current_perc_redund :
                                print('Change in meta-features at {}'.format(self.global_sample_count))
                                self.model[i].new_n_estimators = self.metaKB[round(current_perc_redund,1)] # Round for the moment to get perfect match with meta model
                                self.model[i].update_config(X)
                                print('New number of trees {}'.format(self.metaKB[round(current_perc_redund,1)]))

                                last_perc_redund = current_perc_redund
                                sample_stream = False

                                # Verification variables
                                compt_drift[i] = compt_drift[i] + 1
                                drift_detec_list[i].append(self.global_sample_count)
                                pourc_redund_read[i].append(round(current_perc_redund,1))
                                new_nb_trees[i].append(self.metaKB[round(current_perc_redund,1)])

                                # # DEBUG:
#                                print('Number detected drifts : '+str(compt_drift[i]))
#                                print('Positions detected drifts : '+str(drift_detec_list[i]))
#                                print('Perc redund measured : '+str(pourc_redund_read[i]))

                if X is not None and y is not None:
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                            # Testing time
                            self.running_RAM_H_measurements[i].compute_evaluate_start_time()
                            self.running_time_measurements[i].compute_testing_time_begin()
                            prediction[i].extend(self.model[i].predict(X))
                            self.running_time_measurements[i].compute_testing_time_end()
                            self.running_RAM_H_measurements[i].compute_update_time_increment()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                               self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_RAM_H_measurements[i].compute_evaluate_start_time()
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                                self.running_RAM_H_measurements[i].compute_update_time_increment()
                            else:
                                self.running_RAM_H_measurements[i].compute_evaluate_start_time()
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()
                                self.running_RAM_H_measurements[i].compute_update_time_increment()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_RAM_H_measurements[i].compute_evaluate_start_time()
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                            self.running_RAM_H_measurements[i].compute_update_time_increment()

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= self.max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()

                            for i in range(self.n_models):
                                self.list_acc[i].append(self.current_eval_measurements[i].accuracy_score())

                        update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.metaKB != None :
            for i in range(self.n_models):
                print('Number detected drifts : '+str(compt_drift[i]))
                print('Positions detected drifts : '+str(drift_detec_list[i]))
                print('Perc redund measured : '+str(pourc_redund_read[i]))
                print('New trees number : '+str(new_nb_trees[i]))

        if self.restart_stream:
            self.stream.restart()

        return self.model, self.list_acc, new_nb_trees

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification labels / target values for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task. Not used for regressors.

        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info
