import numpy as np
import math
from operator import mul
from fractions import Fraction
from functools import reduce
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd


class AutoDDM(BaseDriftDetector):

    """
    Description :
        AutoDDM is a dirft detector that adjusts the drift thresholds based on prior information.
        We exploit the periodicity in the data stream when it exists, such that it is more sensitive to true concept drifts while reducing false-positive detections.

    Parameters :
        min_num_instances: int
            The minimum required number of analyzed samples so change can be
            detected. This is used to avoid false detections during the early
            moments of the detector, when the weight of one sample is important.

        warning_level: float
            Warning Level

        out_control_level: float
            Out-control Level

        default_prob: float (0 to 1)
            The initial probability when drift detected and reset. Default value 1.

        ts_length: int
            The length of location buffer. Default value 20.

        confidence: float (0 to 1)
            The default confidence level. Default value 0.95.

        tolerance: int
            The tolerance range of matching. Default value 1000. E.g. 500 plus/minus tolerance will match to 500

        c: float
            A Laplacian constant used in the threshold function. Default value 0.05.


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

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0,
                 default_prob=1, ts_length=20, confidence=0.95, tolerance = 1000, c = 0.05):
        """
        :param min_num_instances:
        :param warning_level:
        :param out_control_level:
        :param default_prob:
        :param ts_length:
        :param confidence: The default confidence level.
        :param tolerance: The tolerance range of matching.
        :param c: A Laplacian constant.
        """
        super().__init__()
        self.sample_count = 1
        self.global_ratio = 1.0
        self.pr = 0
        self.std = 0
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.global_prob = default_prob
        self.local_prob = default_prob
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.default_prob = default_prob
        self.drift_ts = []
        self.reset()
        self.diff = -1
        self.ts_prediction = -1
        self.TP_detected = False
        self.FP_detected = False
        self.ts_length = ts_length
        self.period = 1
        self.confidence = confidence
        self.tolerance = tolerance
        self.c = c

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.local_prob = self.default_prob
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.global_ratio = 0
        self.pr = 1
        self.std = 0


    def add_element(self, prediction, n):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        n: int
            This parameter indicates the current timestamp t.

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """

        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        if (len(self.drift_ts) >= self.ts_length):
            self.global_prob = self.calculate_pr(n - self.drift_ts[0], len(self.drift_ts))
        else:
            self.global_prob = 1
        self.local_prob = self.calculate_pr(self.sample_count, 1)
        self.pr = self.threshold_function(self.global_ratio * self.global_prob + (1 - self.global_ratio) * self.local_prob)
        self.std = np.sqrt(self.pr * (1 - self.pr) / float(self.sample_count))

        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.diff > 0:
            if self.sample_count <= self.diff:
                self.global_ratio = self.sample_count / self.diff
            else:
                # Drift missed
                self.global_ratio = 1 - (self.sample_count - self.diff) / self.sample_count
        else:
            self.global_ratio = 0


        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std


        if (self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min):
            self.in_concept_change = True


        if ((self.miss_prob + self.miss_std > self.pr + self.out_control_level * self.std) and (self.global_ratio > self.confidence)):
            # print("Drift detected by AutoDDM at " + str(n))
            self.in_concept_change = True


        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False


    def nCk(self, n, k):
        return int(reduce(mul, (Fraction(n - i, i + 1) for i in range(k)), 1))


    def calculate_pr(self, ove, spe, n=1, x=1):
        if ove == 1:
            return self.default_prob
        if spe == 0:
            return self.default_prob
        else:
            return self.nCk(spe, x) * self.nCk(ove - spe, n - x) / self.nCk(ove, n)

    def threshold_function(self, pr):
        return math.sqrt(pr) / (self.c * self.global_ratio + math.sqrt(pr))

    def get_pr(self):
        return self.pr

    def get_std(self):
        return self.std

    def get_global_ratio(self):
        return self.global_ratio

    def get_drift_ts(self):
        return self.drift_ts

    def get_min_pi(self):
        return self.miss_prob_min

    def get_min_si(self):
        return self.miss_sd_min

    def get_pi(self):
        return self.miss_prob

    def detect_TP(self, n):
        """ A true concept drift is detected
        :param n: The timestamp when the true concept drift is detected
        """
        self.drift_ts.append(n)

        if (len(self.drift_ts) >= self.ts_length):
            # Majority Vote approach
            ts = np.diff(self.drift_ts)
            periods = []
            for i in range(0, math.floor(self.ts_length / 2)):
                current = i
                for j in range(current + 2, min(self.ts_length, math.floor(current + self.ts_length / 2))):
                    if j + 1 < self.ts_length - 1:
                        if abs(ts[j] - ts[current]) < self.tolerance:
                            if abs(ts[j + 1] - ts[current + 1]) < self.tolerance:
                                periods.append(j - current)
                                current = j

            if len(periods) == 0:
                self.period = 1
            else:
                self.period = math.floor(max(set(periods), key=periods.count))

        if (len(self.drift_ts) > self.ts_length):
            ts = np.diff(self.drift_ts)

            if (self.period <= 1):
                self.diff = -1
            else:
                # TRUE POSITIVE detected
                learn = ts[-(2 * self.period):]
                # Averaging the effect of variations.
                for i in range(self.period):
                    average = (learn[i] + learn[i + self.period])/2
                    learn[i] = average
                    learn[i + self.period] = average
                self.confidence = (max(learn) - self.tolerance) / max(learn)
                temp_df = pd.DataFrame(
                    {'timestamp': pd.date_range('2000-01-01', periods=len(learn), freq=None),
                     'ts': learn})
                temp_df.set_index('timestamp', inplace=True)
                hw_model = ExponentialSmoothing(temp_df, trend=None, seasonal='add', seasonal_periods=self.period).fit(
                    optimized=True)
                predict_result = np.array(hw_model.predict(start=len(learn), end=len(learn)))[0]
                self.diff = predict_result

            self.ts_prediction = n + self.diff
            self.drift_ts.pop(0)
        return


    def detect_FP(self, n):
        """ A  false positive is detected
        :param n: The timestamp when the false positive is detected
        """
        self.diff = self.ts_prediction - n
        return

    def get_ts_object(self):
        return self.drift_ts
