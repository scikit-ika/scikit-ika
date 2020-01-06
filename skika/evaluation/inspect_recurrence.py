
from copy import deepcopy
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class InspectorVisualizer():
    def __init__(self, name="test.pdf"):
        self.name = name
        self.accuracy_sum = 0
        self.accuracy_mean = 0
        self.recent_seen = deque()
        self.recent_sum = 0
        self.accuracy_recent = 0
        self.observed_samples = 0
        self.sample_ids = []
        self.accuracy_plot = []

        self.__configure()

        self.state_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', 
                             '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    def hold(self):
        plt.savefig(self.name)
        # plt.show(block=True)

    def __configure(self):
        # plt.ion()
        self.fig = plt.figure(figsize=(18, 10))
        self.sub_plot_obj = self.fig.add_subplot(11 + 2 * 100)
        self.sub_plot_current = self.fig.add_subplot(12 + 2 * 100)
        self.sub_plot_obj.set_autoscale_on(True)
        self.sub_plot_obj.autoscale_view(True, True, True)
        self.sub_plot_current.set_autoscale_on(True)
        self.sub_plot_current.autoscale_view(True, True, True)

        self.lc = LineCollection([])
        self.lc_c = LineCollection([])
        self.lc_s = LineCollection([])
        self.lc_s_segs = [[[0, 0.25], [0, 0.25]]]
        self.sub_plot_obj.add_collection(self.lc)
        self.sub_plot_current.add_collection(self.lc_c)
        self.sub_plot_current.add_collection(self.lc_s)
        self.colors = []
        self.segments = [None]
        self.on_screen_segs = 0
        self.alt_state_lcs = None

    def on_new_train_step(self, sample_id, data_exposure, model_exposure, X, y, p):
        # plt.pause(1e-9)
        if data_exposure['drift']:
            self.sub_plot_obj.axvline(x=sample_id, color="red")
            self.sub_plot_current.axvline(x=sample_id, color="red")
        if model_exposure['found_change']:
            self.sub_plot_obj.axvline(x=sample_id, color="green")
            self.sub_plot_current.axvline(x=sample_id, color="green")
        print('**********************')
        print(f"{model_exposure}")
        print('**********************')
        if model_exposure['current_sensitivity']:
            sx = sample_id
            sy = ((model_exposure['current_sensitivity'] - 0.05) / (0.05 * 1.5))* 0.5 + 0.25
            print(f"X: {sx}, Y: {sy}")
            seg = [sx, sy]
            
            self.lc_s_segs.append([self.lc_s_segs[-1][1], seg])
            self.lc_s.set_segments(self.lc_s_segs)
            # self.sub_plot_current.plot(sx, sy)
            # self.sub_plot_current.axhline(y=sy, xmin = 0.98, xmax = 1, color = "green")

        color = self.state_colors[model_exposure['active_state']]
        self.sample_ids.append(sample_id)
        is_correct = y == p
        self.accuracy_sum += is_correct
        self.recent_sum += is_correct
        self.recent_seen.append(is_correct)
        if len(self.recent_seen) > 300:
            self.recent_sum -= self.recent_seen.popleft()
        self.observed_samples += 1
        self.accuracy_mean = self.accuracy_sum / self.observed_samples
        self.accuracy_recent = self.recent_sum / len(self.recent_seen)

        if self.segments[-1] is None:
            self.segments[-1] = [[sample_id, self.accuracy_recent[0]],
                                 [sample_id, self.accuracy_recent[0]]]
            self.colors.append(color)

        if abs(self.segments[-1][1][1] - self.accuracy_recent[0]) < 0.01 and self.colors[-1] == color:
            self.segments[-1][1][0] = sample_id
        else:
            self.segments.append(
                [self.segments[-1][1], [sample_id, self.accuracy_recent[0]]])
            self.colors.append(color)

        screen_min = self.sample_ids[-1] - 1000
        while self.segments[self.on_screen_segs][0][0] < screen_min and self.segments[self.on_screen_segs][1][0] < screen_min:
            self.on_screen_segs += 1

        self.accuracy_plot.append(self.accuracy_recent[0])

        if model_exposure['reset_alternative_states'] or (self.alt_state_lcs is None and len(model_exposure['alternative_states']) > 0):
            self.sub_plot_obj.axvline(x=sample_id, color="blue", ymin=0.5)
            if model_exposure['signal_confidence_backtrack']:
                self.sub_plot_obj.axvline(x=sample_id, color="yellow", ymin=0.5)
            if model_exposure['signal_confidence_backtrack']:
                self.sub_plot_obj.axvline(x=sample_id, color="red", ymin=0.5)
            print(model_exposure)
            # input()
            self.alt_state_lcs = []
            self.alt_state_segs = []
            self.alt_state_stats = []
            for alt_state in model_exposure['alternative_states']:
                as_color = self.state_colors[alt_state[0]
                                             ] if alt_state[0] >= 0 else "black"
                self.alt_state_lcs.append(LineCollection(
                    [], color=as_color, linestyle='dotted', linewidth=1))
                self.alt_state_segs.append([[[sample_id, 0], [sample_id, 0]]])
                self.alt_state_stats.append([deque(), 0])
                self.sub_plot_obj.add_collection(self.alt_state_lcs[-1])

            self.alt_state_conf_lcs = []
            self.alt_state_conf_segs = []
            self.alt_state_conf_stats = []
            for alt_state in model_exposure['alternative_states']:
                as_color = self.state_colors[alt_state[0]
                                             ] if alt_state[0] >= 0 else "black"
                self.alt_state_conf_lcs.append(LineCollection(
                    [], color=as_color, linestyle='dotted', linewidth=1))
                self.alt_state_conf_segs.append([[[sample_id, 0], [sample_id, 0]]])
                self.alt_state_conf_stats.append([deque(), 0])
                self.sub_plot_obj.add_collection(self.alt_state_conf_lcs[-1])

        for ai, alt_state in enumerate(model_exposure['alternative_states']):
            self.alt_state_stats[ai][0].append(alt_state[1])
            self.alt_state_stats[ai][1] += alt_state[1]
            if len(self.alt_state_stats[ai][0]) > 300:
                self.alt_state_stats[ai][1] -= self.alt_state_stats[ai][0].popleft()
            if abs(self.alt_state_segs[ai][-1][1][1] - self.alt_state_stats[ai][1] / len(self.alt_state_stats[ai][0])) < 0.01:
                self.alt_state_segs[ai][-1][1][0] = sample_id
            else:
                self.alt_state_segs[ai].append([self.alt_state_segs[ai][-1][1], [
                                               sample_id, self.alt_state_stats[ai][1] / len(self.alt_state_stats[ai][0])]])
            self.alt_state_lcs[ai].set_segments(self.alt_state_segs[ai])
            
            print("model exposure: ")
            print(model_exposure['alternative_states_difference_confidence'])
            if alt_state[0] not in model_exposure['alternative_states_difference_confidence']:
                continue
            print(f"plotting {alt_state[0]}")
            alt_stat_current_conf = model_exposure['alternative_states_difference_confidence'][alt_state[0]]
            self.alt_state_conf_stats[ai][0].append(alt_stat_current_conf)
            self.alt_state_conf_stats[ai][1] = (alt_stat_current_conf * 0.5) + (0.1 - 0.025)
            if self.alt_state_conf_segs[ai][-1][0][1] == 0:
                self.alt_state_conf_segs[ai][-1][0][1] = self.alt_state_conf_stats[ai][1]
            if len(self.alt_state_conf_stats[ai][0]) > 300:
                self.alt_state_conf_stats[ai][0].popleft()
            print(f"Current end: {self.alt_state_conf_segs[ai][-1][1][1]}, new value: {self.alt_state_conf_stats[ai][1]}")
            if abs(self.alt_state_conf_segs[ai][-1][1][1] - self.alt_state_conf_stats[ai][1]) < 0.01:
                print("continue Seg")
                self.alt_state_conf_segs[ai][-1][1][0] = sample_id
            else:
                print("New Seg")
                self.alt_state_conf_segs[ai].append([self.alt_state_conf_segs[ai][-1][1], [
                                               sample_id, self.alt_state_conf_stats[ai][1]]])
            self.alt_state_conf_lcs[ai].set_segments(self.alt_state_conf_segs[ai])

        if model_exposure['load_restore_state'] is not None:
            restore_id = model_exposure['load_restore_state'][0]
            new_id = model_exposure['load_restore_state'][1]
            restore_point = data_exposure['restore_states'][restore_id][2]
            self.sub_plot_obj.plot([restore_point, sample_id], [
                                   0.1, 0.1], color=self.state_colors[new_id])

        self.lc.set_segments(self.segments)
        self.lc.set_color(self.colors)
        self.lc_c.set_segments(self.segments[self.on_screen_segs:])
        self.lc_c.set_color(self.colors[self.on_screen_segs:])
        self.sub_plot_obj.set_ylim((0, 1.1))
        self.sub_plot_obj.set_xlim(0, self.sample_ids[-1])
        self.sub_plot_obj.relim()
        self.sub_plot_obj.autoscale_view(True, True, True)
        self.sub_plot_current.set_ylim((0, 1.1))
        self.sub_plot_current.set_xlim(screen_min, self.sample_ids[-1] + 10)
        self.sub_plot_current.relim()
        self.sub_plot_current.autoscale_view(True, True, True)


class DataExposure:
    def __init__(self, data):
        self.drift_points = {}
        if 'drift_points' in data:
            self.drift_points = data['drift_points']

    def get_exposed_info_at_sample(self, sample_index, restore_states):
        sample_exposed_info = {}
        sample_exposed_info['drift'] = False
        if sample_index in self.drift_points:
            sample_exposed_info['drift'] = True
        sample_exposed_info['restore_states'] = restore_states

        return sample_exposed_info


class ModelExposure:
    def __init__(self, model):
        self.model = model

    def get_exposed_info_at_sample(self, sample_index):
        sample_exposed_info = {}
        if hasattr(self.model, 'found_change'):
            sample_exposed_info['found_change'] = self.model.found_change
        if hasattr(self.model, 'active_state'):
            sample_exposed_info['active_state'] = self.model.active_state
        if hasattr(self.model, 'reset_alternative_states'):
            sample_exposed_info['reset_alternative_states'] = self.model.reset_alternative_states
        if hasattr(self.model, 'alternative_states'):
            sample_exposed_info['alternative_states'] = self.model.alternative_states
        if hasattr(self.model, 'states'):
            sample_exposed_info['states'] = self.model.states
        if hasattr(self.model, 'set_restore_state'):
            sample_exposed_info['set_restore_state'] = self.model.set_restore_state
        if hasattr(self.model, 'load_restore_state'):
            sample_exposed_info['load_restore_state'] = self.model.load_restore_state
        if hasattr(self.model, 'alternative_states_difference_confidence'):
            sample_exposed_info['alternative_states_difference_confidence'] = self.model.alternative_states_difference_confidence
        if hasattr(self.model, 'signal_confidence_backtrack'):
            sample_exposed_info['signal_confidence_backtrack'] = self.model.signal_confidence_backtrack
        if hasattr(self.model, 'signal_difference_backtrack'):
            sample_exposed_info['signal_difference_backtrack'] = self.model.signal_difference_backtrack
        if hasattr(self.model, 'current_sensitivity'):
            sample_exposed_info['current_sensitivity'] = self.model.current_sensitivity

        return sample_exposed_info


class InspectPrequential:
    def __init__(self,
                 max_samples=100000,
                 pretrain_size=200,
                 output_file=None,
                 show_plot=None,
                 data_expose=None,
                 name="test.pdf"):

        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_expose = DataExposure(
            data_expose if data_expose is not None else {})
        self.model_expose = None
        self.name = name

        self.test_samples = []
        self.state_tests = {}

        self.restore_states = {}
        

    def evaluate(self, stream, model, model_names=None):
        self.stream = stream
        self.model = model
        self.model_expose = ModelExposure(self.model)
        self.visualizer = InspectorVisualizer(name=self.name)
        self.global_sample_count = 0
        self.model = self._train_and_test()
        if self.show_plot:
            self.visualizer.hold()

        return model

    def _train_and_test(self):
        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        if self.pretrain_size > 0:
            X, y = self.stream.next_sample(self.pretrain_size)
            self.model.partial_fit(X=X, y=y)
            self.global_sample_count = self.pretrain_size

        while (self.global_sample_count < actual_max_samples) and self.stream.has_more_samples():
            X, y = self.stream.next_sample()
            prediction = self.model.predict(X)
            self.model.partial_fit(X, y)

            data_exposed_info = self.data_expose.get_exposed_info_at_sample(
                self.global_sample_count, self.restore_states)
            model_exposed_info = self.model_expose.get_exposed_info_at_sample(
                self.global_sample_count)
            self.visualizer.on_new_train_step(
                self.global_sample_count, data_exposed_info, model_exposed_info, X, y, prediction)

            if len(self.test_samples) < 100:
                self.test_samples.append(X)
            print(model_exposed_info['states'])
            for state_id in model_exposed_info['states']:
                state_predictions = []
                state = model_exposed_info['states'][state_id]
                for test_index, test_X in enumerate(self.test_samples):
                    state_predictions.append(state.main_model.predict(test_X))

                if state.id not in self.state_tests:
                    self.state_tests[state.id] = state_predictions

                if self.state_tests[state.id] != state_predictions:

                    if state.id != model_exposed_info['active_state']:
                        for alt_state_info in model_exposed_info['alternative_states']:
                            alt_state = alt_state_info[2]
                            if alt_state.id == state.id:
                                alt_model = alt_state.main_model
                                alt_preds = []
                                for test_X in self.test_samples:
                                    alt_preds.append(alt_model.predict(test_X))
                                if alt_preds == state_predictions:
                                    print(
                                        f"state {state_id}: alt same as state")

                        print(f"state {state.id} predictions different")
                        same_sum = 0
                        for pi in range(min(len(state_predictions), len(self.state_tests[state.id]))):
                            lh = self.state_tests[state.id][pi]
                            rh = state_predictions[pi]
                            if lh == rh:
                                same_sum += 1
                        print(
                            f"Similarity {same_sum / len(state_predictions)}")
                        # input()
                    self.state_tests[state.id] = state_predictions

            for state_id in model_exposed_info['states']:
                state_model = model_exposed_info['states'][state_id].main_model
                for alt_state_info in model_exposed_info['alternative_states']:
                    alt_state = alt_state_info[2]
                    alt_model = alt_state.main_model
                    if alt_model is state_model:
                        print("alt is state")
                        # input()

            if model_exposed_info['load_restore_state'] is not None:
                restore_id = model_exposed_info['load_restore_state'][0]
                new_id = model_exposed_info['load_restore_state'][1]
                print(model_exposed_info['states'])
                print(model_exposed_info['states']
                      [restore_id].main_model.splits_since_reset)
                # print(self.restore_states)
                print(self.restore_states[restore_id]
                      [1].main_model.splits_since_reset)
                same_sum = 0
                for pi in range(len(self.state_tests[restore_id])):
                    if self.state_tests[restore_id][pi] == self.restore_states[restore_id][0][pi]:
                        same_sum += 1
                print(
                    f"Similarity {same_sum / len(self.restore_states[restore_id][0])}")
                if self.state_tests[restore_id] != self.restore_states[restore_id][0]:
                    print("Restore Not Equal")

                    # input()

            if model_exposed_info['set_restore_state'] is not None:
                restore_id = model_exposed_info['set_restore_state']
                self.restore_states[restore_id] = [self.state_tests[restore_id], deepcopy(
                    model_exposed_info['states'][restore_id]), self.global_sample_count]

            self.global_sample_count += 1
