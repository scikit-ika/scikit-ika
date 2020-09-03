import sys, os, math
import argparse
import random
import pickle
import json
from skika.data.reccurring_concept_stream import RCStreamType
from skika.data.reccurring_concept_stream import ConceptOccurence
from skika.data.reccurring_concept_stream import RecurringConceptStream
from skika.data.reccurring_concept_stream import RecurringConceptGradualStream

import numpy as np
import pandas as pd

class DatastreamOptions:
    """
    Options for generating a concept.
    """
    def __init__(self, noise, num_concepts, hard_diff, easy_diff, hard_appear, easy_appear, hard_prop, examples_per_appearence,
            stream_type, seed, gradual):
        self.noise = noise
        self.num_concepts = num_concepts
        self.hard_diff = hard_diff
        self.easy_diff = easy_diff
        self.hard_appearences = hard_appear
        self.easy_appearences = easy_appear
        self.hard_proportion = hard_prop
        self.examples_per_appearence = examples_per_appearence
        self.stream_type = stream_type
        self.seed = seed
        self.gradual = gradual


def generate_concept_chain(concept_desc, sequential):
    """
    Given a list of availiable concepts, generate a dict with (start, id) pairs
    giving the start of each concept.

    Parameters
    ----------

    sequential: bool
        If true, concept transitions are
        determined by ID without randomness.
    """
    concept_chain = []
    num_samples = 0
    more_appearences = True
    appearence = 0
    while more_appearences:
        concepts_still_to_appear = []
        for cID in concept_desc:
            concept = concept_desc[cID]
            if concept.appearences > appearence:
                concepts_still_to_appear.append(concept)
        more_appearences = len(concepts_still_to_appear) > 0
        for concept in concepts_still_to_appear:
            concept_chain.append(concept.id)
            num_samples += concept.examples_per_appearence
        appearence += 1
    if not sequential:
        random.shuffle(concept_chain)
    return concept_chain, num_samples

def generate_pattern_concept_chain(concept_desc, sequential):
    """
    Given a list of availiable concepts, generate a dict with (start, id) pairs
    giving the start of each concept.
    This is generated using a random markov model, so specific transtion patterns
    have unique properties.

    Parameters
    ----------

    sequential: bool
        If true, concept transitions are
        determined by ID without randomness.
    """
    concept_chain = {}
    num_samples = 0
    more_appearences = True
    appearence = 0



    pattern = {}
    seg_lengths = {}
    # Set current
    for current in concept_desc:
        if current not in pattern:
            pattern[current] = {}
            seg_lengths[current] = {}
        for previous in concept_desc:
            if previous not in pattern[current]:
                pattern[current][previous] = {}
                seg_lengths[current][previous] = np.random.randint(concept_desc[current].examples_per_appearence - (concept_desc[current].examples_per_appearence//2), concept_desc[current].examples_per_appearence + (concept_desc[current].examples_per_appearence//2))
            for next_concept in concept_desc:
                num_seen = np.random.randint(0, 10)

                segment_length = seg_lengths[current][previous]
                if next_concept not in pattern[current][previous]:
                    pattern[current][previous][next_concept] = [num_seen, segment_length]
    print(pattern)
    last_concept_used = None
    for i,current in enumerate(list(concept_desc.keys())):
        concept_chain[num_samples] = current
        if last_concept_used is None:
            transition_point = concept_desc[current].examples_per_appearence * 5
            num_samples += transition_point
        if last_concept_used is not None and (i + 1) < len(concept_desc):
            print(pattern[current].keys())
            next_choice = pattern[current][last_concept_used][list(concept_desc.keys())[i+1]]
            transition_point = next_choice[1]
            num_samples += transition_point
        last_concept_used = current

    add_first = True
    seen_count = {}
    print(concept_chain)
    while more_appearences:
        print(concept_chain)
        concepts_still_to_appear = []
        for cID in concept_desc:
            concept = concept_desc[cID]
            if concept.id not in seen_count:
                seen_count[concept.id] = 0
            if concept.appearences > seen_count[concept.id]:
                if concept.id != last_concept_used:
                    concepts_still_to_appear.append(concept)
        more_appearences = len(concepts_still_to_appear) > 0
        if not more_appearences:
            break

        next_concept = np.random.choice(concepts_still_to_appear)
        if add_first:
            previous_concept_id = list(concept_desc.keys())[-2]
            current_concept_id = list(concept_desc.keys())[1]
            matching_patterns = pattern[current_concept_id][previous_concept_id]
            next_options = []
            for next_id in matching_patterns:
                next_options.append((next_id, *matching_patterns[next_id]))
            total_transition_count = sum([x[1] for x in next_options])
            next_choice = np.random.choice(list(range(len(next_options))), p = [x[1] / total_transition_count for x in next_options])
            next_choice = next_options[next_choice]
            print(next_choice)
            transition_point = next_choice[2]
            num_samples += transition_point
            add_first = False

        seen_count[next_concept.id] += 1
        concept_chain[num_samples] = next_concept.id
        transition_point = concept_desc[next_concept.id].examples_per_appearence
        if last_concept_used is not None:
            matching_patterns = pattern[next_concept.id][last_concept_used]
            next_options = []
            for next_id in matching_patterns:
                next_options.append((next_id, *matching_patterns[next_id]))
            total_transition_count = sum([x[1] for x in next_options])
            next_choice = np.random.choice(list(range(len(next_options))), p = [x[1] / total_transition_count for x in next_options])
            next_choice = next_options[next_choice]
            transition_point = next_choice[2]
        num_samples += transition_point
        last_concept_used = next_concept.id

    print(concept_chain)
    print(num_samples)
    return concept_chain, num_samples


def generate_experiment_concept_chain(ds_options, sequential, pattern):
    """
    Generates a list of concepts for a datastream.

    Parameters
    ----------

    ds_options:
        options for the data stream

    sequential: bool
        If concepts should be sequential not random

    pattern: bool
        If transitions should have an underlying pattern

    Returns
    -------
    concept_chain: dict<int><int>

    num_samples: int

    concept_descriptions: list<ConceptOccurence>
    """
    num_hard = math.floor(ds_options.hard_proportion * ds_options.num_concepts)
    num_easy = ds_options.num_concepts - num_hard

    concept_desc = {}
    cID = 0
    for i in range(num_hard):
        concept = ConceptOccurence(cID, ds_options.hard_diff, ds_options.noise, ds_options.hard_appearences, ds_options.examples_per_appearence)
        concept_desc[cID] = concept
        cID += 1
    for i in range(num_easy):
        concept = ConceptOccurence(cID, ds_options.easy_diff, ds_options.noise, ds_options.easy_appearences, ds_options.examples_per_appearence)
        concept_desc[cID] = concept
        cID += 1
    if pattern:
        cc, ns = generate_pattern_concept_chain(concept_desc, sequential)
    else:
        cc, ns = generate_concept_chain(concept_desc, sequential)
    return cc, ns, concept_desc


class ExperimentOptions:
    def __init__(self, seed, stream_type, directory):
        self.seed = seed
        self.stream_type = stream_type
        self.experiment_directory = directory
        self.batch_size = 1

def make_reuse_folder(experiment_directory):
    if not os.path.exists(experiment_directory):
        print('making directory')
        print(experiment_directory)
        os.makedirs(experiment_directory)
        os.makedirs(f'{experiment_directory}{os.sep}archive')

def get_concepts(gt_concepts, ex_index, num_samples):
    """ Given [(gt_concept, start_i, end_i)...]
        Return the ground truth occuring at a given index."""

    gt_concept = None
    for gt_c, s_i, e_i in gt_concepts:
        if s_i <= ex_index < e_i:
            gt_concept = gt_c
            break
    return (gt_concept)

def get_model_drifts(num_samples, datastream):
    detections = np.zeros(num_samples)
    for d_i, d in enumerate(datastream.get_drift_info().keys()):
        if d >= len(detections):
            continue
        detections[d] = 1
    return detections

def get_concept_by_example(num_samples, ground_truth_concepts):
    gt_by_ex = []
    for ex in range(num_samples):
        sample_gt_concept = get_concepts(ground_truth_concepts, ex, num_samples)
        gt_by_ex.append(sample_gt_concept)
    return gt_by_ex

def get_concepts_from_model(concept_chain, num_samples):
    # Have a dict of {ts: concept}
    # Transform to [(concept, start_ts, end_ts)]
    switch_indexes = list(concept_chain.keys())
    gt_concept = concept_chain[switch_indexes[0]]
    start = switch_indexes[0]
    seen_unique = []
    ground_truth_concepts = []
    for ts_i,ts in enumerate(switch_indexes[1:]):
        end, new_gt_concept = ts, concept_chain[ts]
        if gt_concept not in seen_unique:
            seen_unique.append(gt_concept)
        gt_concept_index = seen_unique.index(gt_concept)
        ground_truth_concepts.append((gt_concept_index, start, end))
        gt_concept, start = new_gt_concept, end
    end = num_samples
    if gt_concept not in seen_unique:
        seen_unique.append(gt_concept)
    gt_concept_index = seen_unique.index(gt_concept)
    ground_truth_concepts.append((gt_concept_index, start, end))

    return get_concept_by_example(num_samples, ground_truth_concepts)

class NPEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def saveStreamToArff(filename, stream_examples, stream_supplementary, arff):
    """ Save examples to an ARFF file.

    Parameters
    ----------

    filename: str
        filename with extention

    stream_examples: list
        list of examples [[X, y]]

    stream_supplementary: list
        list of supplementary info for each observation

    arff: bool
        Use arff or CSV

    """
    with open(f"{filename}", 'w') as f:
        if len(stream_examples) > 0:
            if arff:
                f.write(f"@RELATION stream\n")
                first_example = stream_examples[0]
                for i, x in enumerate(first_example[0].tolist()[0]):
                    values = []
                    for row in stream_examples:
                        try:
                            values.append(row[0].tolist()[0][i])
                        except:
                            print(row)

                        # values = np.unique(np.array([x[0].tolist()[0][i] for x in stream_examples]))
                    values = np.unique(np.array(values))
                    if len(values) < 10:
                        # f.write(f"@ATTRIBUTE x{i}  {{{','.join([str(x) for x in values.tolist()])}}}\n")
                        f.write(f"@ATTRIBUTE x{i}  NUMERIC\n")
                    else:
                        f.write(f"@ATTRIBUTE x{i}  NUMERIC\n")

                for i, y in enumerate(first_example[1].tolist()):
                    values = np.unique(np.array([y[1].tolist()[i] for y in stream_examples]))
                    if len(values) < 10:
                        # f.write(f"@ATTRIBUTE y{i}  {{{','.join([str(x) for x in values.tolist()])}}}\n")
                        f.write(f"@ATTRIBUTE y{i}  NUMERIC\n")
                    else:
                        f.write(f"@ATTRIBUTE y{i}  NUMERIC\n")
                f.write(f"@DATA\n")
            for l in stream_examples:
                for x in l[0].tolist()[0]:
                    f.write(f"{x},")
                for y in l[1].tolist():
                    f.write(f"{y},")
                f.write(f"\n")
    file_type = "ARFF" if arff else "csv"
    with open(f"{filename.replace(f'.{file_type}', '_supp.txt')}", 'w') as f:
        for line in stream_supplementary:
            # print(line)
            # print(pickle.dumps(line), file=f)
            print(json.dumps(line, cls=NPEncoder), file=f)


def save_stream(options, ds_options, pattern = False, arff = False):
    """ Create, generate and save a data stream to csv or ARFF.

    Parameters
    ----------

    options: ExperimentOptions
        options for the experiment

    ds_options: DatastreamOptions
        options for the stream

    pattern: bool
        Should use a pattern for concept ordering

    arff: bool
        Save to ARFF


    """
    cc, ns, desc = generate_experiment_concept_chain(ds_options, options.sequential, pattern)
    options.ds_length = ns
    options.concept_chain = cc
    print(desc)

    if ds_options.gradual:
        datastream = RecurringConceptGradualStream(options.stream_type, ns, ds_options.noise, options.concept_chain, window_size = options.window_size, seed = options.seed, desc = desc)
    else:
        datastream = RecurringConceptStream(options.stream_type, ns, ds_options.noise, options.concept_chain, seed = options.seed, desc = desc)
    with open(f"{options.experiment_directory}{os.sep}{ds_options.seed}_concept_chain.pickle", "wb") as f:
        pickle.dump(datastream.concept_chain, f)
    with open(f"{options.experiment_directory}{os.sep}{ds_options.seed}_dsinfo.txt", "w+") as f:
        f.write(json.dumps(options.__dict__, default=lambda o: '<not serializable>'))
        f.write('\n')
        f.write(json.dumps(ds_options.__dict__, default=lambda o: '<not serializable>'))
    ns = datastream.num_samples
    print(datastream.concept_chain)
    stream_examples = []
    stream_supplementary = []
    update_percent = ns // 1000
    ex = 0
    while datastream.has_more_samples():
        X,y = datastream.next_sample(options.batch_size)
        supp = datastream.get_supplementary_info()
        stream_examples.append((X, y))
        stream_supplementary.append(supp)
        ex += 1
        # print(f"{ex}\r", end = "")
        if ex % update_percent == 0:
            print(f"{ex / update_percent}%\r", end = "")

    gts = get_concepts_from_model(datastream.concept_chain, ns)

    ground_truth = np.array(gts)
    sys_results = {}
    sys_results['ground_truth_concept'] = np.array(gts)
    sys_results['drift_occured'] = get_model_drifts(ns, datastream)
    res_data = pd.DataFrame(data = sys_results)
    res_data.to_csv(f'{options.experiment_directory}{os.sep}drift_info.csv')

    file_type = "ARFF" if arff else "csv"
    arff_full_filename = f"{options.experiment_directory}{os.sep}stream-{ds_options.seed}.{file_type}"
    arff_filename = f"{ds_options.seed}.{file_type}"
    saveStreamToArff(arff_full_filename, stream_examples, stream_supplementary, arff)

if __name__ == '__main__':
        # Set config params, get commandline params
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int,
        help="Random seed", default=None)
    ap.add_argument("-w", "--window", type=int,
        help="window", default=1000)
    ap.add_argument("-g", "--gradual", action="store_true",
        help="set if gradual shift is desired")
    ap.add_argument("-m", "--many", action="store_true",
        help="Generate many, not from options")
    ap.add_argument("-u", "--uniform", action="store_true",
        help="layout concepts sequentially")
    ap.add_argument("-st", "--streamtype",
        help="tdata generator for stream", default="STAGGER")
    ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
    ap.add_argument("-n", "--noise", type=float,
            help="noise", default=0)
    ap.add_argument("-nc", "--numconcepts", type=int,
        help="Number of Concepts", default=10)

    ap.add_argument("-hd", "--harddifficulty", type=int,
        help="Difficulty for a hard concept", default=3)
    ap.add_argument("-ed", "--easydifficulty", type=int,
        help="Difficulty for an easy concept", default=0)
    ap.add_argument("-ha", "--hardappear", type=int,
        help="How many times a hard concept appears", default=10)
    ap.add_argument("-ea", "--easyappear", type=int,
        help="How many times an easy concept appears", default=10)
    ap.add_argument("-hp", "--hardprop", type=float,
        help="Proportion of hard to easy concepts", default=0.1)
    ap.add_argument("-epa", "--examplesperapp", type=int,
        help="How many examples each concept appearence lasts for", default=4000)
    ap.add_argument("-r", "--repeat", type=int,
        help="Number of Concepts", default=1)
    ap.add_argument("-p", "--pattern", action="store_true")
    ap.add_argument("-a", "--arff", action="store_true")
    args = vars(ap.parse_args())

    seed = args['seed']
    if seed == None:
        seed = random.randint(0, 10000)
        args['seed'] = seed

    noise = args['noise']
    num_concepts = args['numconcepts']
    st = args['streamtype']
    print(args['many'])
    if args['many']:
        # for st in ['RBF', 'TREE', 'WINDSIM']:

            # for noise in [0, 0.05, 0.1, 0.25]:
        for noise in [0, 0.05, 0.1]:
            for st in ['TREE', 'RBF']:
                # for num_concepts in [5, 25, 50, 100]:
                # for num_concepts in [50]:
                for nc in [50]:
                    for d in [1, 2, 3]:

                        for hp in [0, 0.05, 0.1, 0.15]:
                            if st == 'TREE' and d == 1 and hp == 0:
                                continue
                            for r in range(0, 3):
                                seed = random.randint(0, 10000)
                                args['seed'] = seed
                                ds_options = DatastreamOptions(noise, num_concepts, args['harddifficulty'] + d, args['easydifficulty'] + d, args['hardappear'],
                                        args['easyappear'], hp, args['examplesperapp'], RCStreamType[st], seed, args['gradual'])
                                experiment_info = ds_options.__dict__.copy()
                                experiment_info.pop('seed')
                                experiment_info = list(experiment_info.values())
                                experiment_name = '_'.join((str(x) for x in experiment_info)).replace('.', '-')
                                experiment_directory = f"{os.getcwd()}{os.sep}{args['directory']}{os.sep}{noise}{os.sep}{experiment_name}{os.sep}{ds_options.seed}"
                                options = ExperimentOptions(seed, ds_options.stream_type, experiment_directory)
                                make_reuse_folder(options.experiment_directory)
                                save_stream(options, ds_options, arff = args['arff'])
    else:
        for r in range(args['repeat']):
            ds_options = DatastreamOptions(noise, num_concepts, args['harddifficulty'], args['easydifficulty'], args['hardappear'],
                    args['easyappear'], args['hardprop'], args['examplesperapp'], RCStreamType[st], seed, args['gradual'])

            experiment_info = ds_options.__dict__.copy()
            experiment_info.pop('seed')
            experiment_info = list(experiment_info.values())
            experiment_name = '_'.join((str(x) for x in experiment_info)).replace('.', '-')
            experiment_directory = f"{os.getcwd()}{os.sep}{args['directory']}{os.sep}{noise}{os.sep}{experiment_name}{os.sep}{ds_options.seed}"
            options = ExperimentOptions(seed, ds_options.stream_type, experiment_directory)
            options.sequential = args['uniform']
            options.window_size = args['window']
            make_reuse_folder(options.experiment_directory)
            save_stream(options, ds_options, pattern = args['pattern'], arff = args['arff'])
            seed = random.randint(0, 10000)
            args['seed'] = seed
