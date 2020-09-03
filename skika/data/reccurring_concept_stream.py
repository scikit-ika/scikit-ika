import random
from enum import Enum

import numpy as np
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.data import SineGenerator

from skika.data.wind_sim_generator import WindSimGenerator

class ConceptOccurence:
    """
    Represents a concept in a stream
    """
    def __init__(self, id, difficulty, noise, appearences, examples_per_appearence):
        self.id = id
        self.difficulty = difficulty
        self.noise = noise
        self.appearences = appearences
        self.examples_per_appearence = examples_per_appearence

    def __repr__(self):
        return f"<id: {self.id}, difficulty: {self.difficulty}, noise: {self.noise}, appearences: {self.appearences}, e_p_a: {self.examples_per_appearence}"


class RCStreamType(Enum):
    AGRAWAL = 0
    STAGGER = 1
    TREE = 2
    SEA = 3
    SINE = 4
    WINDSIM = 5
    RBF = 6
    WINDSIMD = 7


class Concept:
    """ Base concept class.
    A 'Concept' can be thought of as a relationship
    between features and label. Here we model different
    concept using streams produced by different
    generating functions, i.e. each concept is a given
    distribution of data and an observation is drawn
    from one concept.

    Parameters
    ----------
    stream: datastream
        The stream the concept will
        draw observations from.

    """
    def __init__(self, stream):
        self.datastream = stream

    def activate(self):
        pass

    def deactivate(self):
        pass

    def next_sample(self, batch_size):
        X, y = self.datastream.next_sample(batch_size)
        return (X, y)

    def get_info(self):
        return self.datastream.get_info()

    def n_remaining_samples(self):
        return self.datastream.n_remaining_samples()

    def n_samples(self):
        return self.datastream.n_samples()

    def get_datastream(self):
        return self.datastream

    def get_moa_string(self, start, end):
        return ""

    def get_supplementary_info(self):
        return None


class AGRAWALConcept(Concept):
    """ An AGRAWAL concept.

    Parameters
    ----------
    concept_id: int
        The ID of the AGRAWAL generating function
        to use. Should be within 0-9.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    """
    def __init__(self, concept_id=0, seed=None, noise=0):
        self.cf = concept_id
        self.seed = seed
        stream = AGRAWALGenerator(
            concept_id, random_state=seed, perturbation=noise)
        stream.prepare_for_use()
        super().__init__(stream)

    def get_moa_string(self, start, end):
        return f"(generators.AGRAWALGenerator -f {self.cf} -i {self.seed})"


class TREEConcept(Concept):
    """ A TREE concept.

    Parameters
    ----------
    concept_id: int
        The ID of the concept.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.
    desc: ConceptOccurence
        A class which describes the specific concept.

    """
    def __init__(self, concept_id=0, seed=None, noise=0, desc=None):
        self.cf = concept_id
        self.seed = seed
        self.difficulty = 0 if desc == None else desc.difficulty
        stream = RandomTreeGenerator(tree_random_state=seed, sample_random_state=seed,
                                     max_tree_depth=self.difficulty+2, min_leaf_depth=self.difficulty, n_classes=2)
        stream.prepare_for_use()
        super().__init__(stream)

    def get_moa_string(self, start, end):
        return f"(generators.RandomTreeGenerator -r {self.seed} -i {self.seed})"


class RBFConcept(Concept):
    """ An RBF concept.

    Parameters
    ----------
    concept_id: int
        The ID of the concept.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.
    desc: ConceptOccurence
        A class which describes the specific concept.

    """
    def __init__(self, concept_id=0, seed=None, noise=0, desc=None):
        self.cf = concept_id
        self.seed = seed
        self.difficulty = 0 if desc == None else desc.difficulty
        self.n_classes = 2
        # self.n_features = self.difficulty + 10
        self.n_features = 10
        self.n_centroids = self.difficulty * 5 + 15
        stream = RandomRBFGenerator(model_random_state=seed, sample_random_state=seed,
                                    n_centroids=self.n_centroids, n_classes=self.n_classes, n_features=self.n_features)
        stream.prepare_for_use()
        super().__init__(stream)

    def get_moa_string(self, start, end):
        return f"(generators.RandomRBFGenerator -r {self.seed} -i {self.seed} -n {self.n_centroids} )"


class SEAConcept(Concept):
    """ A SEA concept.

    Parameters
    ----------
    concept_id: int
        The ID of the SEA generating function
        to use. Should be within 0-9.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    """
    def __init__(self, concept_id=0, seed=None, noise=0):
        self.cf = concept_id
        self.seed = seed
        stream = SEAGenerator(
            classification_function=concept_id, random_state=seed)
        stream.prepare_for_use()
        super().__init__(stream)

    def get_moa_string(self, start, end):
        return f"(generators.SEAGenerator -f {self.cf + 1} -i {self.seed})"


class SINEConcept(Concept):
    """ A SINE concept.

    Parameters
    ----------
    concept_id: int
        The ID of the SINE generating function
        to use. Should be within 0-9.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    """
    def __init__(self, concept_id=0, seed=None, noise=0):
        self.cf = concept_id
        self.seed = seed
        stream = SineGenerator(
            classification_function=concept_id, random_state=seed)
        stream.prepare_for_use()
        super().__init__(stream)

    def get_moa_string(self, start, end):
        return f"(generators.SineGenerator -f {self.cf + 1} -i {self.seed})"


class STAGGERConcept(Concept):
    """ A STAGGER concept.

    Parameters
    ----------
    concept_id: int
        The ID of the STAGGER generating function
        to use. Should be within 0-3.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    """
    def __init__(self, concept_id=0, seed=None, noise=0):
        stream = STAGGERGenerator(
            classification_function=concept_id, random_state=seed)
        self.cf = concept_id
        self.seed = seed
        stream.prepare_for_use()
        super().__init__(stream)

    def get_moa_string(self, start, end):
        return f"(generators.STAGGERGenerator -f {self.cf + 1} -i {self.seed})"


class WindSimConcept(Concept):
    """ A WINDSIM concept.

    Parameters
    ----------
    concept_id: int
        The ID of the concept.
    seed: int
        The seed used by the random number generator.
    noise: int
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.
    desc: ConceptOccurence
        A class which describes the specific concept.

    """
    WINDSIMSTREAM = WindSimGenerator(num_sensors=20, sensor_pattern='grid')

    def __init__(self, concept_id=0, seed=None, noise=0, desc=None):
        self.cf = concept_id
        self.seed = seed
        self.difficulty = 0 if desc == None else desc.difficulty
        stream = self.WINDSIMSTREAM
        if not stream.prepared:
            stream.prepare_for_use()
        self.concept_id = concept_id
        super().__init__(stream)

    def activate(self):
        # self.datastream.set_concept_directed(self.seed, 1 + self.difficulty * 4)
        self.datastream.set_concept(self.seed, 1 + self.difficulty * 4)

        super().activate()

    def deactivate(self):
        super().deactivate()

    def next_sample(self, batch_size):
        # self.datastream.set_concept_directed(self.seed, 1 + self.difficulty * 4)
        self.datastream.set_concept(self.seed, 1 + self.difficulty * 4)
        return super().next_sample(batch_size)

    def get_info(self):
        return self.datastream.get_info(self.concept_id)

    def get_supplementary_info(self):
        return {"seed": self.seed, "difficulty": self.difficulty, **self.datastream.get_concept_supp_info(self.seed, 1 + self.difficulty * 4)}


class RecurringConceptStream:
    """ A stream featuring abrupt drift between given concepts.

    Parameters
    ----------

    rctype: RCStreamType
        An enum describing the type of stream

    num_samples: int
        The number of samples in the stream

    noise: float
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    concept_chain: list<int> or dict
        A dict with key observation number and value
        the concept beginning at that observation
        or
        A list of concept ids. A dict will be generated
        with each concept lasting its length given in desc
        or uniform length.

    seed: int
        Random seed.

    desc: dict<int><ConceptOccurence>
        A map of concept ID to options

    boost_first_occurance: bool
        If true, double the observations drawn from
        the first occurence of a concept. Allows
        a better model to be built and stored.

    Examples
    --------

    >>> # An example stream using the STAGGER Generator.
    >>> # Starts using generating function 0, then at
    >>> # observation 5000 transitions to generating function
    >>> # 1 then at 10000 transitions back to 0.
    >>> from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, ConceptOccurence
    >>> concept_chain = {0: 0, 5000: 1, 10000: 0}
    >>> num_samples = 15000
    >>> # init concept
    >>> concept_0 = ConceptOccurence(id=0, difficulty=2, noise=0,
                        appearences=2, examples_per_appearence=5000)
    >>> concept_1 = ConceptOccurence(id=1, difficulty=3, noise=0,
                        appearences=1, examples_per_appearence=5000)
    >>> desc = {0: concept_0, 1: concept_1}
    >>> datastream = RecurringConceptStream(
                        rctype=RCStreamType.STAGGER,
                        num_samples=num_samples,
                        noise=0,
                        concept_chain=concept_chain,
                        seed=42,
                        desc=desc,
                        boost_first_occurance=False)
    >>> datastream.has_more_samples()
    True
    >>> datastream.get_drift_info()
    {0: 0, 5000: 1, 10000: 0}
    >>> datastream.n_remaining_samples()
    15000
    >>> datastream.get_stream_info()
    {0: 0, 5000: 1, 10000: 0}
    0 - 5000: STAGGERGenerator(balance_classes=False, classification_function=0,
                    random_state=42)
    5000 - 10000: STAGGERGenerator(balance_classes=False, classification_function=1,
                    random_state=43)
    10000 - 15000: STAGGERGenerator(balance_classes=False, classification_function=0,
                    random_state=42)
    >>> datastream.get_moa_stream_info()
    {0: 0, 5000: 1, 10000: 0}
    '(ConceptDriftStream -s (generators.STAGGERGenerator -f 1 -i 42) -d (ConceptDriftStream -s (generators.STAGGERGenerator -f 2 -i 43) -d (generators.STAGGERGenerator -f 1 -i 42) -p 5000 -w 1) -p 5000 -w 1)'
    >>> datastream.get_supplementary_info()
    >>> datastream.next_sample()
    (array([[2., 0., 2.]]), array([0]))
    >>> datastream.n_remaining_samples()
    14999
    >>> datastream.next_sample()
    (array([[2., 0., 0.]]), array([0]))
    >>> datastream.n_remaining_samples()
    14998
    """
    def __init__(self, rctype, num_samples, noise, concept_chain, seed=None, desc=None, boost_first_occurance=True):
        if seed == None:
            seed = random.randint(0, 10000)
        self.random_seed = seed
        self.example_count = 0
        self.drifted = False

        self.rctype = rctype
        self.num_samples = num_samples
        self.noise = noise
        np.random.seed(seed)
        self.num_concepts = len(concept_chain)
        self.concept_chain = {}
        if type(concept_chain) is dict:
            self.concept_chain = concept_chain
        else:
            example_cumulative = 0
            cID = 0
            examples_in_concept = 0
            seen_cID = set()
            for i in concept_chain:
                cID = i
                first_occurance = cID not in seen_cID
                seen_cID.add(cID)
                self.concept_chain[example_cumulative] = cID
                if desc == None or cID not in desc:
                    examples_in_concept = num_samples / (self.num_concepts)
                else:
                    examples_in_concept = desc[cID].examples_per_appearence
                    if boost_first_occurance and first_occurance:
                        examples_in_concept = examples_in_concept * 2
                example_cumulative += examples_in_concept
            self.num_samples = example_cumulative

        self.concepts_used = sorted(
            list(np.unique(np.array(list(self.concept_chain.values())))))
        print(self.concepts_used)
        self.num_concepts_used = len(self.concepts_used)
        lowest_concept_index = sorted(self.concept_chain.keys())[0]
        self.current_concept = self.concept_chain[lowest_concept_index]

        self.concepts = []
        self.num_concepts_availiable = 0
        gen_func = None
        if self.rctype == RCStreamType.AGRAWAL:
            self.num_concepts_availiable = 10
            def gen_func(i, desc): return AGRAWALConcept(
                i, self.random_seed + i)

        if self.rctype == RCStreamType.TREE:
            self.num_concepts_availiable = 100
            def gen_func(i, desc): return TREEConcept(
                i, self.random_seed + i, desc=desc)

        if self.rctype == RCStreamType.RBF:
            self.num_concepts_availiable = 100
            def gen_func(i, desc): return RBFConcept(
                i, self.random_seed + i, desc=desc)

        if self.rctype == RCStreamType.SEA:
            self.num_concepts_availiable = 4
            def gen_func(i, desc): return SEAConcept(i, self.random_seed + i)

        if self.rctype == RCStreamType.SINE:
            self.num_concepts_availiable = 4
            def gen_func(i, desc): return SINEConcept(i, self.random_seed + i)

        if self.rctype == RCStreamType.STAGGER:
            self.num_concepts_availiable = 3
            def gen_func(i, desc): return STAGGERConcept(
                i, self.random_seed + i)

        if self.rctype == RCStreamType.WINDSIM:
            self.num_concepts_availiable = 10000
            def gen_func(i, desc): return WindSimConcept(
                i, self.random_seed + i, desc=desc)

        if self.rctype == RCStreamType.WINDSIMD:
            self.num_concepts_availiable = 10000
            def gen_func(i, desc): return WindSimConcept(
                i, self.random_seed + i, desc=desc)

        print(self.num_concepts_used)
        print(self.num_concepts_availiable)
        if self.num_concepts_used > self.num_concepts_availiable:
            raise ValueError(
                "Generator does not have enough concepts for the given concept chain")

        for i in self.concepts_used:
            concept_description = None
            if desc != None and i in desc:
                concept_description = desc[i]
            generator = gen_func(i, concept_description)
            self.concepts.append(generator)

        self.seen_y_values = []

    def __str__(self):
        return f"Type: {self.rctype}, concept chain: {self.concept_chain}\ncurrent concept: {self.current_concept}, current example: {self.example_count}\ndetector info: {self.concepts[self.current_concept].get_info()}"

    def next_sample(self, batch_size=1):
        samples = self.concepts[self.current_concept].next_sample(batch_size)
        new_y_vals = []
        for s in samples[1]:
            if s not in self.seen_y_values:
                self.seen_y_values.append(s)
            noise_rand = np.random.random_sample()
            if noise_rand < self.noise:
                new_y = np.random.choice(self.seen_y_values)
            else:
                new_y = s
            new_y_vals.append(new_y)
        samples = (samples[0], np.array(new_y_vals))
        concept_switch_possibilities = []
        for concept_switch_index in self.concept_chain.keys():
            if self.example_count < concept_switch_index <= self.example_count + batch_size:
                concept_switch_possibilities.append(concept_switch_index)
        if len(concept_switch_possibilities) > 0:
            concept_switch_possibilities.sort(reverse=True)
            last_concept_index = concept_switch_possibilities[0]
            self.current_concept = self.concept_chain[last_concept_index]
            self.drifted = True

        else:
            self.drifted = False
        self.example_count += batch_size

        return samples

    def has_more_samples(self):
        return self.concepts[self.current_concept].datastream.has_more_samples() and (self.example_count < self.num_samples)

    def get_drift_info(self):
        return self.concept_chain

    def n_remaining_samples(self):
        return self.num_samples - self.example_count

    def get_stream_info(self):
        """ Prints information about the
        concepts included in the stream.

        """
        keys = list(self.concept_chain.keys())
        start_index = keys[0]
        concept = self.concepts[self.concept_chain[start_index]]
        print(self.concept_chain)
        for i in keys[1:]:
            end_index = i
            print(f"{start_index} - {end_index}: {concept.get_info()}")
            start_index = end_index
            concept = self.concepts[self.concept_chain[start_index]]
        print(f"{start_index} - {self.num_samples}: {concept.get_info()}")

    def get_moa_stream_string(self, concepts = None):
        """ Returns a string to run the corresponding
        stream in MOA.

        """
        if concepts is None:
            concepts = self.concepts
        if len(concepts) < 1:
            return ""
        if len(concepts) == 1:
            c = concepts[0]
            concept = c[0]
            start = c[1]
            end = c[2]
            return concept.get_moa_string(start, end)
        else:
            c = concepts[0]
            concept = c[0]
            start = c[1]
            end = c[2]
            return f"(ConceptDriftStream -s {concept.get_moa_string(start, end)} -d {self.get_moa_stream_string(concepts[1:])} -p {end - start} -w 1)"

    def get_moa_stream_info(self):
        """ Returns a string to run the corresponding
        stream in MOA.

        """
        print(self.concept_chain)
        keys = list(self.concept_chain.keys())
        start_index = keys[0]
        concept = self.concepts[self.concept_chain[start_index]]
        concepts = []
        for i in keys[1:]:
            end_index = i
            concepts.append((concept, start_index, end_index))
            start_index = end_index
            concept = self.concepts[self.concept_chain[start_index]]
        concepts.append((concept, start_index, self.num_samples))
        return self.get_moa_stream_string(concepts)

    def get_supplementary_info(self):
        """ Returns supplementary info about
        each concept.

        """
        return self.concepts[self.current_concept].get_supplementary_info()


class RecurringConceptGradualStream(RecurringConceptStream):
    """ A stream featuring gradual drift between given concepts.
    Uses the scikit-multiflow concept drift stream to blend concepts over
    a window.

    Parameters
    ----------

    rctype: RCStreamType
        An enum describing the type of stream

    num_samples: int
        The number of samples in the stream

    noise: float
        The probability that noise will happen in the generation. At each
        new sample generated, the sample with will perturbed by the amount of
        perturbation.
        Values go from 0.0 to 1.0.

    concept_chain: list<int> or dict
        A dict with key observation number and value
        the concept begining at that observation
        or
        A list of concept ids. A dict will be generated
        with each concept lasting its length given in desc
        or uniform length.

    window_size: int
        The number of observations each gradual drift is
        spread over.

    seed: int
        Random seed.

    desc: dict<int><ConceptOccurence>
        A map of concept ID to options

    boost_first_occurance: bool
        If true, double the observations drawn from
        the first occurence of a concept. Allows
        a better model to be built and stored.

    Examples
    --------

    >>> # An example stream using the STAGGER Generator.
    >>> # Starts using generating function 0, then at
    >>> # observation 5000 transitions to generating function
    >>> # 1 then at 10000 transitions back to 0.
    >>> from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptGradualStream, ConceptOccurence
    >>> concept_chain = {0: 0, 5000: 1, 10000: 0}
    >>> num_samples = 15000
    >>> # init concept
    >>> concept_0 = ConceptOccurence(id=0, difficulty=2, noise=0,
                        appearences=2, examples_per_appearence=5000)
    >>> concept_1 = ConceptOccurence(id=1, difficulty=3, noise=0,
                        appearences=1, examples_per_appearence=5000)
    >>> desc = {0: concept_0, 1: concept_1}
    >>> datastream = RecurringConceptGradualStream(
                        rctype=RCStreamType.STAGGER,
                        num_samples=num_samples,
                        noise=0,
                        concept_chain=concept_chain,
                        window_size=1000,
                        seed=42,
                        desc=desc,
                        boost_first_occurance=False)
    >>> datastream.has_more_samples()
    True
    >>> datastream.get_drift_info()
    {0: 0, 5000: 1, 10000: 0}
    >>> datastream.n_remaining_samples()
    15000
    >>> datastream.get_stream_info()
    {0: 0, 5000: 1, 10000: 0}
    0 - 5000: STAGGERGenerator(balance_classes=False, classification_function=0,
                    random_state=42)
    5000 - 10000: STAGGERGenerator(balance_classes=False, classification_function=1,
                    random_state=43)
    10000 - 15000: STAGGERGenerator(balance_classes=False, classification_function=0,
                    random_state=42)
    >>> datastream.get_moa_stream_info()
    {0: 0, 5000: 1, 10000: 0}
    '(ConceptDriftStream -s (generators.STAGGERGenerator -f 1 -i 42) -d (ConceptDriftStream -s (generators.STAGGERGenerator -f 2 -i 43) -d (generators.STAGGERGenerator -f 1 -i 42) -p 5000 -w 1) -p 5000 -w 1)'
    >>> datastream.get_supplementary_info()
    >>> datastream.next_sample()
    (array([[2., 0., 2.]]), array([0]))
    >>> datastream.n_remaining_samples()
    14999
    >>> datastream.next_sample()
    (array([[2., 0., 0.]]), array([0]))
    >>> datastream.n_remaining_samples()
    14998
    """
    def __init__(self, rctype, num_samples, noise, concept_chain, window_size=1000, seed=None, desc=None, boost_first_occurance=True):
        self.in_drift = False
        self.drift_switch = False
        self.window_size = window_size
        self.transition_stream = None
        super().__init__(rctype, num_samples, noise, concept_chain, seed=seed,
                         desc=desc, boost_first_occurance=boost_first_occurance)

    def next_sample(self, batch_size=1):
        if batch_size > 1:
            print("Only batch size of 1 for now")
            return None

        if not self.in_drift:
            samples = self.concepts[self.current_concept].next_sample(
                batch_size)
        else:
            samples = self.transition_stream.next_sample(batch_size)

        last_switch_point = 0 - self.window_size // 2
        next_switch_point = self.num_samples + self.window_size
        self.example_count += batch_size
        for concept_switch_index in sorted(self.concept_chain.keys()):
            if(concept_switch_index <= self.example_count):
                last_switch_point = concept_switch_index
            if concept_switch_index >= self.example_count:
                next_switch_point = concept_switch_index
                break

        self.drifted = False
        if not self.in_drift:
            if self.example_count >= next_switch_point - self.window_size // 2:
                self.in_drift = True
                self.drift_switch = True
                self.transition_stream = ConceptDriftStream(stream=self.concepts[self.concept_chain[last_switch_point]].get_datastream(),
                                                            drift_stream=self.concepts[self.concept_chain[next_switch_point]].get_datastream(), position=self.window_size // 2, width=self.window_size)
                self.transition_stream.prepare_for_use()
        else:
            if self.example_count == next_switch_point:
                self.current_concept = self.concept_chain[next_switch_point]
                self.drifted = True
                self.drift_switch = False
            if self.example_count >= (last_switch_point + self.window_size // 2) and not self.drift_switch:
                self.in_drift = False

        return samples
