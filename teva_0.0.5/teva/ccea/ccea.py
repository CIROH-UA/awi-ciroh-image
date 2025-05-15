""" The Conjunctive Clause Evolutionary Algorithm (CCEA) """
from typing import Callable
import logging
import copy
import datetime

import numpy as np
import pandas as pd
from numpy.random import Generator, MT19937

from teva.utilities import hyge, teva_math
from teva.base import feature
from teva.ccea import conjunctive_clause, evolution, fitness
from teva.base.evolutionary_algorithm import EvolutionaryData, EvolutionaryAlgorithm, EvolutionaryAlgorithmError


class CCEAError(EvolutionaryAlgorithmError):
    """ Default CCEA Error """

class CCEAData(EvolutionaryData):
    """ CCEA run data struct """

class CCEA(EvolutionaryAlgorithm):
    """Conjunctive Clause Evolutionary Algorithm

    :param offspring_per_gen: the number of new offspring created every g_n generations
    :param total_generations: Total number of generations to run the algorithm for
    :param max_order: The maximum order of ccs that can be stored in the archive
    :param n_age_layers: The number of layers that are not archived
    :param layer_size: Maximum population of age layers
    :param gen_per_growth: The number of generations between age cycles
    :param offspring_per_gen: The number of new offspring evolved each generation through crossover or mutation
    :param archive_offspring_per_gen: The number of new offspring evolved each generation from the archive
    :param archive_bin_size: Maximum population of archive bins
    :param num_new_pop: The number of archive offspring created per evolution
    :param p_crossover: Probability that crossover will occur during evolution
    :param p_mutation: Probability that mutation will occur during evolution
    :param tournament_size: Number of parents with replacement that are in the mating tournament,
         only the most fit will mate
    :param selection_exponent: An exponent that increases the probability of selecting CCs whose target observations
        are underrepresented in the archived DNFs for the creation of novel DNFs
    :param use_sensitivity: enable use of feature sensitivity test
    :param sensitivity_threshold: feature sensitivity minimum threshold

    :param fitness_function: The fitness function to use for determining clause fitness The fitness function must
        take the following form:

            fitness_function(number, number, number, number) -> float

    :param seed: random seed for reproducibility
    :param run_name: The name of the current run of this algorithm.  This will be used in logging and output names
    :param smart_seeding: If True, seeding for new DNFs will take the current archive into account to ensure a greater
        diversity of clauses

    :param p_wildcard: Probability that a feature selected from mutation will be removed from the clause
    :param selective_mutation: If true, mutation will use wildcards to be more selective
    :param feature_selection_ratio: The ratio of features to be selected from each parent during crossover
    :param fitness_threshold: The initial threshold to use for CC fitnesses
    :param feature_names: A list of names for each feature that will be used in exporting results
    """
    def __init__(self,
                 # Generic EA Inputs
                 total_generations: int,  # TotGens
                 max_order: int,  # ALna
                 n_age_layers: int,  # MAXcc
                 layer_size: int,  # NonArchLMax
                 gen_per_growth: int = 3,  # GENn
                 offspring_per_gen: int = None,  # POPn
                 archive_offspring_per_gen: int = None,  # ArchOff
                 archive_bin_size: int = None,
                 num_new_pop: int = None,
                 p_crossover: float = 0.5,  # Px
                 p_mutation: float = 0.1,  # Pm
                 tournament_size: int = 3,  # TournSize
                 selection_exponent: float = 3.0,  # SelExp,
                 use_sensitivity: bool = True,
                 sensitivity_threshold: float = 0,
                 fitness_function: Callable = hyge.hygepmf,
                 seed: int = None,
                 run_name: str = None,
                 smart_seeding: bool = True,

                 # CCEA Specific Inputs
                 p_wildcard: float = 0.75,
                 selective_mutation: bool = False,
                 feature_selection_ratio: float = 0.5,
                 fitness_threshold: float = None,
                 feature_names: list[str] = None):

        if p_mutation is None:
            self.p_m = 1.0 / float(layer_size)

        if run_name == "":
            run_name = f"dnfea_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        super().__init__(total_generations=total_generations,
                         layer_size=layer_size,
                         n_age_layers=n_age_layers,
                         max_order=max_order,
                         p_crossover=p_crossover,
                         p_mutation=p_mutation,
                         use_sensitivity=use_sensitivity,
                         sensitivity_threshold=sensitivity_threshold,
                         tournament_size=tournament_size,
                         fitness_function=fitness_function,
                         selection_exponent=selection_exponent,
                         archive_bin_size=archive_bin_size,
                         archive_offspring_per_gen=archive_offspring_per_gen,
                         offspring_per_gen=offspring_per_gen,
                         num_new_pop=num_new_pop,
                         gen_per_growth=gen_per_growth,
                         smart_seeding=smart_seeding,
                         alg_name="ccea",
                         run_name=run_name,
                         seed=seed)

        self.p_wildcard = p_wildcard
        self.feature_selection_ratio = feature_selection_ratio
        self.fitness_threshold = np.log10(fitness_threshold)
        self.feature_names = feature_names
        self.selective_mutation = selective_mutation

    def _choose_seed_simple(self, valid_observations: np.ndarray):
        """ Chooses an observation to use for seeding a new clause by using simple uniform random choice

        :param valid_observations: A list of observations that can be chosen

        :return: The chosen seed observation
        """
        # Select a random observation from the valid observations
        observations = self._fit_data.observations[valid_observations]

        if observations.shape[0] == 0:
            return None

        seed_observation = teva_math.Rand.RNG.choice(observations)

        return seed_observation

    def _choose_seed_influenced(self,
                                valid_observations: np.ndarray,
                                selection_exponent: float = 3.0):
        """ Chooses an observation to use for seeding a new clause by using the archived clauses to influence the choice

        :param valid_observations: A list of observations that can be chosen
        :param selection_exponent: An exponent that will be applied to the difference sums during selection

        :return: The chosen seed observation
        """
        observations = self._fit_data.observations[valid_observations]

        all_values = np.array(self.get_all_archive_values())

        tallies = np.vstack(
            [obs.coverage_mask & valid_observations for obs in all_values]
        ).sum(axis=0)[valid_observations]

        diff_sum = (np.max(tallies) + 1) - tallies
        diff_exp = diff_sum ** selection_exponent
        probs = diff_exp / np.sum(diff_exp)

        # NOTE: This may be incorrect; ensure that it gives the same results
        seed_observation = teva_math.Rand.RNG.choice(observations, p=probs)

        return seed_observation

    def _init_clauses(self, target_class, target_observations: np.ndarray, non_null_counts: np.ndarray):
        """Fill the first age layer with novel clauses (clauses created directly from the
        observation table)

        Novel CCs are randomly created for layer 1 to guarantee they match at least one
        input feature vector associated with the target class :math:`k`, a process known
        as "covering". To accomplish this, we first generate a uniformly distributed
        random integer :math:`j \\in \\{1,\\ldots,L\\}` (where :math:`L` is the number of
        features in each input vector) to specify the order of the CC, and then extract
        the subset of input feature vectory with class :math:`k` having at least this
        many non-missing values. From this subset, we choose one at random. While the
        archive is empty, this random input feature vector is selected according to a
        uniform distribution. Once the archive has been populated with CCs, we use a
        non-uniform distribution to bias the selection toward input feature vectors not
        yet well-covered in the archive. Specifically, we first tally the archived
        clauses that match each input feature vector in the extracted subset. We then sum
        this tally, add one, subtract each feature vector's tally from this value, and
        cube the result (cubing increases the probability that under-represented input
        feature vectors will be selected). We normalize the resulting vector and treat
        this as the probability distribution, then select :math:`j` of the non-missing
        features from the selected feature vector to present in the new clause according
        to this distribution. For each selected feature :math:`i`, we initialize
        :math:`a_i` to contain only the value for feature :math:`i` that occurs in the
        selected input feature vector.

        :param target_class: The target class of this run
        """
        new_clauses = []
        # insert novel clauses into the first age layer
        for _ in range(self.num_new_pop):
            # generate j, the order of the clause
            clause_order = teva_math.Rand.RNG.integers(low=1, high=self.max_order + 1)
            valid_mask = np.array(non_null_counts >= clause_order)

            # Check if there is an archive population.
            # If there is, use it to bias the selection of the seed observation towards
            # under-represented observations in the archive. Otherwise, select a random
            # observation from the valid observations

            if self.smart_seeding and len(self.archive.values()) > 0:
                seed_observation = self._choose_seed_influenced(target_observations & valid_mask)
            else:
                seed_observation = self._choose_seed_simple(target_observations & valid_mask)

            if seed_observation is None:
                continue

            # initialize a clause from the selected seed observation
            clause = conjunctive_clause.ConjunctiveClause.init_clause(observation=seed_observation,
                                                                      classification=target_class,
                                                                      clause_order=clause_order,
                                                                      feature_domains=self._fit_data.feature_domains,
                                                                      rng=teva_math.Rand.RNG)
            clause.calc_coverage(observation_table=self._fit_data.observations, classifications=self._fit_data.classes)
            new_clauses.append(clause)

        self._calc_fitnesses(new_clauses)
        for clause in new_clauses:
            self._add_to_population(clause)

    def _calc_fitnesses(self, children: list[conjunctive_clause.ConjunctiveClause]):
        """ Determines the fate of each new clause in a list by calculating its fitness values

        :param children: The list of clauses to calculate fitnesses on
        """
        child_fates: list[str] = fitness.children_fitness(children=children,
                                                          fitness_function=self.fitness_function,
                                                          fitness_thresholds=self.order_thresholds,
                                                          max_order=self.max_order,
                                                          use_sensitivity=self.use_sensitivity,
                                                          sensitivity_threshold=self.sensitivity_threshold)

        logging.getLogger("ccea").debug("Child Fates: ")
        logging.getLogger("ccea").debug(f"Archived: {np.sum(child_fates == 'archive')}")
        logging.getLogger("ccea").debug(f"Kept: {np.sum(child_fates == 'keep')}")
        logging.getLogger("ccea").debug(f"Removed: {np.sum(child_fates == 'remove')}")

        # handle the fate of each child
        for i, fate in enumerate(child_fates):
            # children[i].calc_coverage(observation_table=self._fit_data.observations,
            #                           classifications=self._fit_data.classes)
            if fate == "archive":
                self._archive_clause(children[i])
            elif fate == "keep":
                self._add_to_population(children[i])

    def _grow_clauses(self):
        """ Increase ages of all populations, archiving or removing the ones in the highest age layer based on their
        fitness.
        """
        # shift each layer up by one
        for layer_index in range(len(self.age_layers) - 1, 0, -1):
            # copy the layer below to the current layer
            self.age_layers[layer_index] = copy.deepcopy(self.age_layers[layer_index - 1])
            # change the clauses now in the current layer to have the same age as the current layer
            for clause in self.age_layers[layer_index]:
                clause.age = layer_index

        # clear the first age layer
        self.age_layers[0].clear()

    def _reduce_populations(self):
        """Handles the reduction of archive and layer populations, as well as the reintroduction of removed archive
        clauses.  The algorithm follows these basic steps:

        1. Remove excess clauses from each archive bin (more than ``self.archive_bin_size``)
        2. Re-introduce young removed archive clauses back into their appropriate age layers
        3. Remove excess clauses from each age layer (more than ``self.layer_size``)
        """
        # reduce population of archive

        archive_removed = []
        for order in sorted(self.archive.keys()):
            # remove duplicate clauses in the bin using set hashing
            order_array = np.array(self.archive[order])
            hashed = np.vectorize(hash)(order_array)
            _, unique_index = np.unique(hashed, return_index=True)
            duplicates_removed = order_array[unique_index].tolist()
            self.archive[order] = duplicates_removed

            # truncate the bin to the maximum bin length
            # NOTE: There are a lot of things being done here in the matlab code that seem unnecessary,
            #       but it should be reassessed
            bin_length = len(self.archive[order])
            if bin_length > self.archive_bin_size:
                # save the removed clauses to the archive removed list
                archive_removed.extend(self.archive[order][self.archive_bin_size:])

            # keep only the clauses that fit in the bin
            self.archive[order] = self.archive[order][: self.archive_bin_size]

            # update the order threshold to be the minimum fitness of this order
            if len(self.archive[order]) >= self.archive_bin_size:
                self.order_thresholds[order - 1] = self.archive[order][-1].fitness

        total_archive_removed = len(archive_removed)

        logging.getLogger("ccea").debug(f"Removed from archive: {total_archive_removed}")

        # if any of the removed clauses are young enough to be in the population, add them back into the population
        # NOTE: I don't believe this is mentioned anywhere in the paper, but we should double-check
        for clause in archive_removed:
            if clause.age < self.n_age_layers:
                self._add_to_population(clause)

        # initialized removed layers list to be an empty list of `n_age_layers` empty lists
        layers_removed: [[conjunctive_clause.ConjunctiveClause]] = [[]] * self.n_age_layers
        # handle habitat overcrowding by truncating each age layer to the maximum layer size
        for layer_index in range(self.n_age_layers):
            layer_length = len(self.age_layers[layer_index])
            if layer_length > self.layer_size:
                # save to `layers_removed` for bookkeeping
                layers_removed[layer_index] = self.age_layers[layer_index][self.layer_size:]
            # truncate the bin
            self.age_layers[layer_index] = self.age_layers[layer_index][: self.layer_size]

        logging.getLogger("ccea").debug(f"Truncated Per Layer: {[len(layer) for layer in layers_removed]}")

    def _evolve_layer(self, layer):
        """ Evolves every clause in a given layer, either by mutating or crossing each with a specified probability.
        Then, determines and executes the fate of each child.

        :param layer: The list of DNFs to evolve. This is intended to be either an age layer or an archive bin.
        """
        # if layer is empty, there is nothing to evolve
        if len(layer) == 0:
            return

        children: list[conjunctive_clause.ConjunctiveClause] = evolution.crossover_or_mutate(
                                                                layer,
                                                                observation_table=self._fit_data.observations,
                                                                classifications=self._fit_data.classes,
                                                                num_clauses=self.offspring_per_gen,
                                                                p_x=self.p_crossover,
                                                                p_m=self.p_mutation,
                                                                p_wc=self.p_wildcard,
                                                                tournament_size=self.tournament_size,
                                                                fitness_function=self.fitness_function,
                                                                selective_mutation=self.selective_mutation,
                                                                feature_selection_ratio=self.feature_selection_ratio)

        for child in children:
            child.calc_coverage(observation_table=self._fit_data.observations,
                                classifications=self._fit_data.classes)

        self._calc_fitnesses(children)

    def fit(self,
            observation_table: np.ndarray | pd.DataFrame,
            classifications: np.ndarray | pd.Series,
            feature_types: list[feature.FeatureType] = None,
            conjunctive_clauses: list | np.ndarray = None) -> np.ndarray:
        """ Prepare this CCEA algorithm to run on data from the observation array.  This function must be run before
        you can call :function:`CCEA.run()` or :function:`CCEA.run_all_targets()`

        :param observation_table: A 2D array where each column represents a feature, and each row represents an
            observation
        :param classifications: A 1D array of classification values where each value corresponds to an observation in
            the observation array
        :param feature_types: A list of the feature type of each feature column.  If undefined, the algorithm will try
            to determine automatically
        :param conjunctive_clauses: NOT USED

        :return: An array of unique classifications, which can be useful if running the algorithm on each class
            separately.
        """

        self.reset()

        self._fit_data = CCEAData()

        if isinstance(observation_table, pd.DataFrame):
            self._fit_data.observations = observation_table.to_numpy()
        else:
            self._fit_data.observations = observation_table

        if isinstance(classifications, pd.Series):
            self._fit_data.classes = classifications.to_numpy()
        else:
            self._fit_data.classes = classifications

        # if feature types are not passed as an argument, determine them automatically
        if feature_types is None:
            feature_types = feature.determine_feature_types(observation_table)

        self._fit_data.feature_domains = feature.find_feature_domains(observation_table, feature_types)
        self._fit_data.unique_classes = np.unique(classifications)

        if self.fitness_threshold is None:
            self.fitness_threshold = np.log10(1 / classifications.shape[0])

        self.order_thresholds = [self.fitness_threshold for _ in range(self.max_order)]

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(0, len(self._fit_data.feature_domains))]

        return self._fit_data.unique_classes
