""" The Disjunctive Normal Form Evolutionary Algorithm """
from typing import Callable
import logging
import copy

import datetime
import numpy as np
import pandas as pd
from numpy.random import Generator, MT19937

from teva.utilities import hyge, teva_math
from teva.base import feature
from teva.ccea import conjunctive_clause
from teva.dnfea import disjunctive_clause, evolution, fitness
from teva.base.evolutionary_algorithm import EvolutionaryAlgorithm, EvolutionaryAlgorithmError, EvolutionaryData

class DNFEAError(EvolutionaryAlgorithmError):
    """ Basic error for the DNFEA """

class DNFEAData(EvolutionaryData):
    """ DNFEA run data struct """

class DNFEA(EvolutionaryAlgorithm):
    """ Disjunctive Normal Form Evolutionary Algorithm

    Generates DNF (Disjunctive Normal Form) clauses that represent a disjunction of previously generated Conjunctive
    Clauses.

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

    :param p_targeted_mutation: Probability of a bit flip mutation
    :param p_targeted_crossover: Probability that a mate is selected on fitness only
    :param p_union: Probability of a union crossover
    :param p_intersection: Probability of an intersection crossover
    :param p_crossover_algorithm: Probability cumulative sum for alternative crossover mate selection
    :param p_mutation_algorithm: Probability cumulative sum for alternative targeted mutation
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
                 use_sensitivity: bool = False,
                 sensitivity_threshold: float = 0,
                 fitness_function: Callable = hyge.hygepmf,
                 seed: int = None,
                 run_name: str = None,
                 smart_seeding: bool = True,  # winit

                 # DNFEA Specific Inputs
                 p_targeted_mutation: float = 0.8,  # Pbf
                 p_targeted_crossover: float = 0.75,  # Pxf
                 p_union: float = 0.5,  # PrUn
                 p_intersection: float = 0.5,  # PrIn
                 p_crossover_algorithm: list[float] = None,  # PxAlt (must sum to 1)
                 p_mutation_algorithm: list[float] = None):
        """ Constructor """
        # calculate non-standard defaults
        if p_crossover_algorithm is None:
            p_crossover_algorithm = [1.0 / 3.0 for _ in range(3)]

        if p_mutation_algorithm is None:
            p_mutation_algorithm = [1.0 / 4.0 for _ in range(4)]

        if run_name == "":
            run_name = f"dnfea_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        super().__init__(total_generations=total_generations,
                         layer_size=layer_size,
                         n_age_layers=n_age_layers,
                         gen_per_growth=gen_per_growth,
                         p_crossover=p_crossover,
                         p_mutation=p_mutation,
                         max_order=max_order,
                         tournament_size=tournament_size,
                         num_new_pop=num_new_pop,
                         fitness_function=fitness_function,
                         selection_exponent=selection_exponent,
                         offspring_per_gen=offspring_per_gen,
                         archive_bin_size=archive_bin_size,
                         archive_offspring_per_gen=archive_offspring_per_gen,
                         use_sensitivity=use_sensitivity,
                         sensitivity_threshold=sensitivity_threshold,
                         smart_seeding=smart_seeding,
                         alg_name="dnfea",
                         run_name=run_name,
                         seed=seed)

        # set members
        self.p_mutation = p_mutation
        self.p_targeted_mutation = p_targeted_mutation
        self.p_targeted_crossover = p_targeted_crossover
        self.p_union = p_union
        self.p_intersection = p_intersection
        self.p_x_alternate = p_crossover_algorithm
        self.p_m_alternate = p_mutation_algorithm
        self.smart_seeding = smart_seeding

        self.current_gen = 0

        self.order_thresholds: list[float] = []

        # get the default logger
        # logging.getLogger("dnfea") = logging.getLogger("dnfea")

    def _init_clauses(self,
                      target_class,
                      target_observations: np.ndarray = None,
                      non_null_counts: np.ndarray = None):
        """ Creates a specified number of new DNF clauses from the input CCs

        :param target_class: The current target class of this run.
        """
        new_clauses = []

        total_ccs = len(self._fit_data.conjunctive_clauses)
        for _ in range(self.num_new_pop):
            # generate j, the order of the clause
            dnf_order = teva_math.Rand.RNG.integers(low=1, high=self.max_order + 1)

            # Check if there is an archive population.
            # If there is, use it to bias the selection of the seed observation towards
            # under-represented clauses in the archive. Otherwise, select a random
            # set of clauses from the valid observations
            seed_indices: np.ndarray

            pool_size = min(self.max_order * 2, total_ccs)

            archive_contains_higher_order_dnfs = False
            for key, val in self.archive.items():
                if key > 1 and len(val) > 0:
                    archive_contains_higher_order_dnfs = True
                    break

            if self.smart_seeding and archive_contains_higher_order_dnfs:
                # Calculate a new probability of selecting a CC for the template of a new DNF. Increase odds of
                # selecting a CC that covers underrepresented target observations in the archive.

                dnfs = self.get_all_archive_values()
                # Count the total number of times that a target observation is covered in the archive population
                covered_counts = np.sum(np.vstack([clause.target_coverage_mask for clause in dnfs]), axis=0)
                covered_counts_max = np.max(covered_counts)
                # Subtract covered_counts from the max, so that the least represented target observations have a higher
                # value
                diff_sum = (covered_counts_max + 1) - covered_counts
                # bias toward underrepresented observations
                diff_sum = diff_sum ** self.selection_exponent
                # Normalize to get probabilities
                observation_probabilities = diff_sum / np.sum(diff_sum)

                # Get the target coverage masks for all the available CCs
                cc_target_coverage_masks = [cc.target_coverage_mask for cc in self._fit_data.conjunctive_clauses]

                # Apply the probabilities to the CCs
                # We first apply the appropriate probability to each covered target observation
                probabilities_applied_to_ccs = np.sum(np.vstack(cc_target_coverage_masks) * observation_probabilities,
                                                      axis=1)
                # Then we normalize by the number of target observations covered by each CC
                normalized_probabilities_applied_to_ccs = (
                        probabilities_applied_to_ccs * np.sum(np.vstack(cc_target_coverage_masks), axis=1)
                ) / np.sum((probabilities_applied_to_ccs * np.sum(np.vstack(cc_target_coverage_masks), axis=1)))

                ccs_with_nonzero_probabilities_mask = normalized_probabilities_applied_to_ccs > 0

                positive_only_probabilities = normalized_probabilities_applied_to_ccs
                positive_only_probabilities[~ccs_with_nonzero_probabilities_mask] = 0
                positive_only_probabilities = positive_only_probabilities / np.sum(positive_only_probabilities)

                num_nonzero_probabilities = np.sum(ccs_with_nonzero_probabilities_mask)

                if num_nonzero_probabilities >= pool_size:
                    seed_indices = teva_math.Rand.RNG.choice(np.arange(total_ccs),
                                                             size=pool_size,
                                                             p=positive_only_probabilities,
                                                             replace=False)
                else:
                    seed_indices = teva_math.Rand.RNG.choice(np.arange(total_ccs),
                                                             size=num_nonzero_probabilities,
                                                             p=positive_only_probabilities,
                                                             replace=False)

                    # Since we don't have enough CCs with a probability > 0, choose from the remaining CCs to get enough
                    ccs_with_zero_probability_indices = np.arange(total_ccs)[~ccs_with_nonzero_probabilities_mask]
                    if ccs_with_zero_probability_indices.shape[0] == 0:
                        print()
                    additional_cc_indices = teva_math.Rand.RNG.choice(ccs_with_zero_probability_indices,
                                                             size=pool_size - num_nonzero_probabilities,
                                                             replace=False)

                    seed_indices = np.concatenate([seed_indices, additional_cc_indices])
            else:
                seed_indices = teva_math.Rand.RNG.choice(np.arange(total_ccs), size=total_ccs, replace=False)


            # reduce the seed indices array to the dnf order size
            seed_indices = teva_math.Rand.RNG.choice(seed_indices,
                                                     size=min(dnf_order, seed_indices.shape[0]),
                                                     replace=False)

            # create cc mask and enable the seed indices
            mask = np.zeros(total_ccs)
            mask[seed_indices] = True

            new_dnf = disjunctive_clause.DisjunctiveClause(cc_mask=mask,
                                                           cc_clauses=self._fit_data.conjunctive_clauses,
                                                           classification=target_class)

            new_dnf.creation_gen = self.current_gen
            new_dnf.calc_coverage(observation_table=self._fit_data.observations)
            new_clauses.append(new_dnf)

        self._calc_fitnesses(new_clauses)
        for clause in new_clauses:
            self._add_to_population(clause)

        logging.getLogger("dnfea").debug(f"Created {self.num_new_pop} new clauses")

    def _calc_fitnesses(self, children: list[disjunctive_clause.DisjunctiveClause]):
        """ Determines the fate of each new clause in a list by calculating its fitness values

        :param children: The list of clauses to calculate fitnesses on
        """
        child_fates: list[str] = fitness.children_fitness(children=children,
                                                          fitness_function=self.fitness_function,
                                                          fitness_thresholds=self.order_thresholds,
                                                          max_order=self.max_order,
                                                          use_sensitivity=self.use_sensitivity,
                                                          sensitivity_threshold=self.sensitivity_threshold)

        logging.getLogger("dnfea").debug("Child Fates: ")
        logging.getLogger("dnfea").debug(f"Archived: {child_fates.count('archive')}")
        logging.getLogger("dnfea").debug(f"Kept: {child_fates.count('keep')}")
        logging.getLogger("dnfea").debug(f"Removed: {child_fates.count('remove')}")

        # handle the fate of each child
        for i, fate in enumerate(child_fates):
            child = children[i]
            child.creation_gen = self.current_gen
            if fate == "archive":
                self._archive_clause(child)
            elif fate == "keep":
                self._add_to_population(child)

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
        for order, order_bin in self.archive.items():
            # remove duplicate clauses in the bin
            order_array = np.array(order_bin)
            hashed = np.vectorize(hash)(order_array)
            _, unique_index = np.unique(hashed, return_index=True)
            duplicates_removed = order_array[unique_index].tolist()
            self.archive[order] = duplicates_removed

            # NOTE: The bin is already sorted
            # truncate the bin to the maximum bin length
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

        logging.getLogger("dnfea").debug(f"Removed from archive: {total_archive_removed}")

        # if any of the removed clauses are young enough to be in the population, add them back into the population
        # NOTE: I don't believe this is mentioned anywhere in the paper, but we should double-check
        archive_remove_copy = archive_removed.copy()
        for clause in archive_removed:
            if clause.age < self.n_age_layers:
                self._add_to_population(clause)
                # if it is re-inserted into the population, remove it from the removed list
                archive_remove_copy.remove(clause)

        logging.getLogger("dnfea").debug(
            f"Reintroduced: {total_archive_removed - len(archive_removed)}"
        )

        # initialzed removed layers list to be an empty list of `n_age_layers` empty lists
        layers_removed: [[conjunctive_clause.ConjunctiveClause]] = [[]] * self.n_age_layers
        # handle habitat overcrowding by truncating each age layer to the maximum layer size
        for layer_index in range(self.n_age_layers):
            layer_length = len(self.age_layers[layer_index])

            # NOTE: The layer is already sorted

            if layer_length > self.layer_size:
                # save to `layers_removed` for bookkeeping
                layers_removed[layer_index] = self.age_layers[layer_index][self.layer_size:]

            # remove the clauses
            self.age_layers[layer_index] = self.age_layers[layer_index][: self.layer_size]

        logging.getLogger("dnfea").debug(
            f"Truncated Per Layer: {[len(layer) for layer in layers_removed]}"
        )

    def _evolve_layer(self, layer: list[disjunctive_clause.DisjunctiveClause]):
        """ Evolves every clause in a given layer, either by mutating or crossing each with a specified probability.
        Then, determines and executes the fate of each child.

        :param layer: The list of DNFs to evolve. This is intended to be either an age layer or an archive bin.
        """
        children = evolution.crossover_or_mutate(population_layer=layer,
                                                 observation_table=self._fit_data.observations,
                                                 num_dnfs=self.num_new_pop,
                                                 p_x=self.p_crossover,
                                                 p_xf=self.p_targeted_crossover,
                                                 p_m=self.p_mutation,
                                                 p_m_alt=self.p_m_alternate,
                                                 p_x_alt=self.p_x_alternate,
                                                 p_f=self.p_targeted_mutation,
                                                 p_i=self.p_intersection,
                                                 p_u=self.p_union,
                                                 tournament_size=self.tournament_size)

        self._calc_fitnesses(children)

    def fit(self,
            observation_table: np.ndarray | pd.DataFrame,
            classifications: np.ndarray | pd.Series,
            feature_types: list[feature.FeatureType] = None,
            conjunctive_clauses: list | np.ndarray = None,
            use_cc_best_fitness: bool = False) -> np.ndarray:
        """ Prepare this DNFEA algorithm to run on data from the observation array.  This function must be run before
        you can call :function:`DNFEA.run()` or :function:`DNFEA.run_all_targets()`

        :param observation_table: A 2D array where each column represents a feature, and each row represents an
            observation
        :param classifications: A 1D array of classification values where each value corresponds to an observation in
            the observation array
        :param conjunctive_clauses: A 1D array of conjunctive clauses, likely generated by :class:`teva.ccea.CCEA`
        :param feature_types: NOT USED
        :param use_cc_best_fitness: If True, the best cc fitness will be used as a baseline for DNF.  Otherwise it will
            use the best

        :return: An array of unique classifications, which can be useful if running the algorithm on each class
            separately.
        """
        self.reset()

        if isinstance(conjunctive_clauses, list) and len(conjunctive_clauses) == 0 or \
            isinstance(conjunctive_clauses, np.ndarray) and conjunctive_clauses.shape[0] == 0:
            raise DNFEAError("The conjunctive clause array was empty; cannot run DNFEA")

        fit_data = DNFEAData()

        if isinstance(observation_table, pd.DataFrame):
            fit_data.observations = observation_table.to_numpy()
        else:
            fit_data.observations = observation_table

        if isinstance(classifications, pd.Series):
            fit_data.classes = classifications.to_numpy()
        else:
            fit_data.classes = classifications

        if isinstance(conjunctive_clauses, list):
            conjunctive_clauses = np.array(conjunctive_clauses)

        fit_data.conjunctive_clauses = conjunctive_clauses
        fit_data.unique_classes = np.unique(classifications)

        max_fitness = -np.inf
        min_fitness = np.inf
        for clause in conjunctive_clauses:
            max_fitness = max(max_fitness, clause.fitness)
            min_fitness = min(min_fitness, clause.fitness)

        if use_cc_best_fitness:
            fit_data.fitness_threshold = min_fitness
        else:
            fit_data.fitness_threshold = max_fitness

        self.order_thresholds = [fit_data.fitness_threshold for _ in fit_data.conjunctive_clauses]
        self._fit_data = fit_data

        return fit_data.unique_classes
