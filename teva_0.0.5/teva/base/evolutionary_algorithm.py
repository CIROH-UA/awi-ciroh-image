""" Base classes for CCEA and DNFEA """
import os
import abc
import sys
import logging
from abc import abstractmethod
from typing import Callable, Literal

import numpy as np
from termcolor import cprint

from teva.utilities import teva_logging, teva_math
from teva.base.feature import FeatureType, FeatureDomain
from teva.base.clause import Clause

class EvolutionaryAlgorithmError(Exception):
    """ Exception related to the Evolutionary Algorithm """

class EvolutionaryData:
    """ A data structure for storing Evolutionary Algorithm fit data """
    def __init__(self):
        self.observations: np.ndarray | None = None
        self.classes: np.ndarray | None = None
        self.feature_domains: list[FeatureDomain] | None = None
        self.unique_classes: np.ndarray | None = None
        self.conjunctive_clauses: np.ndarray | None = None
        self.fitness_threshold: float | None = None

class EvolutionaryAlgorithm(abc.ABC):
    """ Base class for Evolutionary Algorithms like CCEA and DNFEA

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
    :param alg_name: The name of the algorithm that is instantiating this base class.  Used for logging
    :param smart_seeding: If True, seeding for new DNFs will take the current archive into account to ensure a greater
        diversity of clauses
    """
    def __init__(self,
                 total_generations: int,
                 layer_size: int,
                 max_order: int,
                 n_age_layers: int,
                 gen_per_growth: int,
                 offspring_per_gen: int,
                 archive_offspring_per_gen: int,
                 use_sensitivity: bool,
                 sensitivity_threshold: float,
                 archive_bin_size: int,
                 num_new_pop: int,
                 p_crossover: float,
                 p_mutation: float,
                 tournament_size: int,
                 selection_exponent: float,
                 smart_seeding: bool,
                 alg_name: str,
                 run_name: str,
                 fitness_function: Callable,
                 seed: int = None):

        if offspring_per_gen is None:
            offspring_per_gen = total_generations

        if archive_offspring_per_gen is None:
            archive_offspring_per_gen = n_age_layers * offspring_per_gen

        if archive_bin_size is None:
            archive_bin_size = offspring_per_gen

        if layer_size is None:
            layer_size = offspring_per_gen

        if num_new_pop is None:
            num_new_pop = layer_size

        self.tournament_size = tournament_size
        self.total_generations: int = total_generations
        self.layer_size: int = layer_size
        self.n_age_layers: int = n_age_layers
        self.gen_per_growth: int = gen_per_growth
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.max_order = max_order

        self.offspring_per_gen = offspring_per_gen
        self.archive_offspring_per_gen = archive_offspring_per_gen
        self.num_new_pop = num_new_pop
        self.archive_bin_size = archive_bin_size

        self.selection_exponent = selection_exponent
        self.fitness_function = fitness_function
        self.use_sensitivity = use_sensitivity
        self.sensitivity_threshold = np.log10(sensitivity_threshold)

        self.smart_seeding = smart_seeding
        self.run_name = run_name
        self.alg_name = alg_name

        # the index of the age_layers represents the age of that layer
        self.age_layers: [[Clause]] = [[] for _ in range(self.n_age_layers)]
        # the key of the archive/staging layer bins represents the order of that bin
        self.archive: [int, list[Clause]] = {}
        # define data fit object
        self._fit_data: EvolutionaryData | None = None
        self.order_thresholds: list[float] = []

        self.seed = seed

        if seed is not None:
            teva_math.Rand.set_seed(seed)

    def _start_logger(self,
                      output_logging_level: int = logging.INFO,
                      logfile_logging_level: int = logging.INFO):
        """ Starts the logger for this run

        :param output_logging_level: The logging level of the STDOUT reporting stream
        :param logfile_logging_level: The logging level of the Logfile reporting stream
        """
        teva_logging.setup_logger(
            self.alg_name,
            logger_file=f"{self.run_name}",
            output_logging_level=output_logging_level,
            logfile_logging_level=logfile_logging_level,
        )

    def _visualize(self, generation, timers=None):
        """ Writes live graphical ascii output to the console to help visualize what is happening
        behind the scenes

        :param generation: The current generation that the algorithm is on
        :param timers: A dictionary of timer data to be displayed {"name": value}
        """
        if os.name == 'nt':
            _ = os.system('cls')  # For Windows
        else:
            _ = os.system('clear')

        age_colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "light_red", "light_green",
                      "light_yellow", "light_blue", "light_magenta", "light_cyan", "white"]

        cprint(f"GENERATION {generation}/{self.total_generations}", color="white", attrs=["bold"])
        cprint("Age Layers", color="white", attrs=["underline"])
        for i, age_layer in enumerate(self.age_layers):
            n_clauses = len(age_layer)
            max_clauses = self.layer_size

            cprint(f"{i}: ", color="white", end='')

            for clause in age_layer:
                cprint(str(clause.order()), color=age_colors[i], end='')
            # print(colored("|" * n_clauses, color="green"), end='')

            remaining = max(max_clauses - n_clauses, 0)
            cprint("-" * remaining, color="white", end='\n')

        print()

        cprint("Archive Bins", color="white", attrs=["underline"])
        for i in sorted(self.archive.keys()):
            order_bin = self.archive[i]
            n_clauses = len(order_bin)

            cprint(f"{i}: ", color="white", end='')

            for clause in order_bin:
                if clause.age < len(age_colors):
                    cprint(str(clause.age), color=age_colors[clause.age], end='')
                else:
                    cprint(str(clause.age), color="white", end='')

            # cprint("|" * n_clauses, color="green", end='\n')
            print()

        if timers is None:
            timers = {}

        for key in timers.keys():
            print(f"{key}: {int(timers[key] * 1000)}", end="\n")

        print()
        sys.stdout.flush()

    def _add_to_population(self, clause: Clause):
        """ Adds a CC to the CCEA population, using its age to determine which layer to add it to

        :param clause: The clause that should be added to the population
        """
        self._add_to_layer(clause, clause.age)

    def _add_to_layer(self, clause: Clause, age: int):
        """ Adds a CC to a specific age layer in the CCEA population

        :param clause: The clause that should be added to the population
        :param age: The age of the bin in which to add this clause
        """
        self.age_layers[age].append(clause)
        self.age_layers[age] = self._sort_layer(self.age_layers[age])

    def _archive_clause(self, clause: Clause):
        """ Adds a CC to the archive bin associated with its order.  New bins will be created if necessary

        :param clause: The clause that should be added to an archive bin
        """
        clause_order = clause.order()
        # make sure to initialize any bins that are not yet used
        if clause_order not in self.archive:
            self.archive[clause_order] = [clause]
        else:
            self.archive[clause_order].append(clause)

        self.archive[clause_order] = self._sort_layer(self.archive[clause_order])

    @staticmethod
    def _sort_layer(layer: list[Clause],
                    attribute: Literal["fitness", "age"] = "fitness") -> list[Clause]:
        """ Sorts a list of clauses by the given attribute

        :param layer: A list of conjunctive clauses to sort
        :param attribute: The attribute which should be used for sorting

        :return: The sorted list of clauses
        """
        if attribute == "fitness":
            return sorted(layer, key=lambda clause: clause.fitness)
        if attribute == "age":
            return sorted(layer, key=lambda clause: clause.age)

        raise ValueError("Attribute has an invalid value")

    @abc.abstractmethod
    def _init_clauses(self, target_class, target_observations: np.ndarray, non_null_counts: np.ndarray):
        return NotImplementedError

    @abc.abstractmethod
    def _calc_fitnesses(self, children: list[Clause]):
        return NotImplementedError

    @abc.abstractmethod
    def _grow_clauses(self):
        return NotImplementedError

    @abc.abstractmethod
    def _reduce_populations(self):
        return NotImplementedError

    @abc.abstractmethod
    def _evolve_layer(self, layer):
        return NotImplementedError

    def reset(self, reset_archive: bool = True):
        """ Resets the algorithm to a run-ready state

        :param reset_archive: If True, the archive will be reset as well
        """
        self.age_layers = [[] for _ in range(self.n_age_layers)]
        self._fit_data = None
        if reset_archive:
            self.archive = {}

    def get_archived(self, target_class=None) -> dict:
        """ Gets all archived clauses

        :param target_class: The specific target class of clauses to be returned, if None, returns all

        :return: A dictionary of all clauses in the archive
        """
        if target_class is None:
            return self.archive

        new_archive = {}
        for key, val in self.archive.items():
            new_archive[key] = [clause for clause in val if clause.classification == target_class]
        return new_archive

    def get_all_archive_values(self) -> list[Clause]:
        """ Returns all the archived clauses in a list

        :return: A list of every clause in the archive
        """
        all_values = []
        for _, val in self.archive.items():
            all_values.extend(val)
        return all_values

    def run_all_targets(self,
                        output_logging_level: int = logging.INFO,
                        logfile_logging_level: int = logging.INFO,
                        visualize: bool = False):
        """ Run a fitted model on every target class.  To run only one target class at a time,
        see :function:`run()`

        .. note:: Requires :function:`fit()` to have been called before the algorithm can run.

        :param output_logging_level: The logging level of the STDOUT reporting stream
        :param logfile_logging_level: The logging level of the Logfile reporting stream
        :param visualize: Enables console visualization
        """

        if self._fit_data is None:
            raise EvolutionaryAlgorithmError("You must run fit() before calling 'run_all_targets()")

        # TODO: Do some parallelization
        for cls in self._fit_data.unique_classes:
            self.run(cls, output_logging_level, logfile_logging_level, visualize=visualize, reset_on_complete=False)

    @abstractmethod
    def fit(self,
            observation_table: np.ndarray,
            classifications: np.ndarray,
            feature_types: list[FeatureType] = None,
            conjunctive_clauses: list | np.ndarray = None) -> np.ndarray:
        """ Prepare this algorithm to run on data from the observation array.  This function must be run before
        you can call :function:`run()` or :function:`CCEA.run_all_targets()`

        :param observation_table: A 2D array where each column represents a feature, and each row represents an
            observation
        :param classifications: A 1D array of classification values where each value corresponds to an observation in
            the observation array
        :param feature_types: Used by CCEA but not DNFEA
        :param conjunctive_clauses: Used by DNFEA, but not CCEA.

        :return: An array of unique classifications, which can be useful if running the algorithm on each class
            separately.
        """
        raise NotImplementedError

    def run(self,
            target_class,
            output_logging_level: int = logging.INFO,
            logfile_logging_level: int = logging.INFO,
            visualize: bool = False,
            reset_on_complete: bool = True):
        """ Run a fitted model on a specific target class.  To run more than one target class at the same
        time, see :function:`run_all_targets()`

        .. note:: Requires :function:`fit()` to have been called before the algorithm can run.

        :param target_class: The target class to focus this run of the algorithm on
        :param output_logging_level: The logging level of the STDOUT reporting stream
        :param logfile_logging_level: The logging level of the Logfile reporting stream
        :param visualize: Enables console visualization
        :param reset_on_complete: If True, the archive will be reset when the run is completed
        """
        if visualize:
            output_logging_level = logging.ERROR

        if self._fit_data is None:
            raise EvolutionaryAlgorithmError("You must run fit() before calling 'run()")

        # start logging
        self._start_logger(output_logging_level=output_logging_level,
                           logfile_logging_level=logfile_logging_level)
        logging.getLogger(self.alg_name).info("Beginning CCEA run with the following parameters:")
        logging.getLogger(self.alg_name).info(f"Features: {self._fit_data.observations.shape[1]}")
        logging.getLogger(self.alg_name).info(f"Target Class: {target_class}")

        logging.getLogger(self.alg_name).debug(f"T: {target_class} | Generation: {0} / {self.total_generations} : INIT")

        # extract the observations that are of the target class
        target_mask = np.array(self._fit_data.classes == target_class)

        # create a mask of non-null values
        non_null_mask = ~np.isnan(self._fit_data.observations)
        # count the number of non-null values in each observation
        non_null_counts = np.sum(non_null_mask, axis=1)

        self._init_clauses(target_class, target_mask, non_null_counts)

        if visualize:
            self._visualize(0)

        for gen in range(1, self.total_generations):
            is_growth_gen = gen % self.gen_per_growth == 0

            # debug print the current generation, specifying whether it is a growth gen
            gen_string = f"T: {target_class} | Generation: {gen} / {self.total_generations}"
            if is_growth_gen:
                gen_string += " : GROWTH"

            logging.getLogger(self.alg_name).info(gen_string)

            # evolve all normal layers
            logging.getLogger(self.alg_name).debug("Evolve All Layers:")
            for layer in self.age_layers:
                self._evolve_layer(layer)

            # get all values as a 1D list
            all_values = self.get_all_archive_values()

            # sort all clauses in the archive by ascending age
            archive_youngest: [Clause] = sorted(all_values, key=lambda x: x.age)

            # get layer_size * 5 youngest archive clauses
            archive_youngest = archive_youngest[: min(len(archive_youngest), self.layer_size * 5)]

            # evolve the archive youngest
            self._evolve_layer(archive_youngest)

            # if it s a growth gen, grow (age up) current clauses and initialize new ones to refill layer 0
            if is_growth_gen:
                self._grow_clauses()
                self._init_clauses(target_class, target_mask, non_null_counts)

            self._reduce_populations()

            if visualize:
                self._visualize(gen)

        if reset_on_complete:
            self.reset(reset_archive=False)
