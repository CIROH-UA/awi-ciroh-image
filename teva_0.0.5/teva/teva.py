""" The Tandem Evolutionary Algorithm """
import traceback
import typing
import logging
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from teva import dnfea
from teva import ccea
from teva.ccea import conjunctive_clause
from teva.dnfea import disjunctive_clause
from teva.utilities import hyge
from teva.base import feature
from teva.utilities import plotting


class TEVAError(Exception):
    """ Basic TEVA Exception """
    def __init__(self, *args):
        super().__init__(*args)

class TEVAData:
    """ A data structure object for storing TEVA run data """
    def __init__(self):
        self.observations: np.ndarray | None = None
        self.classes: np.ndarray | None = None
        self.feature_types: list[feature.FeatureType] | None = None
        self.feature_domains: list[feature.FeatureDomain] | None = None
        self.unique_classes: np.ndarray | None = None
        self.use_best_cc_fitness: bool = False


class TEVA:
    """ The Tandem Evolutionary Algorithm

    Fuses CCEA and DNFEA to improve results

    CCEA:

    :param ccea_offspring_per_gen: the number of new offspring created every g_n generations
    :param ccea_total_generations: Total number of generations to run the algorithm for
    :param ccea_max_order: The maximum order of ccs that can be stored in the archive
    :param ccea_n_age_layers: The number of layers that are not archived
    :param ccea_layer_size: Maximum population of age layers
    :param ccea_gen_per_growth: The number of generations between age cycles
    :param ccea_offspring_per_gen: The number of new offspring evolved each generation through crossover or mutation
    :param ccea_archive_offspring_per_gen: The number of new offspring evolved each generation from the archive
    :param ccea_archive_bin_size: Maximum population of archive bins
    :param ccea_num_new_pop: The number of archive offspring created per evolution
    :param ccea_p_crossover: Probability that crossover will occur during evolution
    :param ccea_p_mutation: Probability that mutation will occur during evolution
    :param ccea_tournament_size: Number of parents with replacement that are in the mating tournament,
         only the most fit will mate
    :param ccea_selection_exponent: An exponent that increases the probability of selecting CCs whose target observations
        are underrepresented in the archived DNFs for the creation of novel DNFs
    :param ccea_use_sensitivity: enable use of feature sensitivity test
    :param ccea_sensitivity_threshold: feature sensitivity minimum threshold

    :param ccea_fitness_function: The fitness function to use for determining clause fitness The fitness function must
        take the following form:

            fitness_function(number, number, number, number) -> float

    :param ccea_seed: random seed for reproducibility
    :param ccea_smart_seeding: If True, seeding for new DNFs will take the current archive into account to ensure a
        greater diversity of clauses

    :param ccea_p_wildcard: Probability that a feature selected from mutation will be removed from the clause
    :param ccea_selective_mutation: If true, mutation will use wildcards to be more selective
    :param ccea_feature_selection_ratio: The ratio of features to be selected from each parent during crossover
    :param ccea_fitness_threshold: The initial threshold to use for CC fitnesses

    DNFEA:

    :param dnfea_offspring_per_gen: the number of new offspring created every g_n generations
    :param dnfea_total_generations: Total number of generations to run the algorithm for
    :param dnfea_max_order: The maximum order of ccs that can be stored in the archive
    :param dnfea_n_age_layers: The number of layers that are not archived
    :param dnfea_layer_size: Maximum population of age layers
    :param dnfea_gen_per_growth: The number of generations between age cycles
    :param dnfea_offspring_per_gen: The number of new offspring evolved each generation through crossover or mutation
    :param dnfea_archive_offspring_per_gen: The number of new offspring evolved each generation from the archive
    :param dnfea_archive_bin_size: Maximum population of archive bins
    :param dnfea_num_new_pop: The number of archive offspring created per evolution
    :param dnfea_p_crossover: Probability that crossover will occur during evolution
    :param dnfea_p_mutation: Probability that mutation will occur during evolution
    :param dnfea_tournament_size: Number of parents with replacement that are in the mating tournament,
         only the most fit will mate
    :param dnfea_selection_exponent: An exponent that increases the probability of selecting CCs whose target
        observations are underrepresented in the archived DNFs for the creation of novel DNFs
    :param dnfea_use_sensitivity: enable use of feature sensitivity test
    :param dnfea_sensitivity_threshold: feature sensitivity minimum threshold

    :param dnfea_fitness_function: The fitness function to use for determining clause fitness The fitness function must
        take the following form:

            fitness_function(number, number, number, number) -> float

    :param dnfea_seed: random seed for reproducibility
    :param dnfea_smart_seeding: If True, seeding for new DNFs will take the current archive into account to ensure a
        greater diversity of clauses

    :param dnfea_p_targeted_mutation: Probability of a bit flip mutation
    :param dnfea_p_targeted_crossover: Probability that a mate is selected on fitness only
    :param dnfea_p_union: Probability of a union crossover
    :param dnfea_p_intersection: Probability of an intersection crossover
    :param dnfea_p_crossover_algorithm: Probability cumulative sum for alternative crossover mate selection
    :param dnfea_p_mutation_algorithm: Probability cumulative sum for alternative targeted mutation

    General:

    :param feature_names: A list of names for each feature that will be used in exporting results
    :param run_name: The name of the current run of this algorithm.  This will be used in logging and output names
    """
    def __init__(self,
                 # CCEA Required Arguments
                 ccea_total_generations: int,
                 ccea_max_order: int,
                 ccea_n_age_layers: int,
                 ccea_layer_size: int,

                 # DNFEA Required Arguments
                 dnfea_total_generations: int,
                 dnfea_max_order: int,
                 dnfea_n_age_layers: int,
                 dnfea_layer_size: int,

                 # CCEA Default Arguments
                 ccea_gen_per_growth: int = 3,  # GENn
                 ccea_offspring_per_gen: int = None,  # POPn
                 ccea_archive_offspring_per_gen: int = None,  # ArchOff
                 ccea_archive_bin_size: int = None,
                 ccea_num_new_pop: int = None,
                 ccea_p_crossover: float = 0.5,  # Px
                 ccea_p_mutation: float = None,  # Pm
                 ccea_tournament_size: int = 3,  # TournSize
                 ccea_selection_exponent: float = 3.0,  # SelExp,
                 ccea_fitness_function: typing.Callable = hyge.hygepmf,
                 ccea_seed: int = None,
                 ccea_use_sensitivity: bool = True,
                 ccea_sensitivity_threshold: float = -np.inf,
                 ccea_smart_seeding: bool = True,

                 # DNFEA Default Arguments
                 dnfea_gen_per_growth: int = 3,  # GENn
                 dnfea_offspring_per_gen: int = None,  # POPn
                 dnfea_archive_offspring_per_gen: int = None,  # ArchOff
                 dnfea_archive_bin_size: int = None,
                 dnfea_num_new_pop: int = None,
                 dnfea_p_crossover: float = 0.5,  # Px
                 dnfea_p_mutation: float = 0.1,  # Pm
                 dnfea_tournament_size: int = 3,  # TournSize
                 dnfea_selection_exponent: float = 3.0,  # SelExp,
                 dnfea_use_sensitivity: bool = False,
                 dnfea_sensitivity_threshold: float = -np.inf,
                 dnfea_fitness_function: typing.Callable = hyge.hygepmf,
                 dnfea_seed: int = None,
                 dnfea_smart_seeding: bool = True,  # winit

                 # CCEA Specific Inputs
                 ccea_p_wildcard: float = 0.75,
                 ccea_selective_mutation: bool = False,
                 ccea_feature_selection_ratio: float = 0.5,
                 ccea_fitness_threshold: float = None,

                 # DNFEA Specific Inputs
                 dnfea_p_targeted_mutation: float = 0.8,  # Pbf
                 dnfea_p_targeted_crossover: float = 0.75,  # Pxf
                 dnfea_p_union: float = 0.5,  # PrUn
                 dnfea_p_intersection: float = 0.5,  # PrIn
                 dnfea_p_crossover_algorithm: list[float] = None,  # PxAlt (must sum to 1)
                 dnfea_p_mutation_algorithm: list[float] = None,  # PmAlt (must sum to 1)

                 run_name: str = "",
                 feature_names: list[str] = None
                 ):
        # CCEA Required Arguments
        self.ccea_total_generations: int = ccea_total_generations
        self.ccea_max_order: int = ccea_max_order
        self.ccea_n_age_layers: int = ccea_n_age_layers
        self.ccea_layer_size: int = ccea_layer_size

        # DNFEA Required Arguments
        self.dnfea_total_generations: int = dnfea_total_generations
        self.dnfea_max_order: int = dnfea_max_order
        self.dnfea_n_age_layers: int = dnfea_n_age_layers
        self.dnfea_layer_size: int = dnfea_layer_size

        # CCEA Default Arguments
        self.ccea_gen_per_growth: int = ccea_gen_per_growth
        self.ccea_offspring_per_gen: int = ccea_offspring_per_gen
        self.ccea_archive_offspring_per_gen: int = ccea_archive_offspring_per_gen
        self.ccea_archive_bin_size: int = ccea_archive_bin_size
        self.ccea_num_new_pop: int = ccea_num_new_pop
        self.ccea_p_x: float = ccea_p_crossover
        self.ccea_p_m: float = ccea_p_mutation
        self.ccea_tournament_size: int = ccea_tournament_size
        self.ccea_selection_exponent: float = ccea_selection_exponent
        self.ccea_fitness_function: typing.Callable = ccea_fitness_function
        self.ccea_seed: int = ccea_seed
        self.ccea_use_sensitivity: bool = ccea_use_sensitivity
        self.ccea_sensitivity_threshold: float = ccea_sensitivity_threshold
        self.ccea_smart_seeding: bool = ccea_smart_seeding

        # DNFEA Default Arguments
        self.dnfea_gen_per_growth: int = dnfea_gen_per_growth
        self.dnfea_offspring_per_gen: int = dnfea_offspring_per_gen
        self.dnfea_archive_offspring_per_gen: int = dnfea_archive_offspring_per_gen
        self.dnfea_archive_bin_size: int = dnfea_archive_bin_size
        self.dnfea_num_new_pop: int = dnfea_num_new_pop
        self.dnfea_p_x: float = dnfea_p_crossover
        self.dnfea_p_m: float = dnfea_p_mutation
        self.dnfea_tournament_size: int = dnfea_tournament_size
        self.dnfea_selection_exponent: float = dnfea_selection_exponent
        self.dnfea_use_sensitivity: bool = dnfea_use_sensitivity
        self.dnfea_sensitivity_threshold: float = dnfea_sensitivity_threshold
        self.dnfea_fitness_function: typing.Callable = dnfea_fitness_function
        self.dnfea_seed: int = dnfea_seed
        self.dnfea_smart_seeding: bool = dnfea_smart_seeding

        # CCEA Specific Inputs
        self.ccea_p_wc: float = ccea_p_wildcard
        self.ccea_selective_mutation: bool = ccea_selective_mutation
        self.ccea_feature_selection_ratio: float = ccea_feature_selection_ratio
        self.ccea_fitness_threshold: float = ccea_fitness_threshold

        # DNFEA Specific Inputs
        self.dnfea_p_f: float = dnfea_p_targeted_mutation
        self.dnfea_p_xf: float = dnfea_p_targeted_crossover
        self.dnfea_p_u: float = dnfea_p_union
        self.dnfea_p_i: float = dnfea_p_intersection
        self.dnfea_p_x_algorithm: list[float] = dnfea_p_crossover_algorithm  # PxAlt (must sum to 1)
        self.dnfea_p_m_algorithm: list[float] = dnfea_p_mutation_algorithm  # PmAlt (must sum to 1)

        self.run_name = run_name
        self.feature_names = feature_names

        self.ccea_run_name = f"{run_name}_ccea"
        self.dnfea_run_name = f"{run_name}_dnfea"

        self._ccea: ccea.CCEA | None = None
        self._dnfea: dnfea.DNFEA | None = None

        self._ccea_archive_all: dict[typing.Any, list[conjunctive_clause.ConjunctiveClause]] = {}
        self._dnfea_archive_all: dict[typing.Any, list[disjunctive_clause.DisjunctiveClause]] = {}
        self._ccea_archive: dict[typing.Any, dict[typing.Any, list[conjunctive_clause.ConjunctiveClause]]] = {}
        self._dnfea_archive: dict[typing.Any, dict[typing.Any, list[disjunctive_clause.DisjunctiveClause]]] = {}
        self.run_name = run_name

        self._fit_data: TEVAData | None = None

        self._init_ccea()
        self._init_dnfea()

    def _init_ccea(self):
        """ Uses TEVA properties to create a CCEA object """
        self._ccea: ccea.CCEA = ccea.CCEA(offspring_per_gen=self.ccea_offspring_per_gen,
                                          total_generations=self.ccea_total_generations,
                                          max_order=self.ccea_max_order,
                                          layer_size=self.ccea_layer_size,
                                          fitness_threshold=self.ccea_fitness_threshold,
                                          n_age_layers=self.ccea_n_age_layers,
                                          gen_per_growth=self.ccea_gen_per_growth,
                                          archive_bin_size=self.ccea_archive_bin_size,
                                          archive_offspring_per_gen=self.ccea_archive_offspring_per_gen,
                                          p_crossover=self.ccea_p_x,
                                          p_wildcard=self.ccea_p_wc,
                                          p_mutation=self.ccea_p_m,
                                          num_new_pop=self.ccea_num_new_pop,
                                          tournament_size=self.ccea_tournament_size,
                                          selective_mutation=self.ccea_selective_mutation,
                                          use_sensitivity=self.ccea_use_sensitivity,
                                          feature_selection_ratio=self.ccea_feature_selection_ratio,
                                          sensitivity_threshold=self.ccea_sensitivity_threshold,
                                          fitness_function=self.ccea_fitness_function,
                                          selection_exponent=self.ccea_selection_exponent,
                                          seed=self.ccea_seed,
                                          smart_seeding=self.ccea_smart_seeding,
                                          run_name=self.ccea_run_name,
                                          feature_names=self.feature_names)

    def _init_dnfea(self):
        """ Uses TEVA properties to create a DNFEA object """
        self._dnfea: dnfea.DNFEA = dnfea.DNFEA(total_generations=self.dnfea_total_generations,
                                               gen_per_growth=self.dnfea_gen_per_growth,
                                               n_age_layers=self.dnfea_n_age_layers,
                                               max_order=self.dnfea_max_order,
                                               p_crossover=self.dnfea_p_x,
                                               p_mutation=self.dnfea_p_m,
                                               p_targeted_mutation=self.dnfea_p_f,
                                               p_targeted_crossover=self.dnfea_p_xf,
                                               p_union=self.dnfea_p_u,
                                               p_intersection=self.dnfea_p_i,
                                               archive_bin_size=self.dnfea_archive_bin_size,
                                               num_new_pop=self.dnfea_num_new_pop,
                                               p_crossover_algorithm=self.dnfea_p_x_algorithm,
                                               p_mutation_algorithm=self.dnfea_p_m_algorithm,
                                               offspring_per_gen=self.dnfea_offspring_per_gen,
                                               archive_offspring_per_gen=self.dnfea_archive_offspring_per_gen,
                                               tournament_size=self.dnfea_tournament_size,
                                               layer_size=self.dnfea_layer_size,
                                               smart_seeding=self.dnfea_smart_seeding,
                                               selection_exponent=self.dnfea_selection_exponent,
                                               fitness_function=self.dnfea_fitness_function,
                                               use_sensitivity=self.dnfea_use_sensitivity,
                                               sensitivity_threshold=self.dnfea_sensitivity_threshold,
                                               seed=self.dnfea_seed,
                                               run_name=self.dnfea_run_name)

    def reset(self, reset_archive: bool = True):
        """ Resets the algorithm to a run-ready state

        :param reset_archive: If True, the archive will be reset as well
        """
        self._ccea.reset(reset_archive=reset_archive)
        self._dnfea.reset(reset_archive=reset_archive)

        if reset_archive:
            self._ccea_archive = {}
            self._dnfea_archive = {}

    def _export_ccea(self, all_ccs: list[conjunctive_clause.ConjunctiveClause]) -> pd.DataFrame:
        """ Exports a list of CCs to a data presentation dataframe

        :param all_ccs: A list of all the CCs to export to file
        """
        cc_rows = []

        if len(all_ccs) == 0:
            return pd.DataFrame([])

        for i, cc in enumerate(all_ccs):
            row_dict = {
                "class": cc.classification,
                "mask": cc.mask.astype(int),
                "fitness": cc.fitness,
                "order": cc.order(),
                "age": cc.age,
                "cov": cc.cov,
                "ppv": cc.ppv,
                "tp": cc.target_coverage_count,
                "tn": cc.nontarget_count - cc.nontarget_coverage_count,
                "fp": cc.nontarget_coverage_count,
                "fn": cc.target_count - cc.target_coverage_count
            }

            for j in range(cc.mask.shape[0]):
                if cc.mask[j]:
                    feature_set = cc.items[j].feature_set()

                    if cc.items[j].feature_type() == feature.FeatureType.CATEGORICAL:
                        row_dict[self._ccea.feature_names[j]] = feature_set
                    elif cc.items[j].feature_type() == feature.FeatureType.ORDINAL:
                        row_dict[self._ccea.feature_names[j]] = str([int(min(feature_set)), int(max(feature_set))])
                    else:
                        row_dict[self._ccea.feature_names[j]] = str([float(min(feature_set)), float(max(feature_set))])
                else:
                    row_dict[self._ccea.feature_names[j]] = ""

            cc_rows.append(pd.DataFrame(pd.Series(row_dict, name=i)).T)

        return pd.concat(cc_rows)

    def _export_dnfea(self, all_dnfs: list[disjunctive_clause.DisjunctiveClause]) -> pd.DataFrame:
        """ Exports a list of DNFs to a data presentation dataframe

        :param all_dnfs: A list of all the DNFs to export to file
        """
        dnf_rows = []

        if len(all_dnfs) == 0:
            return pd.DataFrame([])

        for i, dnf in enumerate(all_dnfs):
            row_dict = {
                "class": dnf.classification,
                "mask": dnf.mask.astype(int),
                "fitness": dnf.fitness,
                "order": dnf.order(),
                "age": dnf.age,
                "cov": dnf.cov,
                "ppv": dnf.ppv,
                "tp": dnf.target_coverage_count,
                "tn": dnf.nontarget_count - dnf.nontarget_coverage_count,
                "fp": dnf.nontarget_coverage_count,
                "fn": dnf.target_count - dnf.target_coverage_count
            }
            for j in range(dnf.mask.shape[0]):
                if dnf.mask[j]:
                    row_dict[f"cc_{j}"] = 1
                else:
                    row_dict[f"cc_{j}"] = 0

            dnf_rows.append(pd.DataFrame(pd.Series(row_dict, name=i)).T)

        concat = pd.concat(dnf_rows)
        return concat

    def _ccea_stats(self) -> pd.DataFrame:
        """ Calculates CCEA statistics and add them to a dataframe """
        all_ccs = []
        archived_ccs = self.get_all_archived_ccs()
        for _, val in archived_ccs.items():
            all_ccs.extend(val)

        feature_tuples = [[] for _ in range(self._ccea.max_order)]
        for i, clause in enumerate(all_ccs):
            for j in range(self._ccea.max_order):
                if clause.mask[j]:
                    feature_set = clause.items[j].feature_set()

                    if clause.items[j].feature_type() != feature.FeatureType.CATEGORICAL:
                        feature_tuples[j].append((min(feature_set), max(feature_set)))

        feature_rows = []

        for i in range(self.ccea_max_order):
            feat = feature_tuples[i]
            if len(feature_tuples[i]) == 0:
                feature_rows.append(pd.DataFrame(pd.Series({}, name=self._ccea.feature_names[i])).T)
                continue

            mins, maxes = zip(*feat)
            mins = np.array(mins)
            maxes = np.array(maxes)
            diffs = maxes - mins

            feature_dict = {
                "min_min": np.min(mins).item(),
                "min_max": np.max(mins).item(),
                "min_median": np.median(mins).item(),
                "min_mean": np.mean(mins).item(),

                "max_min": np.min(maxes).item(),
                "max_max": np.max(maxes).item(),
                "max_median": np.median(maxes).item(),
                "max_mean": np.mean(maxes).item(),

                "range_min": np.min(diffs).item(),
                "range_max": np.max(diffs).item(),
                "range_median": np.median(diffs).item(),
                "range_mean": np.mean(diffs).item(),

                "count": maxes.shape[0]
            }

            feature_rows.append(pd.DataFrame(pd.Series(feature_dict, name=self._ccea.feature_names[i])).T)

        concat = pd.concat(feature_rows)
        return concat

    def export(self, ccea_filepath: str = "", dnfea_filepath: str = ""):
        """ Export CCEA and DNFEA results to an Excel file

        :param ccea_filepath: The filepath of the Excel file for CCEA results
        :param dnfea_filepath: The filepath of the Excel file for DNFEA results
        """
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if ccea_filepath == "":
            ccea_filepath = f"{'ccea_results'}_{now}.csv"

        if dnfea_filepath == "":
            dnfea_filepath = f"{'dnfea_results'}_{now}.csv"


        try:
            archived_ccs = self.get_all_archived_ccs()
            if len(archived_ccs.values()) > 0:
                # feature_stats = self._ccea_stats()
                with pd.ExcelWriter(ccea_filepath) as cc_writer:
                    #                 feature_stats.to_excel(cc_writer, sheet_name="Feature_Stats")

                    for classification, cc in archived_ccs.items():
                        target_df = self._export_ccea(cc)
                        target_df.to_excel(cc_writer, sheet_name=f"CCEA_{classification}")
            else:
                print("Not enough CCs to write to file")
        except Exception as e:
            raise e
            # print("Failed to write CCEA data file: ", str(e))

        try:
            archived_dnfs = self.get_all_archived_dnfs()
            if len(archived_dnfs.values()) > 0:
                with pd.ExcelWriter(dnfea_filepath) as dnf_writer:
                    for classification, dnf in archived_dnfs.items():
                        target_df = self._export_dnfea(dnf)
                        target_df.to_excel(dnf_writer, sheet_name=f"DNFEA_{classification}")
            else:
                print("Not enough DNFs to write to file")
        except Exception as e:
            raise e
            # print("Failed to write DNFEA data file: ", str(e))

    def get_ccea(self) -> ccea.CCEA:
        """ Gets the active CCEA algorithm object :class:`ccea.CCEA`
        :return: The active CCEA algorithm.
        """
        return self._ccea

    def get_dnfea(self) -> dnfea.DNFEA:
        """ Gets the active DNFEA algorithm object :class:`dnfea.DNFEA`
        :return: The active DNFEA algorithm.
        """
        return self._dnfea

    def get_archived_ccs(self, target_class=None) -> dict:
        """ Get ccea archive as a dictionary

        :param target_class: If provided, limits results to a single target class

        :return: The requested portion of the ccea archive
        """
        archive = self._ccea_archive
        if target_class is None:
            return archive

        return archive[target_class]

    def get_archived_dnfs(self, target_class=None) -> dict:
        """ Get dnfea archive as a dictionary

        :param target_class: If provided, limits results to a single target class

        :return: The requested portion of the dnfea archive
        """
        archive = self._dnfea_archive
        if target_class is None:
            return archive

        return archive[target_class]

    def get_all_archived_ccs(self) -> dict[typing.Any, list[conjunctive_clause.ConjunctiveClause]]:
        """ Gets all archived ccs as a dictionary of lists with the form: {class: [clauses]}

        :return: A dictionary of archived ccs with the class as keys
        """
        return self._ccea_archive_all

    def get_all_archived_dnfs(self) -> dict[typing.Any, list[disjunctive_clause.DisjunctiveClause]]:
        """ Gets all archived dnfs as a dictionary of lists with the form: {class: [clauses]}

        :return: A dictionary of archived dnfs with the class as keys
        """
        return self._dnfea_archive_all

    def fit(self,
            observation_table: np.ndarray | pd.DataFrame,
            classifications: np.ndarray | pd.Series,
            feature_types: list[feature.FeatureType] = None,
            use_best_cc_fitness: bool = False) -> np.ndarray:
        """ Prepare this TEVA algorithm to run on data from the observation array.  This function must be run before
        you can call :function:`TEVA.run()` or :function:`TEVA.run_all_targets()`

        :param observation_table: A 2D array where each column represents a feature, and each row represents an
            observation
        :param classifications: A 1D array of classification values where each value corresponds to an observation in
            the observation array
        :param feature_types: A list of the feature type of each feature column.  If undefined, the algorithm will try
            to determine automatically
        :param use_best_cc_fitness: If True, the DNF will use the best cc fitness as a baseline instead of the worst

        :return: An array of unique classifications, which can be useful if running the algorithm on each class
            separately.
        """
        self.reset()

        self._fit_data = TEVAData()

        if isinstance(observation_table, pd.DataFrame):
            self._fit_data.observations = observation_table.to_numpy()
        else:
            self._fit_data.observations = observation_table

        if isinstance(classifications, pd.Series):
            self._fit_data.classes = classifications.to_numpy()
        else:
            self._fit_data.classes = classifications

        self._fit_data.feature_types = feature_types

        # if feature types are not passed as an argument, determine them automatically
        if feature_types is None:
            feature_types = feature.determine_feature_types(self._fit_data.observations)

        self._fit_data.feature_domains = feature.find_feature_domains(self._fit_data.observations, feature_types)
        self._fit_data.unique_classes = np.unique(self._fit_data.classes)

        self._fit_data.use_best_cc_fitness = use_best_cc_fitness

        return self._fit_data.unique_classes

    def run_all_targets(self,
                        output_logging_level: int = logging.INFO,
                        logfile_logging_level: int = logging.INFO,
                        visualize: bool = False):
        """ Run a fitted model on every target class.  To run only one target class at a time,
        see :function:`run()`

        .. note:: Requires :function:`TEVA.fit()` to have been called before the algorithm can run.

        :param output_logging_level: The logging level of the STDOUT reporting stream
        :param logfile_logging_level: The logging level of the Logfile reporting stream
        :param visualize: Enables console visualization
        """

        if self._fit_data is None:
            raise TEVAError("You must run TEVA.fit() before calling 'TEVA.run_all_targets()")

        for unique_class in self._fit_data.unique_classes:
            try:
                self.run(target_class=unique_class,
                         output_logging_level=output_logging_level,
                         logfile_logging_level=logfile_logging_level,
                         visualize=visualize,
                         reset_on_complete=False)
            except dnfea.dnfea.DNFEAError as e:
                logging.getLogger("dnfea").error(f"Target Failed: {unique_class} | DNFEA | {e}")
                traceback.print_exc()
            except ccea.ccea.CCEAError as e:
                logging.getLogger("ccea").error(f"Target Failed: {unique_class} | CCEA | {e}")
                traceback.print_exc()

    def run(self,
            target_class: typing.Any,
            output_logging_level: int = logging.INFO,
            logfile_logging_level: int = logging.INFO,
            visualize: bool = False,
            reset_on_complete: bool = True):
        """ Run a fitted TEVA model on a specific target class.  To run more than one target class at the same
        time, see :function:`TEVA.run_all_targets()`

        .. note:: Requires :function:`TEVA.fit()` to have been called before the algorithm can run.

        :param target_class: The target class to focus this run of the algorithm on
        :param output_logging_level: The logging level of the STDOUT reporting stream
        :param logfile_logging_level: The logging level of the Logfile reporting stream
        :param visualize: Enables console visualization
        :param reset_on_complete: If True, the archive will be reset when the run is completed
        """

        if self._fit_data is None:
            raise TEVAError("You must run TEVA.fit() before calling 'TEVA.run()")

        try:
            self._init_ccea()
            self._ccea.fit(observation_table=self._fit_data.observations,
                           classifications=self._fit_data.classes,
                           feature_types=self._fit_data.feature_types)

            self._ccea.run(target_class,
                           output_logging_level=output_logging_level,
                           logfile_logging_level=logfile_logging_level,
                           visualize=visualize)

            self._ccea_archive[target_class] = self._ccea.get_archived()
            self._ccea_archive_all[target_class] = self._ccea.get_all_archive_values()
        except Exception as e:
            raise ccea.ccea.CCEAError from e

        try:
            self._init_dnfea()
            self._dnfea.fit(observation_table=self._fit_data.observations,
                            classifications=self._fit_data.classes,
                            conjunctive_clauses=self._ccea.get_all_archive_values(),
                            use_cc_best_fitness=self._fit_data.use_best_cc_fitness)

            self._dnfea.run(target_class,
                            output_logging_level=output_logging_level,
                            logfile_logging_level=logfile_logging_level,
                            visualize=visualize)

            self._dnfea_archive[target_class] = self._dnfea.get_archived()
            self._dnfea_archive_all[target_class] = self._dnfea.get_all_archive_values()
        except Exception as e:
            raise dnfea.dnfea.DNFEAError from e

        if reset_on_complete:
            self.reset(reset_archive=False)

    def _do_plot(self,
                 ccs,
                 dnfs,
                 fitnesses,
                 boundaries,
                 target_title,
                 figure,

                 x_label: str = "Observation Coverage",
                 autoscale_x_axis: bool = False,
                 autoscale_y_axis: bool = False,
                 fig_title: str = None,

                 contour_color: tuple = (0.75, 0.75, 0.75),
                 contour_linestyle="--",
                 contour_linewidth=0.5,

                 cc_point_size: float = 8,
                 cc_marker: str = "o",
                 cc_min_order_color: tuple = (0.75, 1.0, 0.75),
                 cc_max_order_color: tuple = (0.0, 1.0, 0.0),
                 cc_ylabel: str = "CC Positive Predictive Value",

                 dnf_point_size: float = 8,
                 dnf_marker: str = "s",
                 dnf_min_order_color: tuple = (0.75, 0.75, 1.0),
                 dnf_max_order_color: tuple = (0.0, 0.0, 1.0),
                 dnf_ylabel: str = "DNF Positive Predictive Value"):
        if figure is None:
            figure, axis = plt.subplots(1, dpi=300)
        else:
            axis = figure.get_axes()[0]

        if fitnesses is not None and boundaries is not None:
            plotting.plot_contours(fitnesses,
                                   boundaries,
                                   axis=axis,
                                   color=contour_color,
                                   linestyle=contour_linestyle,
                                   linewidth=contour_linewidth)

        axis.set_xlabel(x_label)

        if ccs is None and dnfs is not None:
            cc_axis = None
            dnf_axis = axis
        elif dnfs is None and ccs is not None:
            cc_axis = axis
            dnf_axis = None
        elif dnfs is not None and ccs is not None:
            cc_axis = axis
            dnf_axis = axis.twinx()

        cc_title = ""
        if ccs is not None:
            cc_title = " | CCS"
            plotting.plot_ccs(ccs,
                              axis=cc_axis,
                              point_size=cc_point_size,
                              marker=cc_marker,
                              min_order_color=cc_min_order_color,
                              max_order_color=cc_max_order_color)

            if not autoscale_x_axis:
                cc_axis.set_xlim(0.0, 1.05)
            if not autoscale_y_axis:
                cc_axis.set_ylim(0.0, 1.05)

            cc_axis.set_ylabel(cc_ylabel)

        dnf_title = ""
        if dnfs is not None:
            dnf_title = " | DNFS"

            plotting.plot_dnfs(dnfs,
                               axis=dnf_axis,
                               point_size=dnf_point_size,
                               marker=dnf_marker,
                               min_order_color=dnf_min_order_color,
                               max_order_color=dnf_max_order_color)

            if not autoscale_x_axis:
                dnf_axis.set_xlim(0.0, 1.05)
            if not autoscale_y_axis:
                dnf_axis.set_ylim(0.0, 1.05)
            dnf_axis.set_ylabel(dnf_ylabel)
            dnf_axis.grid(False)

        if fig_title is None:
            fig_title = f"TEVA: Coverage vs Fitness{cc_title}{dnf_title} - {target_title}"
        figure.suptitle(fig_title)
        axis.grid(True, linewidth=0.25)
        return figure

    def plot(self,
             target_class,
             plot_ccs: bool = True,
             plot_dnfs: bool = True,
             plot_contours: bool = True,
             figure: plt.Figure = None,

             x_label: str = "Observation Coverage",
             autoscale_x_axis: bool = False,
             autoscale_y_axis: bool = False,
             fig_title: str = None,

             contour_color: tuple = (0.75, 0.75, 0.75),
             contour_linestyle="--",
             contour_linewidth=0.5,

             cc_point_size: float = 8,
             cc_marker: str = "o",
             cc_min_order_color: tuple = (0.75, 1.0, 0.75),
             cc_max_order_color: tuple = (0.0, 1.0, 0.0),
             cc_ylabel: str = "CC Positive Predictive Value",

             dnf_point_size: float = 8,
             dnf_marker: str = "s",
             dnf_min_order_color: tuple = (0.75, 0.75, 1.0),
             dnf_max_order_color: tuple = (0.0, 0.0, 1.0),
             dnf_ylabel: str = "DNF Positive Predictive Value") -> plt.Figure:
        """ Generates a plot for the given target class

        :param target_class: The target class to plot
        :param plot_ccs: If True, CCs will be added to the plot
        :param plot_dnfs: If True, DNFs will be added to the plot
        :param plot_contours: If True, contours will be added to the plot
        :param figure: If provided, this figure will be used, otherwise a new one will be created

        :param x_label: The label of the X axis
        :param autoscale_x_axis: If True, the x-axis will be autoscaled
        :param autoscale_y_axis: If True, the y-axis will be autoscaled

        :param contour_color: The color of the countour lines
        :param contour_linestyle: The style of the contour lines
        :param contour_linewidth: The width of the contour lines

        :param cc_point_size: The marker style for the CC points
        :param cc_marker: The size of the CC points
        :param cc_min_order_color: The color of the minimum CC order
        :param cc_max_order_color: The color of the maximum CC order
        :param cc_ylabel: The label of the Y axis for CCs

        :param dnf_point_size: The marker style for the DNF points
        :param dnf_marker: The size of the DNF points
        :param dnf_min_order_color: The color of the minimum DNF order
        :param dnf_max_order_color: The color of the maximum DNF order
        :param dnf_ylabel: The label of the Y axis for DNFs

        :return: The figure to which the plot was drawn
        """
        if self._fit_data is None:
            raise TEVAError("You must run TEVA.fit() before calling 'TEVA.plot()")

        fitness = None
        borders = None
        ccs = None
        dnfs = None

        if plot_contours:
            fitness, borders = hyge.hygcontours(self._fit_data.classes,
                                                target_class,
                                                25,
                                                self._ccea.fitness_function)

        if plot_ccs:
            ccs = self._ccea_archive[target_class]

        if plot_dnfs:
            dnfs = self._dnfea_archive[target_class]

        target_title = f"{target_class}"

        return self._do_plot(ccs=ccs,
                             dnfs=dnfs,
                             fitnesses=fitness,
                             boundaries=borders,
                             target_title=target_title,
                             figure=figure,

                             x_label=x_label,
                             autoscale_x_axis=autoscale_x_axis,
                             autoscale_y_axis=autoscale_y_axis,
                             fig_title=fig_title,

                             contour_color=contour_color,
                             contour_linestyle=contour_linestyle,
                             contour_linewidth=contour_linewidth,

                             cc_point_size=cc_point_size,
                             cc_marker=cc_marker,
                             cc_min_order_color=cc_min_order_color,
                             cc_max_order_color=cc_max_order_color,
                             cc_ylabel=cc_ylabel,

                             dnf_point_size=dnf_point_size,
                             dnf_marker=dnf_marker,
                             dnf_min_order_color=dnf_min_order_color,
                             dnf_max_order_color=dnf_max_order_color,
                             dnf_ylabel=dnf_ylabel)

    def plot_all(self,
                 plot_ccs: bool = True,
                 plot_dnfs: bool = True,
                 plot_contours: bool = True,
                 single_plot: bool = False,

                 x_label: str = "Observation Coverage",
                 autoscale_x_axis: bool = False,
                 autoscale_y_axis: bool = False,
                 fig_title: str = None,

                 contour_color: tuple = (0.75, 0.75, 0.75),
                 contour_linestyle="--",
                 contour_linewidth=0.5,

                 cc_point_size: float = 8,
                 cc_marker: str = "o",
                 cc_min_order_color: tuple = (0.75, 1.0, 0.75),
                 cc_max_order_color: tuple = (0.0, 1.0, 0.0),
                 cc_ylabel: str = "CC Positive Predictive Value",

                 dnf_point_size: float = 8,
                 dnf_marker: str = "s",
                 dnf_min_order_color: tuple = (0.75, 0.75, 1.0),
                 dnf_max_order_color: tuple = (0.0, 0.0, 1.0),
                 dnf_ylabel: str = "DNF Positive Predictive Value"):
        """ Generates plots for all unique classes

        :param plot_ccs: If True, CCs will be added to the plot
        :param plot_dnfs: If True, DNFs will be added to the plot
        :param plot_contours: If True, contours will be added to the plot
        :param single_plot: If True, all targets will be drawn on the same figure, otherwise a figure will be created
            for each

        :param x_label: The label of the X axis
        :param autoscale_x_axis: If True, the x-axis will be autoscaled
        :param autoscale_y_axis: If True, the y-axis will be autoscaled

        :param contour_color: The color of the countour lines
        :param contour_linestyle: The style of the contour lines
        :param contour_linewidth: The width of the contour lines

        :param cc_point_size: The marker style for the CC points
        :param cc_marker: The size of the CC points
        :param cc_min_order_color: The color of the minimum CC order
        :param cc_max_order_color: The color of the maximum CC order
        :param cc_ylabel: The label of the Y axis for CCs

        :param dnf_point_size: The marker style for the DNF points
        :param dnf_marker: The size of the DNF points
        :param dnf_min_order_color: The color of the minimum DNF order
        :param dnf_max_order_color: The color of the maximum DNF order
        :param dnf_ylabel: The label of the Y axis for DNFs

        :return: The figure to which the plot was drawn
        """

        if self._fit_data is None:
            raise TEVAError("You must run TEVA.fit() before calling 'TEVA.plot_all()")

        figure = None
        if single_plot:
            figure, axes = plt.subplots(1, dpi=300)

        for unique_class in self._fit_data.unique_classes:
            ccs = None
            if plot_ccs:
                if unique_class not in self._ccea_archive:
                    continue
                ccs = self._ccea_archive[unique_class]

            dnfs = None
            if plot_dnfs:
                if unique_class not in self._dnfea_archive:
                    continue
                dnfs = self._dnfea_archive[unique_class]

            fitnesses = None
            boundaries = None
            if plot_contours:
                fitnesses, boundaries = hyge.hygcontours(self._fit_data.classes,
                                                         unique_class,
                                                         25,
                                                         self._ccea.fitness_function)

            if figure is not None:
                target_title = "All"
            else:
                target_title = f"{unique_class}"

            self._do_plot(ccs=ccs,
                          dnfs=dnfs,
                          fitnesses=fitnesses,
                          boundaries=boundaries,
                          target_title=target_title,
                          figure=figure,

                          x_label=x_label,
                          autoscale_x_axis=autoscale_x_axis,
                          autoscale_y_axis=autoscale_y_axis,
                          fig_title=fig_title,

                          contour_color=contour_color,
                          contour_linestyle=contour_linestyle,
                          contour_linewidth=contour_linewidth,

                          cc_point_size=cc_point_size,
                          cc_marker=cc_marker,
                          cc_min_order_color=cc_min_order_color,
                          cc_max_order_color=cc_max_order_color,
                          cc_ylabel=cc_ylabel,

                          dnf_point_size=dnf_point_size,
                          dnf_marker=dnf_marker,
                          dnf_min_order_color=dnf_min_order_color,
                          dnf_max_order_color=dnf_max_order_color,
                          dnf_ylabel=dnf_ylabel)
