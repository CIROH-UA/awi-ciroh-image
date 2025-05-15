""" Functions related to testing the fitness of a Clause """
import copy
from typing import Callable

import numpy as np
from teva.base import clause as base_clause
from teva.utilities import teva_logging

def ratio_test(clause: base_clause.Clause) -> bool:
    """ Performs the fitness ratio test which will determine whether a clause is worth keeping in the population.

    pass = target_coverage / total_coverage > target_observations / total_observations

    :param clause: The clause on which to perform the ratio test

    :return: True, if the ratio test is passed; False, otherwise
    """
    target_coverage = clause.target_coverage_count
    target_observations = clause.target_count
    total_coverage = clause.coverage_count
    total_observations = clause.total_observations

    # if this clause covers nothing, it will automatically fail the ratio test
    if total_coverage == 0:
        return False

    target_ratio: float = target_observations / total_observations
    match_ratio: float = target_coverage / total_coverage

    return match_ratio > target_ratio


def calc_fitness(clause: base_clause.Clause, fitness_function: Callable) -> float:
    """ Calculates the fitness of a disjunctive clause using the provided fitness function.

    The fitness function must take the following form:

        fitness_function(number, number, number, number) -> float

    :param clause: The clause whose fitness will be determined
    :param fitness_function: The fitness function to use for determining clause fitness

    :return: The fitness value of the provided clause
    """
    # determine the coverage
    target_coverage = clause.target_coverage_count
    target_observations = clause.target_count
    total_coverage = clause.coverage_count
    total_observations = clause.total_observations

    try:
        fitness = fitness_function(k=target_coverage,
                                   N=total_observations,
                                   K=target_observations,
                                   n=total_coverage)
    except Exception as e:
        raise ValueError("""
        Fitness: Fitness function failed.  
        Ensure it has the correct signature: fitness_function(number, number, number, number) -> float
        """) from e

    return fitness


def sensitivity(clause: base_clause.Clause,
                sensitivity_threshold: float,
                original_fitness: float) -> bool:
    """ Determines the sensitivity of a clause by testing the fitness of the clause after disabling each feature.
    Returns an array of sensitivity values, where each value represents the change in fitness after a feature
    has been removed.  A negative value represents an increase in fitness, where a positive represents a decrease, where
    the fitness is to be minimized.

    :param clause: The conjunctive clause to test
    :param sensitivity_threshold: The threshold of feature sensitivity below which feature sensitivity values must be
    :param original_fitness: The fitness value of the clause before running the ``feature_sensitivity()`` algorithm

    :return: True if the sensitivity test is passed; False otherwise
    """
    # Initialize the sensitivity array
    # NOTE: this array gets initialized to NaN in the Matlab code
    sensitivity_fitnesses = original_fitness * np.ones(len(clause))
    # for every active feature,
    for item_index in range(len(clause)):
        # If not active, we skip
        if not clause.mask[item_index]:
            continue

        # create a copy of the clause and remove the current feature
        temp_clause = copy.deepcopy(clause)
        temp_clause.disable_item(item_index)

        # apply the fitness function on the modified clause
        fitness = clause.fitness

        # if the ratio passes, reduce the original fitness of the current feature by the new fitness
        if ratio_test(clause):
            sensitivity_fitnesses[item_index] -= fitness

        if sensitivity_fitnesses[item_index] > sensitivity_threshold:
            return False

        del temp_clause

    return True


def test_fitness(clause: base_clause.Clause,
                 fitness_function: Callable,
                 fitness_threshold: float,
                 max_order: int,
                 use_sensitivity: bool = False,
                 sensitivity_threshold: float = -np.inf) -> str:
    """ Determine the fitness of a clause and return its expected fate.

        ratio_test & fitness_test & sensitivity_test   -> "archive"
        ratio_test & fitness_test & !sensitivity_test  -> "keep"
        ratio_test & !fitness_test                     -> "keep"
        !ratio_test                                    -> "remove

    :param clause: The clause whose fitness will be determined
    :param fitness_function: The fitness function to use for determining clause fitness
    :param fitness_threshold: The threshold below which the fitness must be to pass the fitness test
    :param max_order: The maximum order that may be stored in the archive
    :param use_sensitivity: If True, sensitivity testing will be used
    :param sensitivity_threshold: The sensitivity threshold for the sensitivity test

    :return: The expected fate of the child - ["archive", "keep", "remove"]
    """

    # determine the number of matching observations, and the number of matching that are of the target class

    # if the ratio test is passed
    if ratio_test(clause):
        # calculate the fitness
        fitness = calc_fitness(clause, fitness_function)
        clause.fitness = fitness

        # if it passes the fitness test, use sensitivity if necessary
        if clause.order() <= max_order and fitness <= fitness_threshold:
            # test that all features pass the sensitivity test.  If sensitivity is not enabled, it always passes
            pass_sensitivity = True
            if use_sensitivity:
                pass_sensitivity = sensitivity(clause=clause,
                                               sensitivity_threshold=sensitivity_threshold,
                                               original_fitness=fitness)
            if pass_sensitivity:
                return "archive"

        return "keep"

    # if it failed the ratio test, remove it
    return "remove"


def test_fitnesses(children: [base_clause.Clause],
                   fitness_function: Callable,
                   fitness_thresholds: list[float],
                   max_order: int,
                   use_sensitivity: bool = False,
                   sensitivity_threshold: float = -np.inf) -> list[str]:
    """ Determines the fitness and the fate of all clauses in a list.  Calls :function:`test_fitness()`

    :param children: The list of children on which to calculate fitness
    :param fitness_function: The fitness function to use for determining clause fitness
    :param fitness_thresholds: The thresholds below which the fitness of each order must be to pass the fitness test
    :param max_order: The maximum order that may be stored in the archive
    :param use_sensitivity: If True, sensitivity testing will be used
    :param sensitivity_threshold: The sensitivity threshold for the sensitivity test

    :return: A list of the expected fates of each child - [["archive", "keep", "remove"]]
    """
    fates: list[str] = []

    for clause in children:
        try:
            thresholds = fitness_thresholds[min(clause.order() - 1, max_order - 1)]
            fates.append(test_fitness(clause=clause,
                                      fitness_function=fitness_function,
                                      use_sensitivity=use_sensitivity,
                                      max_order=max_order,
                                      fitness_threshold=thresholds,
                                      sensitivity_threshold=sensitivity_threshold))
        except IndexError:
            print(f"ERROR: Bad threshold index: {len(fitness_thresholds)} | {min(max_order - 1, clause.order() - 1)}")

    return fates
