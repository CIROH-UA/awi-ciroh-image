""" Functions related to testing the fitness of a DNF """
from typing import Callable

import numpy as np

from teva.dnfea import disjunctive_clause
from teva.base import fitness as base_fitness


def calc_fitness(clause: disjunctive_clause.DisjunctiveClause, fitness_function: Callable,) -> float:
    """ Calculates the fitness of a disjunctive clause using the provided fitness function

    The fitness function must take the following form:

        fitness_function(number, number, number, number) -> float

    :param clause: The clause whose fitness will be determined
    :param fitness_function: The fitness function to use for determining clause fitness

    :return: The fitness value of the provided clause
    """
    return base_fitness.calc_fitness(clause=clause, fitness_function=fitness_function)


def child_fitness(clause: disjunctive_clause.DisjunctiveClause,
                  fitness_function: Callable,
                  fitness_threshold: float,
                  max_order: int,
                  use_sensitivity: bool = False,
                  sensitivity_threshold: float = -np.inf) -> str:
    """ Determine the fitness of a DNF clause and return its expected fate.

        ratio_test & fitness_test   -> "archive"
        ratio_test & !fitness_test  -> "keep"
        !ratio_test                 -> "remove

    :param clause: The clause whose fitness will be determined
    :param fitness_function: The fitness function to use for determining clause fitness
    :param fitness_threshold: The threshold below which the fitness must be to pass the fitness test
    :param max_order: The maximum order that may be stored in the archive
    :param use_sensitivity: If True, sensitivity testing will be used
    :param sensitivity_threshold: The sensitivity threshold for the sensitivity test

    :return: The expected fate of the child - ["archive", "keep", "remove"]
    """
    return base_fitness.test_fitness(clause=clause,
                                     fitness_function=fitness_function,
                                     fitness_threshold=fitness_threshold,
                                     max_order=max_order,
                                     use_sensitivity=use_sensitivity,
                                     sensitivity_threshold=sensitivity_threshold)


def children_fitness(children: [disjunctive_clause.DisjunctiveClause],
                     fitness_function: Callable,
                     fitness_thresholds: list[float],
                     max_order: int,
                     use_sensitivity: bool = False,
                     sensitivity_threshold: float = -np.inf) -> list[str]:
    """ Determines the fitness and the fate of all DNF clauses in a list.  Calls :function:`child_fitness()`

    :param children: The list of children on which to calculate fitness
    :param fitness_function: The fitness function to use for determining clause fitness
    :param fitness_thresholds: The thresholds below which the fitness of each order must be to pass the fitness test
    :param max_order: The maximum order that may be stored in the archive
    :param use_sensitivity: If True, sensitivity testing will be used (Not currently implemented for DNF)
    :param sensitivity_threshold: The sensitivity threshold for the sensitivity test (Not currently implemented for DNF)

    :return: A list of the expected fates of each child - [["archive", "keep", "remove"]]
    """
    return base_fitness.test_fitnesses(children=children,
                                       fitness_function=fitness_function,
                                       fitness_thresholds=fitness_thresholds,
                                       max_order=max_order,
                                       use_sensitivity=use_sensitivity,
                                       sensitivity_threshold=sensitivity_threshold)
