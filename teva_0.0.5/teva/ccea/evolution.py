""" Evolution functions for CCEA """
from typing import Callable
import traceback
import logging

import numpy as np

from teva.ccea import conjunctive_clause, mutation, crossover, fitness
from teva.utilities import teva_math

def _perform_crossover(population_layer: [conjunctive_clause.ConjunctiveClause],
                       clause_index: int,
                       tournament_size: int,
                       feature_selection_ratio: float,
                       fitness_function: Callable) -> conjunctive_clause.ConjunctiveClause | None:
    # find a suitable mate
    potential_mate_population = np.delete(np.arange(len(population_layer)), clause_index)

    if potential_mate_population.shape[0] == 0:
        logging.getLogger("ccea").error("No potential mates")
        return None

    # get `tournament_size` potential mates from the population
    potential_mates = teva_math.Rand.RNG.choice(potential_mate_population, size=tournament_size, replace=True)

    # apply the fitness function over all potential mates to create an array of fitness values
    fitness_vec = np.vectorize(lambda clause: fitness.calc_fitness(clause=clause,
                                                                   fitness_function=fitness_function))
    fitness_values = fitness_vec(np.array(population_layer)[potential_mates])

    # find the minimum fitness values and the corresponding indices of those mates
    mate_index = potential_mates[teva_math.argmin(fitness_values)]

    return crossover.mate_clauses(population_layer[clause_index], population_layer[mate_index], feature_selection_ratio)


def _perform_mutation(population_layer: [conjunctive_clause.ConjunctiveClause],
                      clause_index: int,
                      p_m: float,
                      p_wc: float,
                      selective_mutation: bool = False) -> conjunctive_clause.ConjunctiveClause:
    return mutation.mutate_clause(population_layer[clause_index], p_m=p_m, p_wc=p_wc, selective=selective_mutation)


def crossover_or_mutate(population_layer: [conjunctive_clause.ConjunctiveClause],
                        observation_table: np.ndarray,
                        classifications: np.ndarray,
                        num_clauses: int,
                        p_x: float,
                        p_m: float,
                        p_wc: float,
                        tournament_size: int,
                        fitness_function: Callable,
                        selective_mutation: bool = False,
                        feature_selection_ratio: float = 0.5) -> list[conjunctive_clause.ConjunctiveClause]:
    """ Performs either crossover or mutation on ``num_clauses`` clauses in the population layer with a probability
    of ``p_x`` :math:`(P_x)`.

    :param population_layer: The total population of the current age layer
    :param observation_table: The matrix of observation data
    :param classifications: The classification of each observation row in the observation table
    :param num_clauses: The number of conjunctive clauses that will undergo crossover or mutation.
    :param p_x: (:math:`P_x`) The discrete uniform probability that a clause will use crossover.  Otherwise, it will
        use mutation.
    :param p_m: (:math:`P_m`) The probability that a feature will be selected for mutation
    :param p_wc: (:math:`P_{wc}`) The probability that a feature selected from mutation will be removed from the clause
        (become a wild-card)
    :param tournament_size: The number of potential mates to test when crossing, with replacement
    :param fitness_function: The fitness function for conjunctive clauses to be used in mate selection
    :param selective_mutation: If True, the selective mutation algorithm will be used [NOT YET IMPLEMENTED]
    :param feature_selection_ratio:

    :return: A list of any newly created children, whether through crossover or mutation
        If ``return_counts`` is True, also returns the number of crossed and number of mutated offspring
    """
    num_clauses = min(num_clauses, len(population_layer))

    if num_clauses > len(population_layer):
        logging.getLogger("ccea").error("Evolution: 'num_clauses' must be <= to the size of the population layer")
        # raise ValueError("'num_clauses' must be less than or equal to the size of the population layer")

    if tournament_size <= 0:
        raise ValueError("'tournament_size' must be greater than 0")

    offspring: [conjunctive_clause.ConjunctiveClause] = []

    # NOTE: This uses the first `num_to_evolve` instead of a random selection; check this is what we want
    for clause_index in range(num_clauses):

        if population_layer[clause_index].order() == 0:
            print(population_layer[clause_index])

        try:
            if teva_math.Rand.RNG.random() < p_x:
                new_child = _perform_crossover(population_layer,
                                               clause_index,
                                               tournament_size,
                                               feature_selection_ratio,
                                               fitness_function=fitness_function)
            else:
                new_child = _perform_mutation(population_layer,
                                              clause_index,
                                              p_m=p_m,
                                              p_wc=p_wc,
                                              selective_mutation=selective_mutation)

            if new_child is not None:
                new_child.calc_coverage(observation_table, classifications)
                offspring.append(new_child)

        except ValueError as e:
            logging.getLogger("ccea").error(f"Evolution: {e}")
            traceback.print_exc()

    return offspring
