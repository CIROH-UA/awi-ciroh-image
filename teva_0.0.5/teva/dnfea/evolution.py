""" Functions for evolving DNF clauses (mutation or crossover) """
import logging

import numpy as np

from teva.dnfea import disjunctive_clause, mutation, crossover
from teva.utilities import teva_math

def _perform_crossover(population_layer: [disjunctive_clause.DisjunctiveClause],
                       dnf_index: int,
                       tournament_size: int,
                       p_xf: float,
                       p_x_alt: list[float],
                       p_u: float,
                       p_i: float) -> disjunctive_clause.DisjunctiveClause:
    """ Chooses between a variety of techniques for selecting mates for crossover with the given DNF.  Then, performs
    crossover between the two mates.

    Using probabilities, one of the following four methods are chosen for mate selection:

        1. Choose at random with a uniform distribution
        2. The DNF that covers the most target observations not covered by the current DNF
        3. The DNF that covers the most target observations that are not covered by the current DNF while minimizing the
         number of new non-target observations covered.
        4. The DNF with the least number of non-target observations that are not covered by the current DNF

    :param population_layer: The list of disjunctive clauses that crossover can be performed on
    :param dnf_index: The index of the disjunctive clause that will be used as the first mate
    :param tournament_size: The number of possible mates that will be considered for selection
    :param p_xf: The chance that simple crossover will be used
    :param p_x_alt: The probability weights of the remaining three crossover algorithms
    :param p_u: The probability of a crossover union
    :param p_i: The probability of a crossover intersection

    :return: A new clause generated through crossover
    """
    potential_mates = teva_math.Rand.RNG.choice(population_layer, size=tournament_size, replace=True)

    mate_a = population_layer[dnf_index]
    mate_b = None

    if teva_math.Rand.RNG.random() > p_xf:
        mate_b = crossover.selection_simple(potential_mates=potential_mates)
    else:
        random_num = teva_math.Rand.RNG.choice(np.arange(3), p=p_x_alt)
        match random_num:
            case 0:
                mate_b = crossover.selection_most_covered(potential_mates=potential_mates,
                                                           mate_a=mate_a)
            case 1:
                mate_b = crossover.selection_most_covered_minimized(potential_mates=potential_mates,
                                                                     mate_a=mate_a)
            case 2:
                mate_b = crossover.selection_least_uncovered(potential_mates=potential_mates,
                                                              mate_a=mate_a)
            case _:
                logging.getLogger("dnfea").error("Evolution: Invalid choice")

    new_child = crossover.crossover(parent_a=mate_a,
                                    parent_b=mate_b,
                                    p_u=p_u,
                                    p_i=p_i)

    return new_child

def _perform_mutation(clause: disjunctive_clause.DisjunctiveClause,
                      p_m: float,
                      p_m_alt: list[float],
                      p_f: float) -> disjunctive_clause.DisjunctiveClause:
    """ Performs a mutation on the given disjunctive clause and returns a new DNF

    :param clause: The clause to use as a mutation parent
    :param p_f: The probability that the standard mutation will be used.  Otherwise, the targeted will be used.
    :param p_m: If standard mutation is selected, the chance that a particular bit will be flipped
    :param p_m_alt: If targeted mutation is selected, the probability weights for each of the four targeted algorithms.
        sum([float, float, float, float]) must equal 1.

    :return: A new DNF generated through mutation
    """
    return mutation.mutate(parent=clause,
                           p_m=p_m,
                           p_f=p_f,
                           p_m_alt=p_m_alt)


def crossover_or_mutate(population_layer: [disjunctive_clause.DisjunctiveClause],
                        observation_table: np.ndarray,
                        num_dnfs: int,
                        p_x: float,
                        p_xf: float,
                        p_m: float,
                        p_u: float,
                        p_i: float,
                        p_m_alt: list[float],
                        p_x_alt: list[float],
                        p_f: float,
                        tournament_size: int) -> list[disjunctive_clause.DisjunctiveClause]:
    """ Either mutates or mates each clause in a given population layer, generating ``num_dnfs`` new clauses.

    :param population_layer: The list of disjunctive clauses to be evolved
    :param observation_table: The observation data table
    :param classifications: The classification of each observation row

    :param p_x: The probability that a crossover will occur instead of a mutation
    :param p_f: The probability that the standard mutation will be used.  Otherwise, the targeted will be used.
    :param p_m: If standard mutation is selected, the chance that a particular bit will be flipped
    :param p_m_alt: If targeted mutation is selected, the probability weights for each of the four targeted algorithms.
        sum([float, float, float, float]) must equal 1.
    :param num_dnfs: The total number of dnfs to create
    :param tournament_size: The number of possible mates that will be considered for selection
    :param p_xf: The chance that simple crossover will be used
    :param p_x_alt: The probability weights of the remaining three crossover algorithms
    :param p_u: The probability of a crossover union
    :param p_i: The probability of a crossover intersection

    :return: A list of newly created clauses
    """
    num_dnfs = min(num_dnfs, len(population_layer))

    if num_dnfs > len(population_layer):
        logging.getLogger("ccea").error("Evolution: 'num_clauses' must be <= to the size of the pop layer")

    if tournament_size <= 0:
        raise ValueError("'tournament_size' must be greater than 0")

    offspring: [disjunctive_clause.DisjunctiveClause] = []

    for dnf_index in range(num_dnfs):

        if population_layer[dnf_index].order() == 0:
            print(population_layer[dnf_index])

        try:
            if teva_math.Rand.RNG.random() < p_x:
                new_child = _perform_crossover(population_layer=population_layer,
                                               dnf_index=dnf_index,
                                               tournament_size=tournament_size,
                                               p_xf=p_xf,
                                               p_x_alt=p_x_alt,
                                               p_i=p_i,
                                               p_u=p_u)
            else:
                new_child = _perform_mutation(population_layer[dnf_index],
                                              p_f=p_f,
                                              p_m=p_m,
                                              p_m_alt=p_m_alt)

            new_child.calc_coverage(observation_table)
            offspring.append(new_child)

        except ValueError as e:
            logging.getLogger("dnfea").error(f"Evolution: {e}")
            raise e

    return offspring
