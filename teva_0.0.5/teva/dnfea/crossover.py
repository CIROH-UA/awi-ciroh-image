""" Functions for handling mating between two DNF clauses """
import numpy as np

from teva.dnfea import disjunctive_clause
from teva.utilities import teva_math


def selection_most_covered(potential_mates: list[disjunctive_clause.DisjunctiveClause],
                           mate_a: disjunctive_clause.DisjunctiveClause) -> disjunctive_clause.DisjunctiveClause:
    """ Selects a mate from the potential mates by choosing the clause that covers the most target observations that
    are not covered by ``mate_a``.

    :param potential_mates: The list of potential mating clauses to choose from
    :param mate_a: The first parent for which a mate will be found
    :return: The mate that has been chosen
    """
    observations_not_covered = ~mate_a.coverage_mask

    # count number of non-covered observations that are covered by each mate
    mate_counts = []
    for mate in potential_mates:
        n_mates = np.sum(observations_not_covered & mate.target_coverage_mask)
        mate_counts.append(n_mates)

    # get best mate
    mate_b = teva_math.argmax(mate_counts)

    return potential_mates[mate_b]


def selection_most_covered_minimized(potential_mates: list[disjunctive_clause.DisjunctiveClause],
                                     mate_a: disjunctive_clause.DisjunctiveClause) \
                                        -> disjunctive_clause.DisjunctiveClause:
    """ Selects a mate from the potential mates by choosing the clause that covers the most target observations that
    are not covered by ``mate_a`` while also covering the least non-target observations that are not covered by
    ``mate_a``.

    :param potential_mates: The list of potential mating clauses to choose from
    :param mate_a: The first parent for which a mate will be found

    :return: The mate that has been chosen
    """
    observations_not_covered = ~mate_a.coverage_mask
    non_target_not_covered = mate_a.nontarget_mask & observations_not_covered

    # count number of non-covered observations that are covered by each mate
    covered_counts = []
    uncovered_counts = []
    for mate in potential_mates:
        n_covered = np.sum(observations_not_covered & mate.target_coverage_mask)
        n_uncovered = np.sum(non_target_not_covered & mate.nontarget_coverage_mask)
        covered_counts.append(n_covered)
        uncovered_counts.append(n_uncovered)

    # get mate with most covered and least uncovered
    mate_b = teva_math.argmax(np.array(covered_counts) - np.array(uncovered_counts))

    return potential_mates[mate_b]


def selection_least_uncovered(potential_mates: list[disjunctive_clause.DisjunctiveClause],
                              mate_a: disjunctive_clause.DisjunctiveClause) -> disjunctive_clause.DisjunctiveClause:
    """ Selects a mate from the potential mates by choosing the clause that covers the least non-target clauses
    not covered by ``mate_a``.

    :param potential_mates: The list of potential mating clauses to choose from
    :param mate_a: The first parent for which a mate will be found

    :return: The mate that has been chosen
    """
    observations_not_covered = ~mate_a.coverage_mask
    non_target_not_covered = mate_a.nontarget_mask & observations_not_covered

    # count number of non-covered observations that are covered by each mate
    mate_counts = []
    for mate in potential_mates:
        n_mates = np.sum(non_target_not_covered & mate.nontarget_coverage_mask)
        mate_counts.append(n_mates)

    # get best mate
    mate_b = teva_math.argmin(mate_counts)

    return potential_mates[mate_b]


def selection_simple(potential_mates: list[disjunctive_clause.DisjunctiveClause]) \
        -> disjunctive_clause.DisjunctiveClause:
    """ Selects a mate from the list of potential mates with the best fitness.

    :param potential_mates: The list of potential mating clauses

    :return: The potential mate with the best fitness
    """
    # NOTE: The potential mates should already be sorted by fitness
    # get the mate with the best fitness
    mate = potential_mates[0]

    return mate


def _mask_union(parent_a: disjunctive_clause.DisjunctiveClause,
                parent_b: disjunctive_clause.DisjunctiveClause) -> np.ndarray:
    """ Performs the simple union of the parent masks

    :param parent_a: The first mating parent clause
    :param parent_b: The second mating parent clause

    :return: An unioned conjunctive clause mask
    """
    return parent_a.mask | parent_b.mask


def _mask_intersection(parent_a: disjunctive_clause.DisjunctiveClause,
                       parent_b: disjunctive_clause.DisjunctiveClause) -> np.ndarray:
    """ Performs the simple intersection of the parent masks

    :param parent_a: The first mating parent clause
    :param parent_b: The second mating parent clause

    :return: An intersected conjunctive clause mask
    """
    return parent_a.mask & parent_b.mask


def _mask_uniform(parent_a: disjunctive_clause.DisjunctiveClause,
                  parent_b: disjunctive_clause.DisjunctiveClause,
                  feature_selection_ratio: float = 0.5) -> np.ndarray:
    """ Uses a uniform random distribution to select which features will be used for crossover.

    If rand() < ``feature_selection_ratio``, the bit will be selected from a, otherwise it will be selected from b.

    :param parent_a: The first mating parent clause
    :param parent_b: The second mating parent clause
    :param feature_selection_ratio: The selection ratio between the two parents; a higher number will favor a, a lower
        number will favor b.

    :return: A uniformly crossed mask
    """
    # create inverse random boolean arrays representing the clause bits to use from each parent
    parent_a_clauses: np.ndarray = teva_math.Rand.RNG.random(size=parent_a.mask.shape[0]) < feature_selection_ratio
    parent_b_clauses: np.ndarray = ~parent_a_clauses

    # the features for parent b are the inverse of those for parent a
    parent_a_selection = np.nonzero(parent_a_clauses)[0]
    parent_b_selection = np.nonzero(parent_b_clauses)[0]

    # set all unused bits in both masks temporarily to false
    mask_a = parent_a.mask.copy()
    mask_a[~parent_a_selection] = False
    mask_b = parent_b.mask.copy()
    mask_b[~parent_b_selection] = False

    # the union of both parent masks for each of their selected features
    offspring_mask = mask_a | mask_b

    # if the new mask is equal to either parent's mask, or contains no features, swap the parent selections
    if np.all(offspring_mask == parent_a.mask) \
            or np.all(offspring_mask == parent_b.mask) \
            or offspring_mask.sum() == 0:
        # do the same process again, but inverting the parent selections
        mask_a = parent_a.mask.copy()
        mask_a[~parent_b_selection] = False
        mask_b = parent_b.mask.copy()
        mask_b[~parent_a_selection] = False
        offspring_mask = mask_a | mask_b

    return offspring_mask


def crossover(parent_a: disjunctive_clause.DisjunctiveClause,
              parent_b: disjunctive_clause.DisjunctiveClause,
              p_u: float,
              p_i: float) -> disjunctive_clause.DisjunctiveClause:
    """ Uses two parent clauses to create a new clause either through union or intersection

    :param parent_a: The first parent
    :param parent_b: The second parent
    :param p_u: The probability that the parent masks will be unioned
    :param p_i: the probability that the parent masks will be intersected

    :return: The child created from crossover
    """
    # get a random number for operation selection
    rnd = teva_math.Rand.RNG.random()
    # generate a new mask using one of the crossover operations
    new_mask: np.ndarray
    if rnd < p_u:
        new_mask = _mask_uniform(parent_a, parent_b)
    elif rnd < p_u + p_i:
        in_mask = _mask_intersection(parent_a, parent_b)
        # if the resulting mask will be empty, use uniform crossover instead
        if sum(in_mask) > 1:
            new_mask = in_mask
        else:
            new_mask = _mask_uniform(parent_a, parent_b)
    else:
        new_mask = _mask_uniform(parent_a, parent_b)

    # create a new disjunctive clause with the crossed-over mask
    new_clause = disjunctive_clause.DisjunctiveClause(cc_mask=new_mask,
                                                      cc_clauses=parent_a.items,
                                                      classification=parent_a.classification,
                                                      age=max(parent_a.age, parent_b.age))

    return new_clause
