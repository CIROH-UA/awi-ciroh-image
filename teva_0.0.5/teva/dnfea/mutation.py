""" Functions used by the DNF for mutating clauses """
import logging

import numpy as np

from teva.dnfea import disjunctive_clause
from teva.utilities import teva_math


def _swap_random_bit(mask: np.ndarray, activate: bool) -> np.ndarray:
    """ Activates a random inactive bit, or deactivates a random active bit.

    :param mask: The mask on which to perform the operation
    :param activate: If True, an inactive bit will be activated.  If False, an active bit will be deactivated.

    :return: The modified mask
    """
    if activate and np.all(mask == activate):
        logging.getLogger("dnfea").error("There are no inactive bits to activate")
        return mask
    if not activate and np.all(mask == activate):
        logging.getLogger("dnfea").error("There are no active bits to deactivate")
        return mask

    new_mask = mask.copy()
    # get all indices that are not in the desired state
    indices = np.nonzero(mask != activate)[0]
    # choose a random index from that list
    random_index = teva_math.Rand.RNG.choice(indices)
    # activate or deactivate the value at that index based in user input
    new_mask[random_index] = activate

    return new_mask


def _targeted_enable_uncovered(parent: disjunctive_clause.DisjunctiveClause) -> np.ndarray:
    """Enable the CC that covers the most target observations that are not covered by the parent

    :param parent: The parent DNF.

    :return: the parent mask with the aforementioned CC enabled.
    """
    observations_not_covered = parent.target_mask & ~parent.coverage_mask

    sums = []
    for cc in np.array(parent.items):
        # get the number of target obs not covered by the dnf, that are covered by the cc
        n_obs = np.sum(cc.target_coverage_mask & observations_not_covered)
        sums.append(n_obs)

    if len(sums) == 0:
        logging.getLogger("dnfea").error("Mutation: Sums list is empty")
        return parent.mask.copy()

    max_sum_index = teva_math.argmax(sums)

    child_mask = parent.mask.copy()
    child_mask[max_sum_index] = True
    return child_mask


def _targeted_enable_uncovered_min(parent: disjunctive_clause.DisjunctiveClause) -> np.ndarray:
    """Enable the CC that covers the most target observations, and least non-target observations covered by the DNF

    :param parent: The parent DNF

    :return: the parent mask with the aforementioned CC enabled.
    """
    # get all the target observations not covered by the parent
    target_observations_not_covered = parent.target_mask & ~parent.coverage_mask
    # get all non-target observations not covered by the parent
    non_target_observations_not_covered = ~parent.target_mask & ~parent.coverage_mask

    sums_target = []
    sums_non_target = []
    for cc in np.array(parent.items):
        sums_target.append(np.sum(cc.target_coverage_mask & target_observations_not_covered))
        sums_non_target.append(np.sum(cc.nontarget_coverage_mask & non_target_observations_not_covered))

    # Compute the differences between number of target observations covered and non-target observations covered for
    # each CC
    diff = np.array(sums_target) - np.array(sums_non_target)

    # Mask indicating which CCs cover the most target observations and least non-target observations
    if diff.shape[0] == 0:
        logging.getLogger('dnfea').error("Mutation: The sum array has no elements.")
        return parent.mask.copy()

    max_mask = diff == np.max(diff)
    indices_satisfying_condition = np.flatnonzero(max_mask)

    if indices_satisfying_condition.shape[0] > 1:
        max_sum_index = teva_math.Rand.RNG.choice(indices_satisfying_condition)
    else:
        max_sum_index = indices_satisfying_condition[0]

    child_mask = parent.mask.copy()
    child_mask[max_sum_index] = True
    return child_mask


def _targeted_disable_worst(parent: disjunctive_clause.DisjunctiveClause) -> np.ndarray:
    """ Disable the CC that covers the least unique target observations

    :param parent: The parent DNF

    :return: the parent mask with the aforementioned CC enabled.
    """
    sums = []
    for cc in np.array(parent.items):
        # get the number of target obs not covered by the dnf, that are covered by the cc
        sums.append(cc.target_coverage_count)

    min_sum_index = teva_math.argmin(sums)

    child_mask = parent.mask.copy()
    child_mask[min_sum_index] = False

    return child_mask


def _targeted_disable_mismatched(parent: disjunctive_clause.DisjunctiveClause) -> np.ndarray:
    """ Disable the CC with the most non-target observations

    :param parent: The parent DNF

    :return: the parent mask with the aforementioned CC enabled.
    """
    sums = []
    for cc in np.array(parent.items):
        # get the number of target obs not covered by the dnf, that are covered by the cc
        sums.append(cc.nontarget_coverage_count)

    max_sum_index = teva_math.argmax(sums)
    child_mask = parent.mask.copy()
    child_mask[max_sum_index] = True
    return child_mask


def _targeted_mutation(parent: disjunctive_clause.DisjunctiveClause,
                       p_m_alt: list[float]) -> np.ndarray:
    """ Creates a mutated mask from a parent disjunctive clause using observation data to influence the selection
        of mask bits.

    The different targeted mutation methods have a chance of being used based on ``p_m_alt`` where all four have an
    equal chance by default, and are outlined as follows:

       1) select the CC that covers the most target observations not covered by the DNF
       2) select the CC that covers the most target observations while minimizing the number of new non target
           observations covered
       3) remove the CC that covers the least unique target observations;
       4) remove the CC that has most non-target observations that are only covered by the CC

    :param parent: The parent clause for the mutation.  This clause will not be directly modified but will instead be
        used as a seed for the creation of a new, mutated mask.
    :param p_m_alt: The chance of each of the four mutations occurring: sum([float, float, float, float]) must equal 1.

    :return: A mutated disjunctive clause mask
    """
    p_m_alt = p_m_alt.copy()
    # divide the number of target observations covered by the parent by the number of observations with the target class
    parent_coverage = parent.target_coverage_count / parent.target_count

    if parent.coverage_mask.shape[0] == 0:
        logging.getLogger("dnfea").error("Mutation: Parent coverage is empty")
        parent_accuracy = 0
    else:
        # Calculate the ratio of target observations covered by the DNF to total observations covered by the DNF
        parent_accuracy = parent.target_coverage_count / parent.coverage_count

    # NOTE: Ask if this makes sense for a DNF containing a single CC, and that CC has 100% coverage and 100% accuracy
    # if accuracy and coverage are 100%, do not choose choice 0 or 1
    if parent_accuracy == 1.0 and parent_coverage == 1.0:
        p_m_alt[0] = 0
        p_m_alt[1] = 0
    # if just coverage is 100%, only choose 1
    elif parent_coverage == 1.0:
        p_m_alt[0] = 0
        if parent.order() == 1:
            p_m_alt[2] = 0
            p_m_alt[3] = 0
    elif parent.order() == 1:
        p_m_alt[2] = 0
        p_m_alt[3] = 0

    # if p_m_alt is empty or all zero, something went wrong in the previous if statement
    if np.sum(p_m_alt) == 0:
        logging.getLogger("dnfea").error("Mutation: p_m_alt is empty")

    # normalize the p_m_alt array to sum to 1.0
    p_m_alt = np.array(p_m_alt) / np.sum(p_m_alt)

    # randomly choose an algorithm using p_m_alt as weights
    rnd = np.random.choice(np.arange(4), p=p_m_alt)
    match rnd:
        case 0:
            offspring_mask = _targeted_enable_uncovered(parent=parent)
        case 1:
            offspring_mask = _targeted_enable_uncovered_min(parent=parent)
        case 2:
            offspring_mask = _targeted_disable_worst(parent=parent)
        case 3:
            offspring_mask = _targeted_disable_mismatched(parent=parent)
        case _:
            raise ValueError("Bad index.")

    if np.sum(offspring_mask) == 0:
        logging.getLogger("dnfea").error("Mutation: Mutated offspring mask has no enabled bits")

    return offspring_mask


def _standard_mutation(parent: disjunctive_clause.DisjunctiveClause,
                       p_m: float) -> np.ndarray:
    """Creates a mutated mask from a parent disjunctive clause by flipping bits in the mask

    :param parent: The parent clause for the mutation.  This clause will not be directly modified but will instead be
        used as a seed for the creation of a new, mutated mask.
    :param p_m: The chance that a particular bit in the array will be flipped

    :return: A mutated disjunctive clause mask
    """
    # for each bit in the array, determine whether it will be enabled or disabled
    flip_bits: np.ndarray = teva_math.Rand.RNG.random(size=len(parent)) < p_m
    # if the flip_bits array ends up being all False, ensure that at least one bit will be flipped
    if sum(flip_bits) == 0:
        flip_bits[0] = True

    # get the indices of the bits to flip and copy the parent mask
    flip_bits_indices = np.nonzero(flip_bits)
    offspring_mask = parent.mask.copy()

    # invert all the bits at the indices of the flip_bit mask
    offspring_mask[flip_bits_indices] = ~offspring_mask[flip_bits_indices]

    # if the offspring mask is empty, we need to randomly decide whether to keep bits
    if sum(offspring_mask) == 0:
        offspring_mask = parent.mask.copy()
        # if there's one or less enabled clauses in the mask, activate a random bit, otherwise deactivate one
        if sum(offspring_mask) <= 1:
            # activate a random bit
            offspring_mask = _swap_random_bit(offspring_mask, activate=True)
        else:
            # deactivate a random bit
            offspring_mask = _swap_random_bit(offspring_mask, activate=False)

    return offspring_mask


def mutate(parent: disjunctive_clause.DisjunctiveClause,
           p_f: float,
           p_m: float,
           p_m_alt: list[float]) -> disjunctive_clause.DisjunctiveClause:
    """ Generates a new disjunctive clause by mutating the cc mask of a given parent clause, either by standard or
    targeted mutation.

    Standard Mutation: :function:`_standard_mutation()` - Flips bits at random
    Targeted Mutation: :function:`_targeted_mutation()` - Flips bits using one of four algorithms which take into
        account the coverage of clauses in the observation matrix

    :param parent: The parent clause for the mutation.  This clause will not be directly modified but will instead be
        used as a seed for the creation of a new, mutated clause.
    :param p_f: The probability that the standard mutation will be used.  Otherwise, the targeted will be used.
    :param p_m: If standard mutation is selected, the chance that a particular bit will be flipped
    :param p_m_alt: If targeted mutation is selected, the probability weights for each of the four targeted algorithms.
        sum([float, float, float, float]) must equal 1.
    :return:
    """
    # choose either standard or targeted mutation and get a mutated mask
    if teva_math.Rand.RNG.random() > p_f:
        offspring_mask = _standard_mutation(parent, p_m)
    else:
        offspring_mask = _targeted_mutation(parent=parent,
                                            p_m_alt=p_m_alt)

    # use data from the parent to create a new DNF with the mutated mask
    return disjunctive_clause.DisjunctiveClause(offspring_mask,
                                                parent.items,
                                                parent.classification,
                                                age=parent.age)
