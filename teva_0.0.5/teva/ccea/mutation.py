""" Functions for the mutation of a CC clause """
import copy
import numpy as np

from teva.ccea import conjunctive_clause
from teva.base import feature
from teva.utilities import teva_math

def _select_features_selective(clause: conjunctive_clause.ConjunctiveClause):
    """ THIS FUNCTION WAS NOT USED IN THE ORIGINAL MATLAB CODE, AND HAS NOT YET BEEN IMPLEMENTED

    :param clause:
    :return:
    """
    raise NotImplementedError


def _select_features_random(clause: conjunctive_clause.ConjunctiveClause,
                            p_m: float) -> np.array:
    """ Selects features to be mutated at random, where each feature is selected with a discrete uniform probability of
    :math:`P_m`.

    :param clause: The clause in which selection should occur
    :param p_m: probability that a feature will be selected for mutation

    :return: An array of indices in the feature's :class:`FeatureDomain` that have been selected.
    """
    mutate_mask = np.array(teva_math.Rand.RNG.random(len(clause)) < p_m, dtype=bool)
    # if mutate mask is all False, ensure at least one feature is mutated by setting a random index to True
    if np.all(~mutate_mask):
        random_index = teva_math.Rand.RNG.integers(0, len(clause))
        mutate_mask[random_index] = True
    return np.nonzero(mutate_mask)[0]


def select_features(clause: conjunctive_clause.ConjunctiveClause,
                    selective: bool,
                    p_m: float) -> np.array:
    """ Selects features to be mutated based on :math:`P_m` and whether selective mutation is enabled.

    Selective Mutation is not yet implemented.

    :param clause: The clause in which selection should occur
    :param selective: If True, the selective mutation algorithm will be used
    :param p_m: probability that a feature will be selected for mutation
    :return: An array of indices in the feature's :class:`FeatureDomain` that have been selected.
    """
    if not selective:
        return _select_features_random(clause, p_m)

    return _select_features_selective(clause)


def _mutate_categorical(feat: feature.CategoricalFeature) -> feature.CategoricalFeature:
    """ Mutates a categorical feature by adding or removing values from its value array.

    Since categorical feature values are distinct, there is no inherent order to their list, thus there are no
    explicit bounds on the value array.  Therefore, categorical features are mutated by either removing a value, or
    adding a new one from the feature's domain.

    :param feat: The feature to be mutated

    :return: The mutated feature
    """
    new_feat = copy.copy(feat)
    # determine if we have a boundary condition, and therefore whether we need to disable adding or remove categories
    # do not remove if there is only one value

    if len(new_feat) == 1:
        new_feat.add_random_value()
    elif len(new_feat) == len(new_feat.feature_domain):
        new_feat.remove_random_value()
    else:
        if np.random.rand() < 0.5:
            new_feat.add_random_value()
        else:
            new_feat.remove_random_value()

    return new_feat


def _mutate_numerical(feat: feature.NumericalFeature) -> feature.NumericalFeature:
    """ Mutates a numerical feature by changing the bounds of its values.

    As numerical features are sorted, mutating its value array involves choosing either the lower or upper bound of
    its value array, and either increasing or decreasing it based on various factors.  In general, all of these choices
    use a uniform distribution with randomly chosen values, although boundary cases are treated as needed to ensure
    random choice.

    :param feat: The feature to be mutated

    :return: The mutated feature
    """
    bounds = feat.get_bounds()
    new_feat = copy.copy(feat)

    if len(bounds) == 0:
        raise ValueError("Input feature has no bounds")

    # get lists of potential indices
    potential_lower = np.arange(0, bounds[1])
    potential_upper = np.arange(bounds[0], feat.feature_domain.domain.shape[0])

    # remove the upper and lower bounds indices from their own potentials
    potential_lower = np.delete(potential_lower, bounds[0])
    # TODO: This occasionally throws an index error
    potential_upper = np.delete(potential_upper, bounds[1] - (potential_lower.shape[0] + 1))

    has_lower = potential_lower.shape[0] > 0
    has_upper = potential_upper.shape[0] > 0

    change = ""
    # determine which bound to modify
    if has_lower and not has_upper:
        change = "lower"
    elif not has_lower and has_upper:
        change = "upper"
    # of both have potentials, choose randomly
    elif has_lower and has_upper:
        change = "lower" if teva_math.Rand.RNG.random() > 0.5 else "upper"

    # pick a random index from the potential index arrays
    if change == "lower":
        new_index = teva_math.Rand.RNG.choice(potential_lower, size=1)
        new_feat.set_lower_bound(new_index)
    elif change == "upper":
        new_index = teva_math.Rand.RNG.choice(potential_upper, size=1)
        new_feat.set_upper_bound(new_index)

    return new_feat


def mutate_feature(feat: feature.Feature) -> feature.Feature:
    """ Mutates a feature according to its ``feature_type``

    If the ``feature_type`` of ``feat`` is numerical, use `mutate_numerical()`.

    If the ``feature_type`` of ``feat`` is categorical, use `mutate_categorical()`.

    :param feat: The feature to be mutated

    :raises: ValueError if the provided feature is not a :class:`NumericalFeature` or :class:`CategoricalFeature`
    :return: The mutated feature
    """
    mutated = copy.copy(feat)
    #
    # mutated = copy.deepcopy(feat)
    # mutated = feature.NumericalFeature(feat.feature_set, feat.feature_type, feat.feature_domain)

    # if numerical, modify the bounds of possible indices
    if isinstance(mutated, feature.NumericalFeature):
        return _mutate_numerical(mutated)

    # if categorical, use permutations of indices
    if isinstance(mutated, feature.CategoricalFeature):
        return _mutate_categorical(mutated)

    raise ValueError("Provided feature is neither of type `NumericalFeature` or `CategoricalFeature`.")


def mutate_clause(clause: conjunctive_clause.ConjunctiveClause,
                  p_m: float,
                  p_wc: float,
                  selective: bool = False) -> conjunctive_clause.ConjunctiveClause:
    """ Mutates a chosen clause, using the probabilities to determine where changes will be made, and returns the
    mutated clause.

    The algorithm is as follows:

        1. Choose m features to mutate, with a discrete probability of :math:`P_m`
        2. For each chosen feature, enable it if it has been disabled.
            a. If a feature has been disabled, remove its values
        3. For each chosen feature, if it is enabled, disable it with a discrete probability of :math:`P_{wc}`
            a. If a feature has been newly enabled, initialize its values
        4. For each of the chosen features that is now enabled, mutate its values in a method appropriate to its
         ``feature_type``.
            a. Numerical features will have the range of their values either reduced or increased on one of the tails
            b. Categorical features will have a value randomly added or removed

    :param clause: The clause to mutate
    :param p_m: probability that a feature will be selected for mutation
    :param p_wc: probability that a feature selected from mutation will be removed from the clause (become a wild-card)
    :param selective: If True, the selective mutation algorithm will be used [NOT YET IMPLEMENTED]

    :return: A mutated copy of the conjunctive clause
    """
    offspring = copy.copy(clause)

    # select feature indices to mutate
    feature_indices = select_features(offspring, selective, p_m)

    # select which features will be deactivated
    random_values = teva_math.Rand.RNG.random(size=feature_indices.shape[0])
    deactivate_mask = np.array(random_values < p_wc, dtype=bool)
    # get indices of features to deactivate
    deactivate_indices = np.nonzero(feature_indices[deactivate_mask])[0]

    # create a temporary mask with deactivated inactive features
    deactivated_mask = offspring.mask.copy()
    deactivated_mask[deactivate_indices] = False

    if np.all(~deactivated_mask) and deactivate_indices.shape[0] == 0:
        raise ValueError(f"The input clause has an empty feature mask: {clause.mask}")

    # if the deactivated mask is empty, reactivate a random feature from the inactive mask
    if np.all(~deactivated_mask) and deactivate_indices.shape[0] > 0:
        random_index = teva_math.Rand.RNG.integers(0, deactivate_indices.shape[0])
        deactivated_mask[deactivate_indices[random_index]] = True

    # set the offspring's feature mask to the deactivated mask
    offspring.mask = deactivated_mask
    # ensure every feature in the offspring that is not in the inactive mask is activated
    offspring.mask[feature_indices[~deactivate_mask]] = True

    # disable or enable features
    for i in range(offspring.mask.shape[0]):
        # for every disabled feature, nullify the feature definition in the feature array
        if not offspring.mask[i]:
            offspring.items[i] = None
        # for every enabled feature, if it has no feature values, initialize it from its feature domain
        elif offspring.items[i] is None:
            offspring.items[i] = offspring.feature_domains[i].init_feature()

        # if at this point the feature is enabled, mutate it
        if offspring.mask[i]:
            offspring.items[i] = mutate_feature(offspring.items[i])

    return offspring
