""" Crossover functions for CCEA """
import numpy as np
from teva.ccea import conjunctive_clause
from teva.utilities import teva_math


def mate_clauses(parent_a: conjunctive_clause.ConjunctiveClause,
                 parent_b: conjunctive_clause.ConjunctiveClause,
                 feature_selection_ratio: float = 0.5) -> conjunctive_clause.ConjunctiveClause:
    """ Uses two parent clauses to create a child clause by selecting ``feature_selection_ratio * 100`` percent of
    features from ``parent_a`` at random, and selecting the remaining features from ``parent_b``.

    Selected features will not be modified, and the age of the child will the age of the oldest parent.

    :param parent_a: The first parent clause, who features will be selected from with a probability
        of ``feature_selection_ratio``
    :param parent_b: The second parent clause, who remaining features will be selected from, ensuring that all features
        have been selected from either ``parent_a`` or ``parent_b``
    :param feature_selection_ratio: The ratio that determines the balance of feature selection between the two parents.
        A low value will favor ``parent_b``, and a high value will favor ``parent_a``.

    :return: A newly created conjunctive clause with its features extracted from the two parents, and its age determined
        by the age of the oldest parent.
    """
    # create a random mask of features to be assigned to parent a
    feature_selection_mask = np.array(
        teva_math.Rand.RNG.random(parent_a.mask.shape[0]) < feature_selection_ratio, dtype=bool)

    # the features for parent b are the inverse of those for parent a
    parent_a_selection = np.nonzero(feature_selection_mask)[0]
    parent_b_selection = np.nonzero(~feature_selection_mask)[0]

    mask_a = parent_a.mask.copy()
    mask_a[~parent_a_selection] = False

    mask_b = parent_b.mask.copy()
    mask_b[~parent_b_selection] = False

    # the union of both parent masks for each of their selected features
    offspring_mask = mask_a | mask_b

    # if the new mask is equal to either parent's mask, or contains no features, swap the parent selections
    if (
        np.all(offspring_mask == parent_a.mask) or
        np.all(offspring_mask == parent_b.mask) or
        offspring_mask.sum() == 0
    ):
        mask_a = parent_a.mask.copy()
        mask_a[~parent_b_selection] = False
        mask_b = parent_b.mask.copy()
        mask_b[~parent_a_selection] = False
        offspring_mask = mask_a | mask_b

        # offspring_mask = parent_a.mask[parent_b_selection] & parent_b.mask[parent_a_selection]

    offspring_features = []
    for i in range(mask_a.shape[0]):
        if mask_a[i]:
            offspring_features.append(parent_a.items[i])
        else:
            offspring_features.append(parent_b.items[i])

    for i in range(offspring_mask.shape[0]):
        # for every disabled feature, nullify the feature definition in the feature array
        if not offspring_mask[i]:
            offspring_features[i] = None

    # use the generated arrays to create a new conjunctive clause
    offspring = conjunctive_clause.ConjunctiveClause(feature_mask=offspring_mask,
                                                     feature_domains=parent_a.feature_domains,
                                                     features=offspring_features,
                                                     classification=parent_a.classification,
                                                     age=max(parent_a.age, parent_b.age))

    return offspring
