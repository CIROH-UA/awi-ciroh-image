""" Conjunctive Clause (CC) """
from __future__ import annotations

import numpy as np
from teva.base import feature
from teva.base import clause as base_clause

class ConjunctiveClause(base_clause.Clause):
    """ A conjunctive clause represents a set of features that describe a data class.  This clause is mutated and
    modified over the course of the evolutionary algorithm to change the possible values that each feature comprises,
    as well as which features are used for determining the truth of the statement.

    A conjunctive clause could be considered a type of function, where the input to the function is an array of
    observed values for each feature, as well as the class it is intended to represent, and the return value is the
    truth of whether all values fit the features present.  It is considered true if for each feature that is enabled in
    this clause, the corresponding input observation value is contained within the feature's ``feature_set``.  If a
    feature is not enabled, any observation value is considered valid for that feature, making it a Wild-card.

    A clause is constructed of two arrays, a ``feature_mask``, which is a boolean array that determines whether a
    feature is enabled, and an array of features, which contain the features and their acceptable values.

    An example clause with 5 domain features:
        .. math::
            classname = [0, 1, 2] \\wedge [0.1, 0.3, 0.4] \\wedge [0, 1] \\wedge [a, z, f] \\wedge [uno, tres, dos]

            CC_mask = [0, 0, 1, 1, 0]

            CC_values = null \\wedge null \\wedge [0, 1] \\wedge [a, f] \\wedge null


    A matching observation (the observation is of the class):
        .. math::
            classname = 5, 0, 1, f, tres


    A non-matching observation (the observation is not of the class):
        .. math::
            classname = 2, 0.1, 2, z, quatro

    :param feature_mask: A boolean array representing whether a particular feature is active in this clause
    :param feature_domains: A list of :class:`FeatureDomain` objects that represent the domain of each feature
    :param features: An array of :class:`Feature` objects that represent the actual features to be mutated and evolved
    :param classification: The classification that this clause represents
    :param age: The age layer that this clause belongs in.
    """
    def __init__(self,
                 feature_mask: np.ndarray,
                 feature_domains: list[feature.FeatureDomain],
                 features: list[feature.Feature],
                 classification,
                 age: int = 0):
        if not (feature_mask.shape[0] == len(feature_domains) and feature_mask.shape[0] == len(features)):
            raise ValueError("'feature_mask', 'feature_domains', and 'features' must all be the same length.")

        super().__init__(mask=feature_mask,
                         items=features,
                         classification=classification,
                         age=age)

        if classification is None:
            raise ValueError("'classification' must not be None.")

        self.feature_domains: list[feature.FeatureDomain] = feature_domains

    def covers(self, observation: list | np.ndarray) -> bool:
        """ Returns true if the given observation contains acceptable values for every feature that is present.
            Features that are non-present are wild-cards and can accept any value.

        :param observation: The array of values that represent on observation to test for truth
        :return: True if the observation is described by this clause
        """
        if len(observation) != self.mask.shape[0]:
            raise ValueError(
                f"Observation array must be equal in length to the CC feature mask."
                f" Clause: {self.mask} Observation: {observation}"
            )

        for i, obs in enumerate(observation):
            if self.mask[i] and obs not in self.items[i]:
                return False

        return True

    def calc_coverage(self, observation_table: np.ndarray, classifications: np.ndarray):
        """ Builds observation coverage masks for this clause that will be used for bitwise operations.  Should be run
        once, after creation.

        Creates the following masks, counts and coverage arrays:
          - target_mask: The mask of all observations that are of this clause's class
          - nontarget_mask: The mask of all observations that are _not_ of this clause's class
          - coverage_mask: The mask of all observations covered by this clause (see :function:`_covers`)
          - target_coverage_mask: The mask of all observations that are both in the target, and are covered by this
            clause
          - nontarget_coverage_mask: The mask of all observations that are in the target, but are _not_ covered by this
            clause

          - coverage: An array of the observations present in ``coverage_mask``
          - target_coverage: An array of the observations present in ``target_coverage_mask``
          - nontarget_coverage: An array of the observations present in ``nontarget_coverage_mask``

          - coverage_count: The number of elements that are True in ``coverage_mask``
          - target_coverage_count: The number of elements that are True in ``target_coverage_mask``
          - nontarget_coverage_count: The number of elements that are True in ``nontarget_coverage_mask``

        Also creates the following statistics used for plotting and validation:
          - ppv: The positive predictive value, or percentage of covered observations that are in the target
          - cov: The coverage statistic, or percentage of target observations that are covered by this clause

        :param observation_table: The observation table to calculate coverage over [np.ndarray(n, m)]
        :param classifications: The array of classes associated with each observations [np.ndarray(n, )]
        """
        cov_mask = np.apply_along_axis(func1d=self.covers, arr=observation_table, axis=1)

        class_mask = classifications == self.classification

        # total coverage indices
        self.coverage_mask = cov_mask
        self.target_mask = class_mask
        self.nontarget_mask = ~class_mask

        self._calc_masks(observation_table)

    def remove_feature(self, feat: feature.Feature | int):
        """ Removes a feature from the clause, essentially disabling it in the feature mask

        :param feat: The feature to remove, the index of the feature to remove
        """
        # if a feature is passed, find its index
        if isinstance(feat, feature.Feature):
            feat_idx = np.flatnonzero(self.items == feat)[0]
        elif isinstance(feat, int):
            feat_idx = feat
        else:
            raise TypeError("'feat' must be a Feature or an integer index.")

        self.mask[feat_idx] = False
        self.items[feat_idx] = None

    def __copy__(self):
        return ConjunctiveClause(self.mask.copy(),
                                 self.feature_domains.copy(),
                                 self.items.copy(),
                                 self.classification,
                                 self.age)

    def __str__(self):
        return f"CC: mask={self.mask.tolist()}; class={self.classification}; age={self.age}; fitness={self.fitness};"

    @staticmethod
    def init_clause(observation: np.ndarray,
                    classification,
                    clause_order: int,
                    feature_domains: list[feature.FeatureDomain],
                    rng: np.random.Generator = None) -> ConjunctiveClause:
        """ A factory function to generate a clause

        :param observation: The seed observation to use for initializing the new clause
        :param classification: The class of the seed observation
        :param clause_order: The order of the new clause
        :param feature_domains: The list of feature domains to give to the new clause
        :param rng: A custom numpy RNG generator, if desired

        :return: A new clause initialized from the given inputs
        """
        if rng is None:
            rng = np.random.default_rng()

        # choose which features will be in this clause
        valid_features_idxs = np.nonzero(~np.isnan(observation))[0]
        chosen_feature_idxs = rng.choice(valid_features_idxs, size=clause_order, replace=False)

        features: list[feature.Feature | None] = [None for _ in range(len(feature_domains))]

        # for each of the chosen features, give it an initial feature set of [observation_value]
        for idx in chosen_feature_idxs:
            observation_value = observation[idx]
            feature_domain = feature_domains[idx]
            features[idx] = feature_domain.init_feature(observation_value)

        feature_mask = np.zeros(len(features), dtype=bool)
        feature_mask[chosen_feature_idxs] = True

        return ConjunctiveClause(feature_mask=feature_mask,
                                 feature_domains=feature_domains,
                                 features=features,
                                 classification=classification,
                                 age=0)

    @staticmethod
    def create_empty(mask: np.ndarray = None):
        """ Creates an empty conjunctive clause with the given mask.  This has no real functionality, and is
        intended for testing only.

        :param mask: The feature mask to give the empty array

        :return: An empty conjunctive clause
        """
        if mask is None:
            mask = np.zeros(1)

        domains = [feature.FeatureDomain.create_empty() for _ in range(mask.shape[0])]
        features = [feature.Feature.create_empty() for _ in range(mask.shape[0])]

        clause = ConjunctiveClause(mask, domains, features, 0)

        return clause

class CC(ConjunctiveClause):
    """ Shorthand for :class:`ConjunctiveClause` """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
