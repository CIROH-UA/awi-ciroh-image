"""ConjunctiveClause unit tests"""
import pytest
import numpy as np
from teva.ccea import conjunctive_clause as cc
from teva.base import feature
from teva.ccea import conjunctive_clause as source
from unit_tests import utils
from typing import Optional, Any

########################################################################################
# Fixtures
########################################################################################

@pytest.fixture
def continuous_values():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def ordinal_values():
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def categorical_values():
    return np.array(["a", "b", "c", "d", "e"])


@pytest.fixture
def continuous_feature_domain(continuous_values):
    return feature.NumericalFeatureDomain(continuous_values, feature.CONTINUOUS)


@pytest.fixture
def ordinal_feature_domain(ordinal_values):
    return feature.NumericalFeatureDomain(ordinal_values, feature.ORDINAL)


@pytest.fixture
def categorical_feature_domain(categorical_values):
    return feature.CategoricalFeatureDomain(categorical_values)

@pytest.fixture
def feature_domains(continuous_feature_domain, ordinal_feature_domain, categorical_feature_domain):
    return [continuous_feature_domain, ordinal_feature_domain, categorical_feature_domain]


@pytest.fixture
def values_covered(continuous_values, ordinal_values, categorical_values):
    num_continuous = np.random.randint(1, len(continuous_values))
    num_ordinal = np.random.randint(1, len(ordinal_values))
    num_categorical = np.random.randint(1, len(categorical_values))

    return (np.random.choice(continuous_values, num_continuous),
            np.random.choice(ordinal_values, num_ordinal),
            np.random.choice(categorical_values, num_categorical))

@pytest.fixture
def values_not_covered(continuous_values, ordinal_values, categorical_values, values_covered):
    return (np.setdiff1d(continuous_values, values_covered[0], assume_unique=True),
            np.setdiff1d(ordinal_values, values_covered[1], assume_unique=True),
            np.setdiff1d(categorical_values, values_covered[2], assume_unique=True))


@pytest.fixture
def continuous_feature(values_covered, continuous_feature_domain):
    return feature.NumericalFeature(values_covered[0], continuous_feature_domain)


@pytest.fixture
def ordinal_feature(values_covered, ordinal_feature_domain):
    return feature.NumericalFeature(values_covered[1], ordinal_feature_domain)


@pytest.fixture
def categorical_feature(values_covered, categorical_feature_domain):
    return feature.CategoricalFeature(values_covered[2], categorical_feature_domain)

@pytest.fixture
def cc_all_features_enabled(continuous_feature,
                            ordinal_feature,
                            categorical_feature,
                            feature_domains):
    feature_mask = np.array([True, True, True])
    features = [continuous_feature, ordinal_feature, categorical_feature]

    return cc.ConjunctiveClause(feature_mask, feature_domains, features, True)

@pytest.fixture
def covered_observations(values_covered):
    observations = []

    for _ in range(10):
        observation = make_observation(values_covered[0], values_covered[1], values_covered[2])
        observations.append(observation)

    return np.vstack(observations)

@pytest.fixture
def observations_not_covered(values_not_covered):
    observations = []

    for _ in range(10):
        observation = make_observation(values_not_covered[0], values_not_covered[1], values_not_covered[2])
        observations.append(observation)

    return np.vstack(observations)

@pytest.fixture
def all_observations(covered_observations, observations_not_covered):
    return np.vstack([covered_observations, observations_not_covered])


def make_observation(continuous_values: np.ndarray,
                     ordinal_values: np.ndarray,
                     categorical_values: np.ndarray):
    """Make a random observation with values from the given domains.

    :param continuous_values: The domain of continuous values
    :param ordinal_values: The domain of ordinal values
    :param categorical_values: The domain of categorical values
    :return: A random observation with values from the given domains.
    """
    continuous_values = np.random.choice(continuous_values)
    ordinal_values = np.random.choice(ordinal_values)
    categorical_values = np.random.choice(categorical_values)

    return np.array([continuous_values, ordinal_values, categorical_values],
                    dtype=object)


########################################################################################
# UNIT TESTS
########################################################################################

class TestCovers:
    """Tests for ConjunctiveClause's ``covers`` instance method"""
    @pytest.fixture
    def cc_first_feature_wildcard(self,
                                  continuous_feature,
                                  ordinal_feature,
                                  categorical_feature,
                                  feature_domains):
        feature_mask = np.array([False, True, True])
        features = [None, ordinal_feature, categorical_feature]

        return cc.ConjunctiveClause(feature_mask, feature_domains, features, True)

    @pytest.fixture
    def observations_first_feature_wildcard(self, values_covered, continuous_values):
        observations = []

        for _ in range(10):
            observation = make_observation(continuous_values, values_covered[1], values_covered[2])
            observations.append(observation)

        return np.vstack(observations)

    def test_covers_observations_that_fall_within_all_active_features(self,
                                                                      values_covered,
                                                                      cc_all_features_enabled):
        """Test that a clause covers observations that fall within all active features.
        """
        # Check that the clause covers any observation that falls within the feature domains
        for _ in range(10):
            observation = make_observation(values_covered[0], values_covered[1], values_covered[2])
            assert cc_all_features_enabled.covers(observation)

    def test_wildcard_feature_accepts_any_value_in_domain(self,
                                                          continuous_values,
                                                          values_covered,
                                                          cc_first_feature_wildcard):
        """Test that a clause with a wildcard feature accepts any value in the domain of the feature."""
        # Check that the clause covers any observation that falls within the covered
        # values of the ordinal and categorical features
        for _ in range(10):
            observation = make_observation(continuous_values, values_covered[1], values_covered[2])
            assert cc_first_feature_wildcard.covers(observation)

    def test_does_not_cover_observations_that_have_values_outside_of_active_features(self,
                                                                                     values_not_covered,
                                                                                     cc_all_features_enabled):
        """Test that a clause does not cover observations that have values outside the active features."""
        # Check that the clause does not cover any observation that falls outside the feature domains
        for _ in range(10):
            observation = make_observation(values_not_covered[0],
                                           values_not_covered[1],
                                           values_not_covered[2])
            assert not cc_all_features_enabled.covers(observation)

    def test_covers_observations_with_coverage_precomputed(self,
                                                                      covered_observations,
                                                                      cc_all_features_enabled):
        """Test that a clause covers observations that fall within all active features.
        """
        # An array of classifications associated with each observation. In this case, all
        # observations are classified as True, but this is arbitrary, as the classification
        # is not used to determine coverage.
        classifications = np.array([True] * len(covered_observations))
        cc_all_features_enabled.calc_coverage(covered_observations, classifications)

        # Check that the clause covers any observation that falls within the feature domains
        for observation in covered_observations:
            assert cc_all_features_enabled.covers(observation)

    def test_wildcard_feature_with_coverage_precomputed(self,
                                                          observations_first_feature_wildcard,
                                                          cc_first_feature_wildcard):
        """Test that a clause with a wildcard feature accepts any value in the domain of the feature
        after coverage has been computed.
        """
        # An array of classifications associated with each observation. In this case, all
        # observations are classified as True, but this is arbitrary.
        classifications = np.array([True] * len(observations_first_feature_wildcard))
        cc_first_feature_wildcard.calc_coverage(observations_first_feature_wildcard, classifications)

        # Check that the clause covers any observation that falls within the feature domains
        for observation in observations_first_feature_wildcard:
            assert cc_first_feature_wildcard.covers(observation)

    def test_does_not_cover_with_coverage_precomputed(self,
                                                                                     observations_not_covered,
                                                                                     all_observations,
                                                                                     cc_all_features_enabled):
        # An array of classifications associated with each observation. In this case, all
        # observations are classified as True, but this is arbitrary.
        classifications = np.array([True] * len(all_observations))
        cc_all_features_enabled.calc_coverage(all_observations, classifications)

        # Check that the clause does not cover any observation that falls outside the feature domains
        for observation in observations_not_covered:
            assert not cc_all_features_enabled.covers(observation)


class TestCalcCoverage:
    """Tests for ConjunctiveClause's ``calc_coverage`` instance method"""
    @pytest.fixture
    def target_coverage_count(self, covered_observations):
        # At least one target observation must be covered and one covered observation
        # must not be a target observation
        np.random.randint(1, len(covered_observations))

    @pytest.fixture
    def non_target_coverage_count(self, observations_not_covered):
        # At least one observation that is not covered must be a target observation and
        # one observation that is not covered must not be a target observation
        return np.random.randint(1, len(observations_not_covered))

    @pytest.fixture
    def target_coverage_mask(self, covered_observations, target_coverage_count, observations_not_covered):
        target_coverage_mask = np.zeros(shape=(len(covered_observations),), dtype=bool)
        target_coverage_mask[:target_coverage_count] = True
        np.random.shuffle(target_coverage_mask)

        return np.concat([target_coverage_mask, np.zeros(shape=(len(observations_not_covered),), dtype=bool)])

    @pytest.fixture
    def non_target_coverage_mask(self, observations_not_covered, non_target_coverage_count, covered_observations):
        non_target_coverage_mask = np.zeros(shape=(len(observations_not_covered),), dtype=bool)
        non_target_coverage_mask[:non_target_coverage_count] = True
        np.random.shuffle(non_target_coverage_mask)

        return np.concat([np.zeros(shape=(len(covered_observations),), dtype=bool), non_target_coverage_mask])

    # @pytest.fixture


    @pytest.fixture
    def classifications(self,
                        covered_observations,
                        observations_not_covered,
                        all_observations):
        pass

    @pytest.mark.skip
    def test_correct_coverage_mask(self):
        pass

    @pytest.mark.skip
    def test_correct_coverage_count(self):
        pass

    @pytest.mark.skip
    def test_correct_target_coverage_mask(self):
        pass

    @pytest.mark.skip
    def test_correct_target_coverage_count(self):
        pass

    @pytest.mark.skip
    def test_correct_non_target_coverage_mask(self):
        pass

    @pytest.mark.skip
    def test_correct_non_target_coverage_count(self):
        pass




class TestDescribes:
    observations = np.array([
        [0.1, 1, "a"],
        [0.2, 2, "b"],
        [0.3, 3, "c"],
        [0.4, 4, "d"],
        [0.5, 5, "e"],
    ], dtype=object)
    continuous_feature_domain = feature.NumericalFeatureDomain(observations[:, 0],
                                                               feature.CONTINUOUS)
    ordinal_feature_domain = feature.NumericalFeatureDomain(observations[:, 1],
                                                            feature.ORDINAL)
    categorical_feature_domain = feature.CategoricalFeatureDomain(observations[:, 2])
    target = True

    def make_cc(self, feature_mask: np.ndarray,
                features: Optional[list[feature.Feature]] = None,
                target: Optional[Any] = None):
        if features is None:
            features = [
                feature.NumericalFeature(self.observations[:, 0], self.continuous_feature_domain),
                feature.NumericalFeature(self.observations[:, 1], self.ordinal_feature_domain),
                feature.CategoricalFeature(self.observations[:, 2], self.categorical_feature_domain)
            ]

        if target is None:
            target = self.target

        clause = cc.ConjunctiveClause(feature_mask,
                                      [
                                          self.continuous_feature_domain,
                                          self.ordinal_feature_domain,
                                          self.categorical_feature_domain
                                      ],
                                      features,
                                      classification=target)  # arbitrary

        return clause

    def test_describes_covered_observations_with_same_class(self):
        # Enable all features
        feature_mask = np.array([True, True, True])
        clause = self.make_cc(feature_mask)
        # TODO: Set the dtype to object in the observation table, otherwise describes will fail with string values
        assert clause.describes(np.array([0.1, 1, "a"], dtype=object), True)
        assert clause.describes(np.array([0.1, 2, "b"], dtype=object), True)

    def test_describes_covered_observation_with_coverage_precomputed(self):
        # Array of "true"s. One for each observation
        targets = np.ones(shape=(len(self.observations),), dtype=bool)
        feature_mask = np.array([True, True, True])
        clause = self.make_cc(feature_mask)
        clause.calc_coverage(self.observations, targets)

        for idx in range(len(self.observations)):
            assert clause.describes(self.observations[idx, :], targets[idx])

    def test_does_not_describe_covered_observations_with_different_class(self):
        targets = np.zeros(shape=(len(self.observations),), dtype=bool)
        feature_mask = np.array([True, True, True])
        clause = self.make_cc(feature_mask)

        assert not clause.describes(np.array([0.1, 1, "a"], dtype=object), False)
        assert not clause.describes(np.array([0.1, 2, "b"], dtype=object), False)

        clause.calc_coverage(self.observations, targets)

        for idx in range(len(self.observations)):
            assert not clause.describes(self.observations[idx, :], targets[idx])

    def test_describes_observations_with_wildcard_features(self):
        feature_mask = np.array([True, True, False])
        clause = self.make_cc(feature_mask)

        assert clause.describes(np.array([0.2, 2, "a"], dtype=object), self.target)

        feature_mask = np.array([True, False, True])
        clause = self.make_cc(feature_mask)

        assert clause.describes(np.array([0.2, 4, "b"], dtype=object), self.target)

        feature_mask = np.array([False, True, True])
        clause = self.make_cc(feature_mask)

        assert clause.describes(np.array([0.1, 2, "c"], dtype=object), self.target)

        feature_mask = np.array([False, True, False])
        clause = self.make_cc(feature_mask)

        assert clause.describes(np.array([0.3, 2, "a"], dtype=object), self.target)

    def test_describes_observations_with_wildcard_features_with_coverage_precomputed(self):
        targets = np.ones(shape=(len(self.observations),), dtype=bool)

        feature_mask = np.array([True, True, False])
        clause = self.make_cc(feature_mask)
        clause.calc_coverage(self.observations, targets)

        print()
        utils.try_assert("[True, True, False]",
                         clause.describes(np.array([0.2, 2, "a"], dtype=object), self.target))

        feature_mask = np.array([True, False, True])
        clause = self.make_cc(feature_mask)
        clause.calc_coverage(self.observations, targets)

        utils.try_assert("[True, False, True]",
                         clause.describes(np.array([0.2, 4, "b"], dtype=object), self.target))

        feature_mask = np.array([False, True, True])
        clause = self.make_cc(feature_mask)
        clause.calc_coverage(self.observations, targets)

        utils.try_assert("[False, True, True]",
                         clause.describes(np.array([0.1, 3, "c"], dtype=object), self.target))

        feature_mask = np.array([False, True, False])
        clause = self.make_cc(feature_mask)
        clause.calc_coverage(self.observations, targets)

        utils.try_assert("[False, True, False]",
                         clause.describes(np.array([0.3, 2, "a"], dtype=object), self.target))

    def test_does_not_describe_observations_that_it_does_not_cover(self):
        features = [
            feature.NumericalFeature(np.array([0.3, 0.4]), self.continuous_feature_domain),
            feature.NumericalFeature(np.array([3, 4]), self.ordinal_feature_domain),
            feature.CategoricalFeature(np.array(["c", "d"]), self.categorical_feature_domain)
        ]

        feature_mask = np.array([True, True, True])
        clause = self.make_cc(feature_mask, features)

        print()
        utils.try_assert("[0.3, 5, 'a']",
                         not clause.describes(np.array([0.3, 5, "a"], dtype=object), self.target))
        utils.try_assert("[0.1, 3, 'c']",
                         not clause.describes(np.array([0.1, 3, "c"], dtype=object), self.target))

    def test_does_not_describe_observations_that_it_does_not_cover_with_coverage_precomputed(self):
        targets = np.ones(shape=(len(self.observations),), dtype=bool)
        features = [
            feature.NumericalFeature(np.array([0.3, 0.4]), self.continuous_feature_domain),
            feature.NumericalFeature(np.array([3, 4]), self.ordinal_feature_domain),
            feature.CategoricalFeature(np.array(["c", "d"]), self.categorical_feature_domain)
        ]

        feature_mask = np.array([True, True, True])
        clause = self.make_cc(feature_mask, features)
        clause.calc_coverage(self.observations, targets)

        print()
        utils.try_assert("[0.3, 5, 'a']",
                         not clause.describes(np.array([0.3, 5, "a"], dtype=object), self.target))
        utils.try_assert("[0.1, 3, 'c']",
                         not clause.describes(np.array([0.1, 3, "c"], dtype=object), self.target))


########################################################################################
# LEGACY TESTS
########################################################################################

# def test_conjunctive_clause_init():
#     # generate some test domains
#     domains = [
#         feature.CategoricalFeatureDomain(domain=["one", "two", "three"]),
#         feature.NumericalFeatureDomain(domain=[1, 2, 5, 6, 9, 39, 20], feature_type=feature.ORDINAL),
#         feature.NumericalFeatureDomain(domain=[6, 8, 1, 7, 325, 12, 55], feature_type=feature.ORDINAL),
#         feature.CategoricalFeatureDomain(domain=[True, False]),
#         feature.NumericalFeatureDomain(domain=[-0.22, -0.2, 0.1, 0.2, 0.3, 0.4, 0.9], feature_type=feature.CONTINUOUS)
#     ]
#
#     # use the first value of each domain to initialize a feature for each
#     features = [domain.init_feature(domain.domain[0]) for domain in domains]
#
#     # rand_mask: np.ndarray = np.random.rand(5) < 0.5
#     static_mask: np.ndarray = np.array([True, True, False, False, True])
#
#     static_clause = source.ConjunctiveClause(feature_mask=static_mask,
#                                              feature_domains=domains,
#                                              features=features,
#                                              classification="class")
#
#     print()
#     # TODO: test that each feature is in the feature domain
#     utils.try_assert("STATIC | feature_mask", bool(np.all(static_clause.feature_mask == static_mask)))
#     utils.try_assert("STATIC | features", static_clause.features == features)
#
#     utils.try_assert("STATIC | describes | True", static_clause.describes(np.array(["one", 1, 1, True, -0.22])))
#     utils.try_assert("STATIC | describes | False", not static_clause.describes(np.array(["one", 8, 1, True, .11])))
#     utils.try_assert("STATIC | describes | WC True", static_clause.describes(np.array(["one", 1, None, None, -0.22])))
#     utils.try_assert("STATIC | describes | WC False", not static_clause.describes(np.array([None, 5, 1, True, 0.1])))


# def test_hashing():
#     domains = [
#         feature.CategoricalFeatureDomain(domain=[True, False]),
#         feature.CategoricalFeatureDomain(domain=[True, False]),
#         feature.CategoricalFeatureDomain(domain=[True, False]),
#         feature.CategoricalFeatureDomain(domain=[True, False]),
#         feature.CategoricalFeatureDomain(domain=[True, False]),
#     ]
#
#     features_a = [
#         domains[0].init_feature(domains[0].domain[0]),
#         domains[1].init_feature(domains[1].domain[0]),
#         domains[2].init_feature(domains[2].domain[0]),
#         domains[3].init_feature(domains[3].domain[0]),
#         domains[4].init_feature(domains[4].domain[0]),
#     ]
#
#     features_b = [
#         domains[0].init_feature(domains[0].domain[0]),
#         domains[1].init_feature(domains[1].domain[1]),
#         domains[2].init_feature(domains[2].domain[0]),
#         domains[3].init_feature(domains[3].domain[1]),
#         domains[4].init_feature(domains[4].domain[0]),
#     ]
#
#     clause1 = source.ConjunctiveClause(feature_mask=np.array([True, True, False, False, True]),
#                                        feature_domains=domains,
#                                        features=features_a,
#                                        classification=True)
#
#     clause2 = source.ConjunctiveClause(feature_mask=np.array([True, True, False, False, True]),
#                                        feature_domains=domains,
#                                        features=features_b,
#                                        classification=True)
#
#     utils.try_assert("1 != 2", hash(clause1) != hash(clause2))
#     utils.try_assert("1 == 1", hash(clause1) == hash(clause1))
#     utils.try_assert("2 == 2", hash(clause2) == hash(clause2))
