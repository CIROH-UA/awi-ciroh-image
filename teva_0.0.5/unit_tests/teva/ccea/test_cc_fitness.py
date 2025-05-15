import pytest
# from pathlib import Path
import numpy as np

from teva.base import feature, fitness
from teva.ccea import conjunctive_clause as cc
from teva.ccea import fitness as source
from unit_tests import utils

# mock_feature_types = [
#     feature.FeatureType.CATEGORICAL,
#     feature.FeatureType.CATEGORICAL,
#     feature.FeatureType.CATEGORICAL,
#     feature.FeatureType.CATEGORICAL,
# ]
#
# mock_obs_table = np.vstack([
#     np.array([0, 1, 0, 0]),
#     np.array([0, 1, 0, 0]),
#     np.array([0, 0, 1, 0]),
#     np.array([0, 1, 0, 1]),
#     np.array([0, 0, 0, 0])
# ])
#
# mock_feature_domain = FeatureDomain(np.array())
#

def test__continuous_feature_ratio_test():
    pytest.skip("Don't know why this is skipped")

    num_features = 5
    num_observations = 10

    # mock_obs_classes = np.array([True, False, False, True, True])
    # 5 features, 10 observations, where each feature
    mock_obs_table = np.random.rand(num_observations, num_features)

    # The classes are 3 minus the floored sum of the 2nd, 4th, and 5th column, so
    # [0.5, 0.7, 0.3, 0.8, 0.4] -> 3 - floor(0.7 + 0.8 + 0.4) = 3 - 1 = 2
    mock_obs_classes = 3 - np.floor(np.sum(mock_obs_table[:, [1,3,4]], axis=1))
    target_class = np.random.choice(mock_obs_classes)
    target_features = mock_obs_table[mock_obs_classes == target_class]


    # Compute feature domains
    feature_domains = []
    temp = np.unique(mock_obs_table, axis=0)

    for idx in range(num_features):
        feature_domains.append(
            feature.NumericalFeatureDomain(
                temp[:, idx],
                feature.CONTINUOUS
            )
        )

    # Create clauses
    clauses = []

    for idx in range(num_observations):
        clauses.append(
            cc.ConjunctiveClause.init_clause(
                mock_obs_table[idx,:],
                mock_obs_classes[idx],
                np.random.randint(5),
                feature_domains
            )
        )

def test_ratio_test():
    print()

    clause = cc.ConjunctiveClause(np.zeros(0), [], [], False)
    clause.total_observations = 5
    clause.target_coverage_mask = np.array([True, True, False, False, True])
    clause.target_coverage_count = np.sum(clause.target_coverage_mask)
    clause.target_mask = np.array([True, False, False, False, True])
    clause.target_count = np.sum(clause.target_mask)
    clause.coverage_mask = np.array([False, False, False, False, True])
    clause.coverage_count = np.sum(clause.coverage_mask)

    utils.try_assert("RATIO PASS", fitness.ratio_test(clause))

    clause.target_coverage_mask = np.array([False, False, False, False, True])
    clause.target_coverage_count = np.sum(clause.target_coverage_mask)
    clause.target_mask = np.array([True, True, False, True, True])
    clause.target_count = np.sum(clause.target_mask)
    clause.coverage_mask = np.array([True, True, True, True, False])
    clause.coverage_count = np.sum(clause.coverage_mask)
    utils.try_assert("RATIO FAIL", not fitness.ratio_test(clause))

# def test_test_fitness():
#     clause = cc.ConjunctiveClause(np.zeros(0), [], [], False)
#     clause.total_observations = 5
#     clause.target_coverage_mask = np.array([True, True, False, False, True])
#     clause.target_coverage_count = np.sum(clause.target_coverage_mask)
#     clause.target_mask = np.array([True, False, False, False, True])
#     clause.target_count = np.sum(clause.target_mask)
#     clause.coverage_mask = np.array([False, False, False, False, True])
#     clause.coverage_count = np.sum(clause.coverage_mask)
#
#     def fitness_function_k(k, N, K, n):
#         return k
#     def fitness_function_N(k, N, K, n):
#         return N
#     def fitness_function_K(k, N, K, n):
#         return K
#     def fitness_function_n(k, N, K, n):
#         return n
#
#     # Simply tests that the values are passed through correctly.  Does not test the
#     # hyper-geometric function
#     res_k = fitness.test_fitness(clause, fitness_function_k, max_order=5, fitness_threshold=0.0)
#     res_N = fitness.test_fitness(clause, fitness_function_N, max_order=5, fitness_threshold=0.0)
#     res_K = fitness.test_fitness(clause, fitness_function_K, max_order=5, fitness_threshold=0.0)
#     res_n = fitness.test_fitness(clause, fitness_function_n, max_order=5, fitness_threshold=0.0)
#     print()
#     utils.try_assert("k", res_k == 3)
#     utils.try_assert("N", res_N == 5)
#     utils.try_assert("K", res_K == 2)
#     utils.try_assert("n", res_n == 1)

def test_feature_sensitivity():
    clause = cc.ConjunctiveClause(np.zeros(0), [], [], False)
    clause.target_coverage_mask = np.array([True, True, False, False, True])
    clause.target_mask = np.array([True, False, False, False, True])
    clause.coverage_mask = np.array([False, False, False, False, True])

    utils.try_assert("", fitness.sensitivity(clause, 1.0, 1.0))

def test_clause_fitness():
    pytest.skip("NOT IMPLEMENTED")

def test_child_fitness():
    pytest.skip("NOT IMPLEMENTED")

def test_children_fitness():
    pytest.skip("NOT IMPLEMENTED")