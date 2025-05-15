import pytest
# from pathlib import Path
import numpy as np

from teva.dnfea import disjunctive_clause as dnf
from teva.dnfea import fitness as source
from unit_tests import utils
from teva.base import fitness

def test_ratio_test():
    print()
    clause = dnf.DisjunctiveClause.create_empty()
    clause.total_observations = 5
    clause.target_coverage_count = 3
    clause.target_mask = np.array([True, False, False, False, True])
    clause.target_count = 2
    clause.coverage_count = 1

    utils.try_assert("RATIO PASS", fitness.ratio_test(clause))

    clause.target_coverage_count = 1
    clause.target_mask = np.array([True, True, False, True, True])
    clause.target_count = 4
    clause.coverage_count = 4
    utils.try_assert("RATIO FAIL", not fitness.ratio_test(clause))


# def test__calc_fitness():
#     clause = dnf.DisjunctiveClause.create_empty()
#     clause.target_coverage_mask = np.array([True, True, False, False, True])
#     clause.target_coverage_count = 3
#     clause.target_mask = np.array([True, False, False, False, True])
#     clause.target_count = 2
#     clause.coverage_mask = np.array([False, False, False, False, True])
#     clause.coverage_count = 1
#
#     def fitness_function(k, N, K, n):
#         return k, N, K, n
#
#     # Simply tests that the values are passed through correctly.  Does not test the
#     # hyper-geometric function
#     res = fitness.test_fitness(clause, fitness_function)
#     print()
#     utils.try_assert("k", res[0] == 3)
#     utils.try_assert("N", res[1] == 5)
#     utils.try_assert("K", res[2] == 2)
#     utils.try_assert("n", res[3] == 1)

def test_child_fitness():
    pytest.skip("Not yet implemented")

def test_children_fitness():
    pytest.skip("Not yet implemented")