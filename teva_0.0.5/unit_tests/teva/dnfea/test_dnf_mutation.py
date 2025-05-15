# from pathlib import Path
import numpy as np

from teva.dnfea import mutation as source
from unit_tests import utils

def test__swap_random_bit():
    print()
    mask = np.array([True, False, False, False, False])
    modified = source._swap_random_bit(mask, activate=True)
    utils.try_assert("Activate Any", np.sum(modified) == 2)

    mask = np.array([True, True, True, True, True])
    # modified = source._swap_random_bit(mask, activate=True)
    # utils.try_assert("Activate Full", np.sum(modified) == 5)
    utils.expect_failure("Activate Full Exception",
                         function=source._swap_random_bit,
                         function_args=[mask, True],
                         expected_exception=ValueError)

    mask = np.array([True, True, False, True, True])
    modified = source._swap_random_bit(mask, activate=False)
    utils.try_assert("Deactivate Any", np.sum(modified) == 3)

    mask = np.array([False, False, False, False, False])
    utils.expect_failure("Deactivate Empty Exception",
                         function=source._swap_random_bit,
                         function_args=[mask, False],
                         expected_exception=ValueError)

# def test__targeted_enable_uncovered():
#     clause = dnf.DisjunctiveClause.create_empty(np.array([True, False, False, False]))
#     clause.cc_clauses = [
#         cc.ConjunctiveClause.create_empty(),
#         cc.ConjunctiveClause.create_empty(),
#         cc.ConjunctiveClause.create_empty(),
#         cc.ConjunctiveClause.create_empty(),
#     ]
#     for cc_clause in clause.cc_clauses:
#         cc_clause.target_coverage_mask = np.array([True, False, False, True, False, False])
#     clause.target_mask = np.array([True, False, True, False, True, False])
#     clause.target_count = np.sum(clause.target_mask)
#     clause.coverage_mask = np.array([True, False, True, False, True, False])
#     clause.coverage_count = np.sum(clause.coverage_mask)
#     clause.target_coverage_mask = np.array([True, False, True, False, True, False])
#     clause.target_coverage_count = np.sum(clause.target_coverage_mask)
#
#     child_mask = source._targeted_enable_uncovered(clause)
#     utils.try_assert("", np.all(child_mask == [True, False, False, False]))