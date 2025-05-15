import pytest
import random
from unittest import mock
# from pathlib import Path
from scipy.stats import hypergeom
import numpy as np

from teva.utilities import hyge as source
from unit_tests import utils

def test_hygepmf():
    # 0 <= N
    # 0 <= K <= N
    # 0 <= n <= N
    # max(0, n + K - N) <= k <= min(K, n)
    print()
    utils.expect_failure("N < 0", source.hygepmf, function_kwargs={"k": 2, "n": 1, "K": 1, "N": -2}, expected_exception=ArithmeticError)
    utils.expect_failure("n > N", source.hygepmf, function_kwargs={"k": 1, "n": 2, "K": 1, "N": 1}, expected_exception=ArithmeticError)
    utils.expect_failure("K > N", source.hygepmf, function_kwargs={"k": 0, "n": 0, "K": 2, "N": 1}, expected_exception=ArithmeticError)
    utils.expect_failure("Require Int", source.hygepmf, function_kwargs={"k": 1.0, "n": 1.0, "K": 1.0, "N": 1.0},
                         expected_exception=AttributeError)

    utils.try_assert("Outlier", utils.near(source.hygepmf(600, 1250, 600, 620), -336.8796, 0.0001))

def test_compare_hygpmf():
    print()
    print("Compare hygpmf with hypergeom.pmf within the range usable by hypergeom.pmf:")
    N = random.randint(0, 100)
    K = random.randint(0, N)
    n = random.randint(0, N)
    k = random.randint(max(0, n + K - N), min(K, n))

    a = source.hygepmf(k=k, n=n, K=K, N=N)
    b = float(np.log10(hypergeom.pmf(k=k, M=N, n=K, N=n)))

    utils.try_assert("Hanley == Scipy", utils.near(a, b, 0.0000000000001))

def test_hygcontours():
    pytest.skip("Test Not Yet Implemented")