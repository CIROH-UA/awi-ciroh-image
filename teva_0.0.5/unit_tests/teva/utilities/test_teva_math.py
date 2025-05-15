import numpy as np

from teva.utilities import teva_math as source
from unit_tests import utils

def test_argmax():
    single_max = source.argmax([0, 1, 7, 3, 4, 8, 2, 5])
    double_max = source.argmax([0, 1, 8, 3, 4, 8, 2, 5])

    print()
    utils.try_assert("Single", single_max == 5)
    utils.try_assert("Double", double_max == 5 or double_max == 2)


def test_argmin():
    single_min = source.argmin([1, 0, 7, 3, 4, 8, 2, 5])
    double_min = source.argmin([0, 1, 8, 3, 4, 0, 2, 5])

    print()
    utils.try_assert("Single", single_min == 1)
    utils.try_assert("Double", double_min == 5 or double_min == 0)

def test_flatten_dict():
    dict1 = {
        "a": 1,
        "b": [2, 3, 4, 5],
        "c": np.array([6, 7, 8, 9])
    }

    without_np = source.flatten_dict(dict1)
    with_np = source.flatten_dict(dict1, flatten_numpy=True)

    print()
    utils.try_assert("Without NP", without_np[0:5] == [1, 2, 3, 4, 5] and np.all(without_np[-1] == np.array([6, 7, 8, 9])))
    utils.try_assert("With NP", with_np == [1, 2, 3, 4, 5, 6, 7, 8, 9])
