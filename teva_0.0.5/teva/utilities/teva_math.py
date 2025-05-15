""" Utility math functions used in TEVA """
import numpy as np
from numpy.random import MT19937

class Rand:
    RNG: np.random.Generator = np.random.default_rng(MT19937())

    @staticmethod
    def set_seed(seed):
        Rand.RNG = np.random.default_rng(seed)

def argmax(*args, **kwargs) -> np.signedinteger | np.dtype:
    """ An implementation of :function:`numpy.argmax()` that always returns a value instead of a list of values.
    If there are multiple maximum values, it uses unbiased random choice to choose one.

    For parameter descriptions, see :function:`numpy.argmax()`

    :return: The index of the maximum element in the input array
    """
    res = np.argmax(*args, **kwargs)
    if isinstance(res, np.ndarray):
        return np.random.choice(res)
    return res

def argmin(*args, **kwargs) -> np.signedinteger | np.dtype:
    """ An implementation of :function:`numpy.argmin()` that always returns a value instead of a list of values.
    If there are multiple minimum values, it uses unbiased random choice to choose one.

    For parameter descriptions, see :function:`numpy.argmin()`

    :return: The index of the minimum element in the input array
    """
    res = np.argmin(*args, **kwargs)
    if isinstance(res, np.ndarray):
        return np.random.choice(res)
    return res

def flatten_dict(dictionary: dict, flatten_numpy=False) -> list:
    """ Flattens the elements of a dictionary into a list of all elements.

    :param dictionary: A dictionary to be flattened
    :param flatten_numpy: If true, all numpy arrays will be flattened and their elements appended to the result.
        Otherwise, numpy arrays will be appended as elements.
    :return: All the elements under all keys of the input dictionary in a single list
    """
    result = []
    for key in dictionary.keys():
        val = dictionary[key]
        if isinstance(val, dict):
            result.extend(flatten_dict(val))
        elif isinstance(val, list):
            result.extend(val)
        elif isinstance(val, np.ndarray) and flatten_numpy:
            result.extend(val.flatten().tolist())
        else:
            result.append(val)

    return result
