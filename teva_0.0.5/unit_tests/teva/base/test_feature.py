# from pathlib import Path
import numpy as np
import copy

from teva.base import feature as source
from unit_tests import utils

def array_eq(arr1, arr2):
    print(arr1)
    print(arr2)
    return bool(np.all(arr1 == arr2))

def test_Feature():
    # ORDINAL
    ord_feature_domain = source.NumericalFeatureDomain(np.array([9, 1, 2, 3, 5, 6, 8]), source.FeatureType.ORDINAL)
    ord_feature = source.NumericalFeature(feature_set=np.array([3, 1]), feature_domain=ord_feature_domain)

    print()
    utils.try_assert("Ordinal | set    ", array_eq(ord_feature.feature_set(), np.array([1, 3])))
    utils.try_assert("Ordinal | copy/eq", ord_feature == copy.copy(ord_feature))
    utils.try_assert("Ordinal | len    ", len(ord_feature) == 2)
    utils.try_assert("Ordinal | in     ", 3 in ord_feature)
    utils.try_assert("Ordinal | index  ", ord_feature[0] == 1)

    # CONTINOUS
    cont_feature_domain = source.NumericalFeatureDomain(np.array([9.4, 15.14, -.31, 3.0, 5.55, 6.097, 8.1]),
                                                        source.FeatureType.CONTINUOUS)
    cont_feature = source.NumericalFeature(feature_set=np.array([15.14, 5.55]), feature_domain=cont_feature_domain)

    print()
    # don't forget that it sorts them
    utils.try_assert("Continuous | set    ", array_eq(cont_feature.feature_set(), np.array([5.55, 15.14])))
    utils.try_assert("Continuous | copy/eq", cont_feature == copy.copy(cont_feature))
    utils.try_assert("Continuous | len    ", len(cont_feature) == 2)
    utils.try_assert("Continuous | in     ", 15.14 in cont_feature)
    utils.try_assert("Continuous | index  ", cont_feature[0] == 5.55)

    # CATEGORICAL
    cat_feature_domain = source.CategoricalFeatureDomain(["val1", 1, True, "val2", 19.001])
    cat_feature = source.CategoricalFeature(feature_set=[True, "val1"], feature_domain=cat_feature_domain)

    print()
    utils.try_assert("Categorical | set    ", array_eq(cat_feature.feature_set(), [True, "val1"]))
    utils.try_assert("Categorical | copy/eq", cat_feature == copy.copy(cat_feature))
    utils.try_assert("Categorical | len    ", len(cat_feature) == 2)
    utils.try_assert("Categorical | in     ", True in cat_feature)
    utils.try_assert("Categorical | index  ", cat_feature[0] == True)

def test_FeatureDomain():
    ord_feature_domain = source.NumericalFeatureDomain(np.array([9, 1, 2, 3, 5, 6, 8]), source.FeatureType.ORDINAL)
    cont_feature_domain = source.NumericalFeatureDomain(np.array([9.4, 15.14, -.31]), source.FeatureType.CONTINUOUS)
    cat_feature_domain = source.CategoricalFeatureDomain(["val1", 1, True, "val2", 19.001])

    print()
    utils.try_assert("Ordinal | random", ord_feature_domain.init_feature()[0] in ord_feature_domain)
    utils.try_assert("Ordinal | given ", ord_feature_domain.init_feature(3)[0] == 3)
    utils.try_assert("Ordinal | index ", ord_feature_domain[0] == 1)
    utils.expect_failure("Ordinal | bad   ", ord_feature_domain.init_feature, function_args=[21])

    print()
    utils.try_assert("Continuous | random", cont_feature_domain.init_feature()[0] in cont_feature_domain)
    utils.try_assert("Continuous | given ", cont_feature_domain.init_feature(15.14)[0] == 15.14)
    utils.try_assert("Continuous | index ", cont_feature_domain[0] == -.31)
    utils.expect_failure("Continuous | bad   ", cont_feature_domain.init_feature, function_args=[21])

    print()
    utils.try_assert("Categorical | random", cat_feature_domain.init_feature()[0] in cat_feature_domain)
    utils.try_assert("Categorical | given ", cat_feature_domain.init_feature("val2")[0] == "val2")
    utils.try_assert("Categorical | index ", cat_feature_domain[0] == "val1")
    utils.expect_failure("Categorical | bad1  ", cat_feature_domain.init_feature, function_args=[21])
    utils.expect_failure("Categorical | bad2  ", cat_feature_domain.init_feature, function_args=["bad"])

