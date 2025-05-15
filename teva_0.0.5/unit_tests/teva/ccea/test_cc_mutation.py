import pytest
from unittest import mock
import numpy as np
from teva.base import feature

from teva.ccea import conjunctive_clause as cc
from teva.ccea import mutation as source
from unit_tests import utils
from unit_tests.utils import try_assert


def test__select_features_random():
    domains = [
        feature.CategoricalFeatureDomain([True, False]),
        feature.CategoricalFeatureDomain([True, False]),
        feature.CategoricalFeatureDomain([True, False]),
        feature.CategoricalFeatureDomain([True, False]),
        feature.CategoricalFeatureDomain([True, False]),
    ]

    clause = cc.ConjunctiveClause.init_clause([False, False, True, False, True],
                                              False,
                                              3,
                                              feature_domains=domains)

    selected = source._select_features_random(clause, 0.0)

    print()
    try_assert("Count", 0 < selected.shape[0] <= 5)
    try_assert("Max", selected.max() < 5)
    try_assert("Min", selected.min() >= 0)
    print(selected)

def test__mutate_categorical():
    domain = feature.CategoricalFeatureDomain(["val1", "val2", "val3"])
    feat = domain.init_feature("val1")

    print()
    mutated = source._mutate_categorical(feat)
    utils.try_assert("Adds At Min", len(mutated) == 2)

    feat._feature_idxs = np.array([0, 1, 2])
    mutated2 = source._mutate_categorical(feat)
    utils.try_assert("Removes at Max", len(mutated2) == 2)

    all_valid = True
    for val in feat.feature_set():
        if val not in domain:
            all_valid = False
    try_assert("Valid Values", all_valid)

def test__mutate_numerical():
    dom = np.random.random(10)
    domain = feature.NumericalFeatureDomain(dom, feature.FeatureType.CONTINUOUS)
    feat = domain.init_feature(dom[0])

    mutated = source._mutate_numerical(feat)
    # feat._feature_idxs = np.array([0, 1, 2])
    # mutated2 = source._mutate_categorical(feat)
    # utils.try_assert("Removes at Max", len(mutated2) == 2)
    print()
    all_valid = True
    for val in feat.feature_set():
        if val not in domain:
            all_valid = False
    try_assert("Valid Values", all_valid)

def test_mutate_feature():
    with (mock.patch("teva.ccea.mutation._mutate_numerical") as test_mut_num,
          mock.patch("teva.ccea.mutation._mutate_categorical") as test_mut_cat):

        print()
        source.mutate_feature(feature.NumericalFeature(np.array([1, 2, 3]),
                                                       feature.NumericalFeatureDomain([1, 2, 3],
                                                                                      feature.FeatureType.ORDINAL)))
        utils.try_assert("Numerical Called", test_mut_num.call_count == 1)
        utils.try_assert("Categorical Not Called", test_mut_cat.call_count == 0)

        source.mutate_feature(feature.CategoricalFeature([1, 2, 3], feature.CategoricalFeatureDomain([1, 2, 3])))
        utils.try_assert("Numerical Not Called", test_mut_num.call_count == 1)
        utils.try_assert("Categorical Called", test_mut_cat.call_count == 1)


def test_mutate_clause():
    pytest.skip("No test required")

def test_select_features():
    pytest.skip("No test required")

def test__select_features_selective():
    pytest.skip("Source function not implemented.")