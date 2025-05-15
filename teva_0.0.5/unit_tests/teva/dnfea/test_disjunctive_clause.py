# from pathlib import Path
import numpy as np

from teva.base import feature
from teva.ccea import conjunctive_clause as cc
from teva.dnfea import disjunctive_clause as source
from unit_tests import utils

def test_DisjunctiveClause():
    domains = [
        feature.CategoricalFeatureDomain(domain=["one", "two", "three"]),
        feature.NumericalFeatureDomain(domain=[1, 2, 5, 6, 9, 39, 20], feature_type=feature.ORDINAL),
        feature.NumericalFeatureDomain(domain=[6, 8, 1, 7, 325, 12, 55], feature_type=feature.ORDINAL),
        feature.CategoricalFeatureDomain(domain=[True, False]),
        feature.NumericalFeatureDomain(domain=[-0.22, -0.2, 0.1, 0.2, 0.3, 0.4, 0.9], feature_type=feature.CONTINUOUS)
    ]

    clauses = [
        cc.ConjunctiveClause(feature_mask=np.array([True, False, True, False, False]),
                             feature_domains=domains,
                             features=[domain.init_feature(domain.domain[0]) for domain in domains],
                             classification="class1"),
        cc.ConjunctiveClause(feature_mask=np.array([False, False, False, True, False]),
                             feature_domains=domains,
                             features=[domain.init_feature(domain.domain[0]) for domain in domains],
                             classification="class1"),
        cc.ConjunctiveClause(feature_mask=np.array([True, False, False, False, False]),
                             feature_domains=domains,
                             features=[domain.init_feature(domain.domain[0]) for domain in domains],
                             classification="class1"),
        cc.ConjunctiveClause(feature_mask=np.array([False, False, False, False, True]),
                             feature_domains=domains,
                             features=[domain.init_feature(domain.domain[0]) for domain in domains],
                             classification="class1"),
        cc.ConjunctiveClause(feature_mask=np.array([True, False, False, True, False]),
                             feature_domains=domains,
                             features=[domain.init_feature(domain.domain[0]) for domain in domains],
                             classification="class1"),
    ]

    dnf = source.DisjunctiveClause(cc_mask=np.array([True, False, True, False, False]),
                                   cc_clauses=clauses,
                                   classification="class1")

    print()
    utils.try_assert("Describes", dnf.describes(observation=["one", 1, 6, False, 0.0], observation_class='class1'))
    utils.try_assert("Doesn't Describe", dnf.describes(observation=["one", 0, 6, False, 0.0], observation_class='class1'))
