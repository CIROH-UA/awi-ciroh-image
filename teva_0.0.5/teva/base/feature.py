""" Classes related to Features and Feature Domains """
from __future__ import annotations
import abc
from enum import Enum
import numbers
from abc import abstractmethod

import numpy as np

from teva.utilities import teva_math

class FeatureType(Enum):
    """ Represents the type of :class:`Feature`

    The values of a feature domain that is numeric will be sorted in ascending order.  Mutation of the features
    will either extend or retract the range of allowed values, while leaving the order intact.  Categorical features
    will not be sorted and will be mutated at random.

    :param ORDINAL: A numeric feature; ordinal represents a feature whose values are integers.
    :param CONTINUOUS: A numeric feature; continuous represents a feature whose values are floating.
    :param CATEGORICAL: A categorical feature; categorical represents a feature whose values can be of any type, and
        independent of one another.  Categorical features are commonly nominal (strings), but can be of any input type.
    """
    ORDINAL = 0
    CONTINUOUS = 1
    CATEGORICAL = 2


# create global constants for easy access
ORDINAL = FeatureType.ORDINAL
CONTINUOUS = FeatureType.CONTINUOUS
CATEGORICAL = FeatureType.CATEGORICAL


class Feature(abc.ABC):
    """ A generic Feature, which is a subset of the associated feature domain.

    :param feature_domain: The domain associated with this feature, which defines its possible values
    :param feature_idxs: The indices of the domain values that are contained in this feature
    """
    def __init__(self, feature_domain: FeatureDomain, feature_idxs: np.ndarray):
        self.feature_domain = feature_domain
        self._feature_idxs = feature_idxs

    def __eq__(self, other):
        return isinstance(other, Feature) and \
            self.feature_domain == other.feature_domain and \
            self._feature_idxs.shape == other._feature_idxs.shape and \
            np.all(self._feature_idxs == other._feature_idxs)

    def __contains__(self, item):
        feature_set = self.feature_set()
        if isinstance(feature_set, np.ndarray):
            return np.any(feature_set == item)
        return item in self.feature_set()

    def __len__(self):
        return self._feature_idxs.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.feature_set()[item]
        raise IndexError("Feature indices must be integers")

    def feature_set(self) -> np.array | list:
        """ Returns the list or array of values in this feature's feature set

        :return: The feature set of this feature
        """
        return self.feature_domain[self._feature_idxs]

    def feature_type(self) -> FeatureType:
        """ Returns the feature type of this feature

        :return: The feature type of this feature (the feature type of its domain)
        """
        return self.feature_domain.feature_type

    def __str__(self) -> str:
        if self.feature_domain.feature_type == FeatureType.CATEGORICAL:
            type_string = "CATEGORICAL"
        elif self.feature_domain.feature_type == FeatureType.ORDINAL:
            type_string = "ORDINAL"
        elif self.feature_domain.feature_type == FeatureType.CONTINUOUS:
            type_string = "CONTINUOUS"
        else:
            type_string = "UNKNOWN-TYPE"

        return f"Feature [{type_string}]: {self.feature_set()}"

    def __hash__(self):
        return hash(str(self._feature_idxs.tolist()))

    @staticmethod
    def create_empty(feature_type: FeatureType = FeatureType.CATEGORICAL) -> Feature:
        """ Creates an empty instance of a feature domain, intended to be used in testing.

        :param feature_type: The feature type of the empty domain

        :return: An empty feature domain
        """
        domain = FeatureDomain.create_empty(feature_type)
        if feature_type == FeatureType.CATEGORICAL:
            return CategoricalFeature([False], domain)

        return NumericalFeature(np.array([0]), domain)


class NumericalFeature(Feature):
    """ A feature that contains numerical values that can be sorted

    :param feature_set: The domain subset of this feature
    :param feature_domain: The domain of this feature
    """
    def __init__(self,
                 feature_set: np.ndarray,
                 feature_domain: NumericalFeatureDomain):
        if feature_set.shape[0] == 0:
            raise ValueError("'feature_set' cannot be an empty array.")

        idxs = np.flatnonzero(np.isin(feature_domain.domain, feature_set))

        super().__init__(feature_domain, idxs)

    def __copy__(self):
        return NumericalFeature(self.feature_set(), self.feature_domain)

    def get_bounds(self) -> (int, int):
        """ Returns the upper and lower founds of this numerical feature

        :return: (The lowest value in the feature set, The highest value in the feature set)
        """
        return self._feature_idxs[0], self._feature_idxs[-1] + 1

    def set_lower_bound(self, new_index: int):
        """ Set the lower bound of the feature indices to a new index

        :param new_index: The feature index of the new lower bound
        """
        self._feature_idxs = np.arange(new_index, self._feature_idxs[-1] + 1)

    def set_upper_bound(self, new_index: int):
        """ Set the upper bound of the feature indices to a new index

        :param new_index: The feature index of the new upper bound
        """
        self._feature_idxs = np.arange(self._feature_idxs[0], new_index + 1)


class CategoricalFeature(Feature):
    """ A feature that contains categorical values that are mutually exclusive and are not relative to one another

    :param feature_set: The domain subset of this feature
    :param feature_domain: The domain of this feature
    """
    def __init__(self, feature_set: list | np.ndarray, feature_domain: CategoricalFeatureDomain):
        # force categorical set to a list
        if isinstance(feature_set, np.ndarray):
            feature_set = feature_set.tolist()

        def explicit_index(val, lst) -> int | None:
            for i, element in enumerate(lst):
                if isinstance(val, np.bool_):
                    val = bool(val)
                if isinstance(val, type(element)) and val == element:
                    return i
            return None

        idxs: list[int] = []
        for value in feature_set:
            # this function avoids mistaking ints for bools
            idx = explicit_index(value, feature_domain.domain)
            if idx is not None:
                idxs.append(idx)

        super().__init__(feature_domain, np.array(idxs))

    def __copy__(self):
        return CategoricalFeature(self.feature_set(), self.feature_domain)

    def remove_random_value(self):
        """ Removes a random value from the feature set """
        if self._feature_idxs.shape[0] == 1:
            raise RuntimeError("Cannot remove value. This feature contains a single value.")

        selected = teva_math.Rand.RNG.choice(self._feature_idxs)
        self._feature_idxs = np.delete(self._feature_idxs, np.isin(self._feature_idxs, selected))
        return self.feature_domain.domain[selected]

    def add_random_value(self):
        """ Adds a random value from the domain to the feature set """
        available = np.setdiff1d(np.arange(0, len(self.feature_domain)), self._feature_idxs, assume_unique=True)

        if available.shape[0] == 0:
            raise RuntimeError(
                "Cannot add value to feature."
                " This feature already contains all values in the feature domain"
            )

        selected = teva_math.Rand.RNG.choice(available)

        self._feature_idxs = np.append(self._feature_idxs, selected)
        return self.feature_domain.domain[selected]


# FEATURE DOMAINS
class FeatureDomain(abc.ABC):
    """ Represents the domain of a feature present in a set of input observations, where the domain is the
    array of values that are present in the input observations, and therefore possible for a feature with this domain
    to use.  :class:`FeatureDomain` is intended to be an abstract class, and is extended by
    :class:`NumericalFeatureDomain` and :class:`CategoricalFeatureDomain`, as their domains are treated slightly
    differently.

    :param domain: An array of values representing the entire set of values a :class:`Feature` that has this domain can
        contain.
    :type domain: np.array
    :param feature_type: The type of any features that have this domain.
    :type feature_type: FeatureType(Enum)
    """
    def __init__(self, domain: np.array | list, feature_type: FeatureType):
        self.domain = domain
        self.feature_type = feature_type

    def __len__(self):
        if isinstance(self.domain, np.ndarray):
            return self.domain.shape[0]

        if isinstance(self.domain, list):
            return len(self.domain)

        raise ValueError(f"feature domain is neither a list or an ndarray.  Actual: {type(self.domain)}")

    @abstractmethod
    def __contains__(self, item):
        """ Returns True if `item` is a member of the `domain` """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    def __str__(self) -> str:
        if self.feature_type == FeatureType.CATEGORICAL:
            typestring = "CATEGORICAL"
        elif self.feature_type == FeatureType.ORDINAL:
            typestring = "ORDINAL"
        elif self.feature_type == FeatureType.CONTINUOUS:
            typestring = "CONTINUOUS"
        else:
            typestring = "UNKNOWN-TYPE"

        return f"FeatureDomain [{typestring}]: {self.domain}"

    @abstractmethod
    def init_feature(self, value=None) -> Feature:
        """Initializes a feature object with a random subset of the original feature's ``feature_domain`` domain

        :return: A feature object with a subset of the original feature's domain
        :rtype: Feature
        """
        raise NotImplementedError

    @staticmethod
    def create_empty(feature_type: FeatureType = FeatureType.CATEGORICAL) -> FeatureDomain:
        """ Creates an empty feature domain for use in unit testing

        :param feature_type: The feature type of the empty domain
        :return: An empty feature domain
        """
        if feature_type == FeatureType.CATEGORICAL:
            return CategoricalFeatureDomain([True, False])

        return NumericalFeatureDomain([0, 1], feature_type)


class NumericalFeatureDomain(FeatureDomain):
    """ Represents the domain of a numerical feature present in a set of input observations, where the domain is the
    array of values that are present in the input observations, and therefore possible for a feature with this domain
    to use.

    See documentation for :class:`FeatureDomain` for more information.

    :param domain: An array of values representing the entire set of values a :class:`Feature` that has this domain can
        contain.  This array should contain only numerical values, and will be sorted automatically.
    :type domain: np.array
    :param feature_type: The type of any features that have this domain.
    :type feature_type: FeatureType(Enum)

    """
    def __init__(self, domain: np.array, feature_type: FeatureType):
        sorted_domain = np.sort(domain)
        super().__init__(sorted_domain, feature_type)

    def __contains__(self, item: np.number | NumericalFeature):
        if isinstance(item, NumericalFeature):
            return np.all(np.isin(item.feature_set(), self.domain))
        return np.any(self.domain == item)

    def __getitem__(self, item):
        if isinstance(item, list):
            return self.domain(np.array(item))

        if isinstance(item, np.ndarray):
            return self.domain[item]

        if isinstance(item, int):
            return self.domain[item]

        raise IndexError("Domain indices must be an integer or array-like of integers")

    # override
    def init_feature(self, value=None) -> NumericalFeature:
        """ Creates a new feature using this domain as a base.  By default, it will add one random value from the domain
        to the feature set of the new feature.

        :param value: If specified, this specific value will be added instead of by random choice

        :return: The newly created feature
        """
        if value is None:
            # min index is any index from
            if self.domain.shape[0] == 1:
                min_index = 0
                max_index = 1
            else:
                min_index = teva_math.Rand.RNG.integers(self.domain.shape[0] - 1)
                # max index is any index from min_index to set.max
                max_index = teva_math.Rand.RNG.integers(min_index + 1, self.domain.shape[0])
            return NumericalFeature(feature_set=self.domain[min_index:max_index],
                                    feature_domain=self)

        if value in self:
            return NumericalFeature(np.array([value]), self)

        raise ValueError("Requested value is not in the domain.")


class CategoricalFeatureDomain(FeatureDomain):
    """ Represents the domain of a categorical feature present in a set of input observations, where the domain is the
    array of values that are present in the input observations, and therefore possible for a feature with this domain
    to use.

    See documentation for :class:`FeatureDomain` for more information.

    :param domain: A list of values representing the entire set of values a :class:`Feature` that has this domain can
        contain.  This list can contain any values in any order, as the values are intended to be distinct.
    :type domain: list

    TODO: Do categories need to be unique?  If so we should enforce that, or explain we will remove duplicates.
    """
    def __init__(self, domain: list | np.ndarray):
        # force categorical domain to a list
        if isinstance(domain, np.ndarray):
            domain = domain.tolist()
        super().__init__(domain, FeatureType.CATEGORICAL)

    def __contains__(self, item):
        return item in self.domain

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self.domain[i] for i in item]
        if isinstance(item, np.ndarray):
            return [self.domain[i] for i in item]
        if isinstance(item, int):
            return self.domain[item]

        raise IndexError("Domain indices must be an integer or array-like of integers")

    # override
    def init_feature(self, value=None) -> CategoricalFeature:
        """ Creates a new feature using this domain as a base.  By default, it will add one random value from the domain
        to the feature set of the new feature.

        :param value: If specified, this specific value will be added instead of by random choice

        :return: The newly created feature
        """
        if value is None:
            index = teva_math.Rand.RNG.integers(0, len(self.domain))
            return CategoricalFeature([self.domain[index]], self)

        if value in self:
            feature_set = [value]
            return CategoricalFeature(feature_set, self)

        raise ValueError("Requested value is not in the domain.")


def find_feature_domains(observations: np.ndarray, feature_types: [FeatureType]):
    """ Ingests raw observation data and produces arrays for running the CCEA

    FeatureDomain Types:
        -  :class:`feature.FeatureType`.ORDINAL
        -  :class:`feature.FeatureType`.CONTINUOUS
        -  :class:`feature.FeatureType`.CATEGORICAL

    :param observations: An observation matrix where each column represents a feature, and each row represents
        an observation
    :param feature_types: An array of feature types corresponding to each column in the observation matrix

    :return:
        - A list of feature objects corresponding to each feature column
        - A list of unique classes present in the `classes` vector
    """
    # get an array of unique values for each feature column
    unique_values = [np.unique(column) for column in observations.T]

    feature_domains: list[FeatureDomain] = []
    # for each feature column, create a feature based on its dtype and unique values
    for i, feature_type in enumerate(feature_types):
        # if every value is integer-like, it's an integer row
        if feature_type == FeatureType.CATEGORICAL:
            # if its categorical, add it as a categorical feature
            feature_domains.append(CategoricalFeatureDomain(domain=unique_values[i].tolist()))
        else:
            # if it is integer or floating, add it as a numerical feature
            feature_domains.append(NumericalFeatureDomain(unique_values[i], feature_type))

    return feature_domains


def determine_feature_types(observations: np.ndarray) -> list[FeatureType]:
    """Determine the feature types of the observation table

    :param observations: The table of observations present in the input data

    .. todo::
        - Check how we handle having more columns than `num_features`
        - Add option to have int-like continuous features
    """
    # vectorized function to determine if the columns are numerical or categorical
    vectorized_is_number = np.vectorize(
        lambda x: (np.isnan(x) or isinstance(x, numbers.Number)) and not isinstance(x, bool))
    # vectorized function to determine if the columns are integers or floats. We
    # will naively assume that if a column contains only integers, it is an ordinal
    # feature
    vectorized_is_int = np.vectorize(lambda x: np.isnan(x) or isinstance(x, numbers.Integral))

    # determine if the columns are numerical or categorical
    numerical_mask = np.all(vectorized_is_number(observations), axis=0)

    numerical_cols = np.nonzero(numerical_mask)[0]

    feature_types = []

    for idx in range(observations.shape[1]):
        if idx in numerical_cols:
            if np.all(vectorized_is_int(observations[idx])):
                feature_types.append(FeatureType.ORDINAL)
            else:
                feature_types.append(FeatureType.CONTINUOUS)
        else:
            feature_types.append(FeatureType.CATEGORICAL)

    return feature_types
