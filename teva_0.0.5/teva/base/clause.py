""" Base class for ConjunctiveClause and DisjunctiveClause """
from typing import Any
import abc
import numpy as np

class Clause(abc.ABC):
    """ Base class for Conjunctive and Disjunctive clauses

    :param mask: A binary mask that defines which of the items in ``items`` is active
    :param items: A list of items that can be activated or deactivated
    :param classification: The classification of this clause
    :param age: The age of this clause at inception
    """
    def __init__(self,
                 mask: np.ndarray,
                 items: list,
                 classification,
                 age: int):
        if len(items) != mask.shape[0]:
            raise ValueError("'mask' and 'items' must be of the same length")

        # make sure the mask is a boolean array
        self.mask = np.array(mask, dtype=bool)
        self.items = items

        self.classification = classification
        self.age = age
        self.creation_gen = 0
        self.archive_gen = 0

        self.fitness = -np.inf

        self.coverage_calculated = False

        self.target_mask: np.ndarray | None = None
        self.nontarget_mask: np.ndarray | None = None

        self.coverage_mask: np.ndarray | None = None
        self.nontarget_coverage_mask: np.ndarray | None = None
        self.target_coverage_mask: np.ndarray | None = None

        self.coverage: np.ndarray | None = None
        self.nontarget_coverage: np.ndarray | None = None
        self.target_coverage: np.ndarray | None = None

        self.coverage_count: int = 0
        self.nontarget_coverage_count: int = 0
        self.target_coverage_count: int = 0
        self.target_count: int = 0
        self.nontarget_count: int = 0
        self.total_observations: int = 0

        # positive predictive value
        self.ppv: int = 0
        # coverage percent
        self.cov: int = 0

        self.fitness: float = np.inf

        self.hash_val = None

    @abc.abstractmethod
    def calc_coverage(self, observation_table: np.ndarray, classifications: np.ndarray):
        """ Calculates coverage masks and counts for this clause """
        raise NotImplementedError

    def _calc_masks(self, observation_table):
        """ Calculates masks and statistics for a clause using the observation array

        :param observation_table: The table of data observations
        """
        self.target_coverage_mask = self.coverage_mask & self.target_mask
        self.nontarget_coverage_mask = self.coverage_mask & self.nontarget_mask

        self.coverage = observation_table[self.coverage_mask]
        self.target_coverage = observation_table[self.target_coverage_mask]
        self.nontarget_coverage = observation_table[self.nontarget_coverage_mask]

        self.coverage_count = np.sum(self.coverage_mask)
        self.target_coverage_count = np.sum(self.target_coverage_mask)
        self.nontarget_coverage_count = np.sum(self.nontarget_coverage_mask)
        self.target_count = np.sum(self.target_mask)
        self.nontarget_count = np.sum(self.nontarget_mask)
        self.total_observations = observation_table.shape[0]

        self.ppv = (self.target_coverage_count / self.coverage_count) if self.coverage_count != 0 else 0
        self.cov = self.target_coverage_count / np.sum(self.target_mask)

        # generate the hash so this doesn't have to be many times.  The hash won't change
        self._calc_hash()

    @abc.abstractmethod
    def covers(self, observation: list | np.ndarray) -> bool:
        """ Calculates whether a given observation is covered by this clause """
        raise NotImplementedError

    def describes(self, observation: np.ndarray, observation_class) -> bool:
        """ Returns true if the given observation contains acceptable values for every feature that is present.
            Features that are non-present are wild-cards and can accept any value.

        :param observation: The array of values that represent on observation to test for truth
        :param observation_class: The target class associated with the observation
        :return: True if the observation is described by this clause
        """
        if self.coverage is None:
            return self.covers(observation) and self.classification == observation_class

        coverage = self.target_coverage == observation
        all = np.all(coverage | ~self.mask, axis=1)
        return np.any(all)

        # return observation in self.target_coverage

    def order(self) -> int:
        """ Returns the number of active features in this clause """
        return self.mask.sum()

    def _calc_hash(self):
        """ Calculates and saves the hash of this object so it doesn't need to be recomputed during a hash operation """
        item_hashes = [hash(item) for item in self.items]
        self.hash_val = hash(str(self.mask.tolist() + item_hashes))

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
            np.all(self.mask == other.mask) and \
            self.items == other.items

    def __len__(self) -> int:
        """ Returns the number of total items in this clause """
        return self.mask.shape[0]

    def __contains__(self, item) -> bool:
        """ Returns True if `item` is an activated item in this clause, False otherwise.

        :param item: The item to be checked
        :return: True if the `item` is in this clause and is activated.
        """
        index = np.nonzero(self.items == item)[0][0]
        return bool(self.mask[index])

    def __hash__(self):
        if self.hash_val is None:
            self._calc_hash()

        return self.hash_val

    def __str__(self):
        return f"Clause: order={self.order()}/{self.mask.shape[0]} class={self.classification} age={self.age}"

    def enable_item(self, item: int | Any):
        """ Enables the given item in the mask

        :param item: The index of the item, if an item itself is passed, the index of the item will be calculated
        """
        if isinstance(item, int):
            index = item
        # if an item is passed, find its index
        elif isinstance(item, type(self.items[0])):
            index = np.nonzero(self.items == item)[0][0]
        else:
            raise TypeError("'item' must be either an integer index, or the same type as the items in self.items")

        self.mask[index] = True

    def disable_item(self, item: int | Any):
        """ Disables the given item in the mask

        :param item: The index of the item, if an item itself is passed, the index of the item will be calculated
        """
        if isinstance(item, int):
            index = item
        # if an item is passed, find its index
        elif isinstance(item, type(self.items[0])):
            index = np.nonzero(self.items == item)[0][0]
        else:
            raise TypeError("'item' must be either an integer index, or the same type as the items in self.items")

        self.mask[index] = False

    @staticmethod
    @abc.abstractmethod
    def create_empty(mask: np.ndarray = None):
        """ Creates an empty clause to be used in testing

        :param mask: The mask of the empty clause, otherwise, it will be filled with zeros.

        :return: The newly created empty clause
        """
        raise NotImplementedError
