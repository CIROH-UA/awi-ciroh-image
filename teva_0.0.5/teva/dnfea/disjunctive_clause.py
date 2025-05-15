""" Disjunctive Normal Form Clause """
from __future__ import annotations

import logging

from itertools import compress
import numpy as np

from teva.ccea import conjunctive_clause
from teva.base import clause as base_clause

class DisjunctiveClause(base_clause.Clause):
    """ A disjunctive normal form clause (DNF), or disjunctive clause, represents a set of conjunctive clauses that
    describe a data class :math:`k`. This clause is mutated and modified over the course of the evolutionary algorithm
    to change

    :param cc_mask: A boolean array of length :math:`N_{CC,k}`, where :math:`N_{CC,k}` is the number of conjunctive
                    clauses (CCs) archived by the CCEA for outcome class :math:`k`. The binary values encode the
                    presence (``True``) or absence (``False``) of a given CC in the DNF.
    :param cc_clauses: The list of clauses present in the CC.
    :param classification: The data class, :math:`k`, described by the DNF.
    :param age: The age layer that the DNF belongs to.
    """
    def __init__(self,
                 cc_mask: np.ndarray,
                 cc_clauses: list[conjunctive_clause.ConjunctiveClause],
                 classification,
                 age: int = 0):
        if not cc_mask.shape[0] == len(cc_clauses):
            raise ValueError("'cc_mask' must be the same length as 'cc_clauses'")

        if classification is None:
            raise ValueError("'classification' must not be None.")

        super().__init__(mask=cc_mask,
                         items=cc_clauses,
                         classification=classification,
                         age=age)


    def calc_coverage(self,
                      observation_table: np.ndarray,
                      classifications: np.ndarray = None):
        """Calculate the observation coverage of the DNF.

        :param observation_table: The table of observations.
        :param classifications: An array of the classification of each observation row

        :return: The array of observations described by the DNF.
        """
        active_ccs = compress(self.items, self.mask)

        active_masks = []
        # use the coverage masks of each of the ccs to determine coverage (much faster than recalculating every time)
        # for reference, this took about 0.5s per run before.  Now I can't even see it without using a timer
        for clause in active_ccs:
            active_masks.append(clause.coverage_mask)

        if len(active_masks) <= 0:
            logging.getLogger("dnfea").error("No active ccs")
            return

        self.coverage_mask = np.any(np.vstack(active_masks), 0)
        self.target_mask = self.items[0].target_mask
        self.nontarget_mask = self.items[0].nontarget_mask

        self._calc_masks(observation_table)

    def covers(self, observation: list | np.ndarray) -> bool:
        for i in range(self.mask.shape[0]):
            if self.mask[i] and self.items[i].describes(observation, self.classification):
                return True
        return False

    def __str__(self):
        return f"DNF: order={self.order()}/{self.mask.shape[0]} class={self.classification} age={self.age}"

    @staticmethod
    def create_empty(mask: np.ndarray = None):
        """ Creates an empty disjunctive clause with the given mask.  This has no real functionality, and is
        intended for testing only.

        :param mask: The cc mask to give the empty array

        :return: An empty disjunctive clause
        """
        if mask is None:
            mask = np.zeros(1)

        clauses = [conjunctive_clause.ConjunctiveClause.create_empty() for _ in range(mask.shape[0])]
        clause = DisjunctiveClause(mask, clauses, 0)

        return clause

class DNF(DisjunctiveClause):
    """ Shorthand for :class:`DisjunctiveClause` """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
