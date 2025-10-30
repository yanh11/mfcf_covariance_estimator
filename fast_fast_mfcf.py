import heapq
import itertools
import logging
from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import numpy.linalg as LA

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Example usage from caller:
# logging.basicConfig(level=logging.INFO)  # or DEBUG


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Node = int
Clique = FrozenSet[int]
Separator = FrozenSet[int]


# -----------------------------------------------------------------------------
# Separator wrapper for comparison in priority queue
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SeparatorWrapper:
    separator: FrozenSet[int]
    separator_prior_threshold: FrozenSet[int]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SeparatorWrapper):
            return self.separator == other.separator
        return NotImplemented

    def __le__(self, other: object) -> bool:  # subset-or-equal comparison
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator <= other.separator
                or self.separator_prior_threshold <= other.separator_prior_threshold
            )
        return NotImplemented

    def __ge__(self, other: object) -> bool:  # superset-or-equal comparison
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator >= other.separator
                or self.separator_prior_threshold >= other.separator_prior_threshold
            )
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator < other.separator
                or self.separator_prior_threshold < other.separator_prior_threshold
            )
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator > other.separator
                or self.separator_prior_threshold > other.separator_prior_threshold
            )
        return NotImplemented
# -----------------------------------------------------------------------------
# Helper formatting (debug/pretty-print utilities)
# -----------------------------------------------------------------------------
def format_frozenset(fs: Iterable[int]) -> str:
    """Convert a frozenset (or any iterable) to readable, sorted list format."""
    try:
        return str(sorted(int(x) for x in fs)) if fs else "[]"
    except (ValueError, TypeError):
        return str(sorted(list(fs))) if fs else "[]"


# =============================================================================
# Gains
# =============================================================================
class Gains:
    """Gain function handler (currently supports 'sumsquares')."""

    def __init__(
        self,
        C: np.ndarray,
        threshold: float = 0.0,
        min_clique_size: int = 1,
        gf_type: str = "sumsquares",
    ):
        if gf_type == "sumsquares":
            self._W = np.square(C)
        else:
            raise ValueError(f"Unknown gain function type: {gf_type}")

        self._threshold: float = threshold
        self._min_clique_size: int = min_clique_size

    def get_best_gain(
        self,
        outstanding_nodes_mask: np.ndarray,
        sep: "Separator",
    ) -> Tuple[float, "Node", "SeparatorWrapper"]:
        """
        Vectorized: compute best gain over all outstanding nodes for a given separator.
        """
        if not sep:
            # With empty sep, all gains are 0; choose the first outstanding node
            first = int(outstanding_nodes_mask.argmax())
            empty = frozenset()
            return 0.0, first, SeparatorWrapper(empty, empty)

        # 1) Prepare indices/submatrix and mandatory-picks k
        rows, cols, W_sub = self._prepare_submatrix(outstanding_nodes_mask, sep)

        # number of mandatory picks from top weights
        k = max(0, min(self._min_clique_size - 1, W_sub.shape[1]))

        # 2) Compute row-wise gains and masks needed for reconstruction
        gains, T, topk_idx, topk_in_T = self._row_gains(W_sub, self._threshold, k)

        # 3) Pick best row and reconstruct the kept separator columns for that row
        best_row_pos = int(np.argmax(gains))
        best_node = int(rows[best_row_pos])
        best_gain = float(gains[best_row_pos])

        best_sep = self._separator_for_row(
            cols=cols,
            T_row=T[best_row_pos],
            row_topk_idx=topk_idx[best_row_pos],
            row_topk_in_T=topk_in_T[best_row_pos],
            k=k,
            sep=sep,
        )

        return best_gain, best_node, best_sep

    # --------------------
    # Helpers
    # --------------------

    def _prepare_submatrix(
        self,
        outstanding_nodes_mask: np.ndarray,
        sep: "Separator",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.flatnonzero(outstanding_nodes_mask)
        cols = np.asarray(list(sep), dtype=int)

        W_sub = self._W[np.ix_(rows, cols)]
        return rows, cols, W_sub

    def _row_gains(
        self,
        W_sub: np.ndarray,
        threshold: float,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = W_sub >= threshold
        gains = np.where(T, W_sub, 0.0).sum(axis=1)

        if k:
            topk_idx = np.argpartition(W_sub, -k, axis=1)[:, -k:]
            topk_vals = np.take_along_axis(W_sub, topk_idx, axis=1)
            topk_in_T = np.take_along_axis(T, topk_idx, axis=1)
            gains += np.where(~topk_in_T, topk_vals, 0.0).sum(axis=1)
        else:
            # Keep shapes consistent
            R = W_sub.shape[0]
            topk_idx = np.empty((R, 0), dtype=int)
            topk_in_T = np.empty((R, 0), dtype=bool)

        return gains, T, topk_idx, topk_in_T

    def _separator_for_row(
        self,
        cols: np.ndarray,
        T_row: np.ndarray,
        row_topk_idx: np.ndarray,
        row_topk_in_T: np.ndarray,
        k: int,
        sep: "Separator",
    ) -> "SeparatorWrapper":
        keep_mask = T_row.copy()
        if k:
            keep_mask[row_topk_idx[~row_topk_in_T]] = True

        kept = frozenset(cols[keep_mask])
        return SeparatorWrapper(kept, frozenset(sep))


# =============================================================================
# MFCF
# =============================================================================
class MFCF:
    """Maximum Filtering Clique Forest (MFCF) builder."""

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        threshold: float = 0.0,
        min_clique_size: int = 1,
        max_clique_size: int = 4,
        coordination_number: int = np.inf,
        gain_function_type: str = "sumsquares",
    ) -> None:

        self._threshold = threshold
        self._min_clique_size = min_clique_size
        self._max_clique_size = max_clique_size
        self._coordination_number = coordination_number
        self._gf_type = gain_function_type

    def run(
        self,
        C: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[List[Clique], Counter, List[Node], np.ndarray]:
        """
        Run the MFCF/TMFG process and return:
          cliques, separators_count, perfect_elimination_order (peo), J_logo
        """
        self._C = C
        self._gf = Gains(
            C, self._threshold, self._min_clique_size, self._gf_type
        ).get_best_gain

        self._initialise()
        self._compute_mfcf()

        matrix_for_logo = cov_matrix if cov_matrix is not None else C
        J_logo = self._logo(matrix_for_logo, self._cliques, self._separators_count)
        return self._cliques, self._separators_count, self._peo, J_logo

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------
    def _initialise(self) -> None:
        """Prepare data structures and seed clique."""
        self._gains_pq: List[Tuple[float, Node, SeparatorWrapper]] = []
        self._iteration = 0

        first_cl = self._get_first_clique()

        self._cliques: List[Clique] = [first_cl]
        self._remaining_nodes_count = self._C.shape[0] - len(first_cl)

        self._separators_count: Counter = Counter()
        self._pq_separators: set[Separator] = set()

        self._peo: List[Node] = [v for v in first_cl]  # Perfect elimination order
        self._outstanding_nodes_mask = np.ones(self._C.shape[0], dtype=bool)
        self._outstanding_nodes_mask[list(first_cl)] = False

        self._log_initial_state(first_cl)

        self._process_new_clique_gains(first_cl)

    def _get_first_clique(self, first: int = 1) -> Clique:
        """Seed with node(s) having high sum of weights above-mean edges."""
        C1 = self._C.copy()
        r, c = np.nonzero(self._C <= self._C.mean())
        C1[r, c] = 0
        sums = C1.sum(axis=0)
        cand = np.argsort(-sums, kind="stable")
        return frozenset(cand[:(self._min_clique_size - 1)])

    # -------------------------------------------------------------------------
    # Main algorithm loop
    # -------------------------------------------------------------------------
    def _compute_mfcf(self) -> None:
        """Greedy loop popping best (gain, v, sep) and updating structures."""
        while self._remaining_nodes_count > 0:
            self._iteration += 1
            if self._gains_pq:
                gain, v, sep_wrapper = self._pop_from_pq()
                sep = sep_wrapper.separator
                if self._should_skip_candidate(gain, v, sep_wrapper):
                    continue
            else:
                # No candidates left; force a new clique
                sep = self._cliques[-1]
                gain, v, sep_wrapper = (
                    0.0,
                    int(self._outstanding_nodes_mask.argmax()),
                    SeparatorWrapper(sep, sep),
                )

            v, sep, parent_clique = self._apply_threshold_and_find_parent(gain, v, sep)
            cliques_before = list(self._cliques)
            new_clique = self._add_new_clique(parent_clique, sep, v)

            self._remaining_nodes_count -= 1
            self._check_proposed_separator(sep_wrapper, cliques_before)

            if self._remaining_nodes_count == 0:
                break

            self._update_pq_for_new_separator(sep_wrapper.separator_prior_threshold)
            self._process_new_clique_gains(new_clique)

    # -------------------------------------------------------------------------
    # Candidate checks & parent search
    # -------------------------------------------------------------------------
    def _should_skip_candidate(
        self, gain: float, v: Node, sep_wrapper: SeparatorWrapper
    ) -> bool:
        """Filter heap candidates by availability, size, multiplicity, and validity."""
        # If drop_sep is enabled, disable candidates with a seen/used separator.
        sep = sep_wrapper.separator
        # multiplicity constraint
        if self._separators_count[sep] > self._coordination_number:
            return True
        if np.isnan(gain):
            return True
        if not self._outstanding_nodes_mask[v]:
            # v already used, recompute best gain for this sep and push back to heap
            self._update_pq_for_new_separator(sep_wrapper.separator_prior_threshold)
            return True
        # length constraint
        if not (
            len(sep) >= self._min_clique_size - 1 and len(sep) < self._max_clique_size
        ):
            return True
        # subset-of-some-current-clique constraint
        if not any(sep.issubset(clq) for clq in self._cliques):
            return True
        return False

    def _apply_threshold_and_find_parent(
        self, gain: float, v: Node, sep: Separator
    ) -> Tuple[Node, Separator, Optional[Clique]]:
        """Decide if we start a new component (below threshold) or attach to a parent clique."""
        pos_gain = -gain  # negate back to positive for threshold compare
        if pos_gain < self._threshold:
            # start a new clique
            v = int(self._outstanding_nodes_mask.argmax())
            sep = frozenset()
            parent_clique = frozenset()
        else:
            parent_clique = self._find_parent_clique_for_separator(sep)
        return v, sep, parent_clique

    def _find_parent_clique_for_separator(self, sep: Separator) -> Optional[Clique]:
        """Find a current clique that contains sep."""
        for clq in self._cliques:
            if sep <= clq:
                return clq
        return None

    # -------------------------------------------------------------------------
    # Clique/separator updates
    # -------------------------------------------------------------------------
    def _add_new_clique(
        self, parent_clique: Optional[Clique], sep: Separator, v: Node
    ) -> Clique:
        """Add new clique, keep only maximal cliques, and update PEO/outstanding."""
        new_clique: Clique = frozenset(sep | {v})
        self._peo.append(v)
        self._outstanding_nodes_mask[v] = False

        self._log_added_clique(v, new_clique, parent_clique, sep)

        # keep only maximal cliques (drop strict subsets of the new one)
        if len(new_clique) > 1:
            to_remove = [c for c in self._cliques if c < new_clique]
            for c in to_remove:
                self._cliques.remove(c)

        self._cliques.append(new_clique)
        return new_clique

    def _check_proposed_separator(
        self,
        separator_wrapper: SeparatorWrapper,
        cliques_before: List[Clique],
    ) -> None:
        """
        Consider a proposed separator and record it if it:
          - is non-empty,
          - has length in [min_clique_size - 1, max_clique_size),
          - is not a superset (or equal) of any existing clique,
          - and has not exceeded the multiplicity cap.

        If applicable, also enqueue it for potential reuse.
        """
        sep = separator_wrapper.separator
        if not sep:
            # Empty separator, nothing to record
            return

        if not (self._min_clique_size - 1 <= len(sep) < self._max_clique_size):
            return

        # Must NOT be a superset (or equal) of any prior clique
        not_superset_of_any_prior = not any(sep >= clique for clique in cliques_before)

        recorded = False
        under_multiplicity_cap = self._separators_count[sep] < self._coordination_number
        if not under_multiplicity_cap:
            self._log_processed_separator(sep, recorded)
            return

        if not_superset_of_any_prior:
            self._separators_count[sep] += 1
            recorded = True

        # We might reuse the same separator; keep PQ updated
        if self._remaining_nodes_count != 0:
            self._update_pq_for_new_separator(
                separator_wrapper.separator_prior_threshold
            )

        self._log_processed_separator(sep, recorded)

    def _process_new_clique_gains(self, clq: Clique) -> None:
        """Push gain candidates for all facets of the clique vs outstanding nodes."""
        clique = tuple(sorted(clq))
        clique_size = len(clq)

        facets = (
            [clique]
            if clique_size < self._max_clique_size
            else list(itertools.combinations(clique, clique_size - 1))
        )
        for facet in facets:
            self._update_pq_for_new_separator(frozenset(facet))

    # -------------------------------------------------------------------------
    # priority queue management
    # -------------------------------------------------------------------------
    def _update_pq_for_new_separator(self, sep: Separator) -> None:
        """Recompute and push best gain for a new separator if not already in PQ."""
        if sep in self._pq_separators:
            return
        gain, v, ranked_sep = self._gf(self._outstanding_nodes_mask, sep)
        self._push_to_pq(gain, v, ranked_sep)

    def _pop_from_pq(self):
        gain, v, sep_wrapper = heapq.heappop(self._gains_pq)
        self._pq_separators.remove(sep_wrapper.separator_prior_threshold)
        return gain, v, sep_wrapper

    def _push_to_pq(self, gain: float, v: Node, sep_wrapper: SeparatorWrapper):
        heapq.heappush(self._gains_pq, (-gain, v, sep_wrapper))
        self._pq_separators.add(sep_wrapper.separator_prior_threshold)

    # -------------------------------------------------------------------------
    # logo computation
    # -------------------------------------------------------------------------
    def _logo(
        self, C: np.ndarray, cliques: List[Clique], separators: Counter
    ) -> np.ndarray:
        """Compute sparse inverse estimator via cliques minus separators."""
        J = np.zeros(C.shape)
        # For each clique, add the inverse of the submatrix defined by the clique indices.
        for clq in cliques:
            clqt = tuple(clq)
            J[np.ix_(clqt, clqt)] += LA.inv(C[np.ix_(clqt, clqt)])

        # For each separator, subtract the inverse of the submatrix defined by the separator indices.
        for sep, mult in separators.items():
            if sep:  # non-empty
                sept = tuple(sep)
                J[np.ix_(sept, sept)] -= mult * LA.inv(C[np.ix_(sept, sept)])

        return J

    # -------------------------------------------------------------------------
    # Debug logging
    # -------------------------------------------------------------------------
    def _log_initial_state(self, first_cl: Clique) -> None:
        logger.info("  Seed clique: %s", format_frozenset(first_cl))
        logger.info("  Selected based on gain function maximization")
        logger.info("  Remaining nodes: %d", self._remaining_nodes_count)
        logger.info("---")

    def _log_added_clique(
        self,
        v: Node,
        new_clique: Clique,
        parent_clique: Optional[Clique],
        sep: Separator,
    ) -> None:
        logger.info("Iteration %d", self._iteration)
        logger.info("  Added vertex: %s", v)
        logger.info("  Proposed sub-clique: %s", format_frozenset(sep))
        logger.info(
            "  Parent clique: %s",
            format_frozenset(parent_clique) if parent_clique else None,
        )
        logger.info("  New clique: %s", format_frozenset(new_clique))

    def _log_processed_separator(
        self, proposed_separator: Separator, separator_recorded: bool
    ) -> None:
        minc = self._min_clique_size
        maxc = self._max_clique_size
        if len(proposed_separator) == 0:
            logger.info("  → No separator recorded (empty proposed sub-clique)")
        elif not separator_recorded:
            if len(proposed_separator) >= maxc or len(proposed_separator) < (minc - 1):
                logger.info(
                    "  → Separator NOT recorded (size constraints: %d not in [%d, %d])",
                    len(proposed_separator),
                    minc - 1,
                    maxc - 1,
                )
            else:
                logger.info(
                    "  → Separator NOT recorded (proposed sub-clique equals existing clique - not proper)"
                )
        else:
            logger.info(
                "  → Separator RECORDED: %s", format_frozenset(proposed_separator)
            )
