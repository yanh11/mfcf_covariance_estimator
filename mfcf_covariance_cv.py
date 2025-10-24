import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array, check_is_fitted, _is_arraylike

from mfcf_covariance import MFCFCovariance


# assumes MFCFCovariance from your snippet is importable
# from your_module import MFCFCovariance

@dataclass(frozen=True)
class _FoldScore:
    max_clique_size: int
    fold: int
    train_n: int
    val_n: int
    score: float
    success: bool
    error: Optional[str] = None


class MFCFCovarianceCV(BaseEstimator):
    """
    Cross-validated MFCF-based covariance/precision estimator.

    This estimator selects `max_clique_size` via K-fold cross-validation by
    maximizing the validation Gaussian log-likelihood:
        score = logdet(Theta) - trace(S_val @ Theta)
    where Theta is the LOGO precision estimated on the training fold using
    `MFCFCovariance`, and S_val is the empirical covariance of the validation fold.

    Parameters
    ----------
    # Hyper-parameter to tune
    max_clique_size_grid : Iterable[int] or 'auto', default='auto'
        Candidate values for `max_clique_size`. If 'auto', a grid is formed
        as: range(max(2, min_clique_size), min(8, n_features) + 1).

    # Cross-validation
    cv : int, default=5
        Number of KFold splits.
    shuffle : bool, default=False
        Whether to shuffle samples before splitting.
    random_state : Optional[int], default=None
        Random state for the KFold when shuffle=True.

    # Scoring / stability
    assume_centered : bool, default=False
        Passed to MFCFCovariance and empirical covariance computation.
    error_score : {'raise', float}, default=np.nan
        Score to assign if a fit/eval fails for a fold/setting. If 'raise',
        the exception is raised. If a float, the error is caught and this score
        is used for that fold.

    # Parameters forwarded to MFCFCovariance
    threshold : float, default=0.0
    min_clique_size : int, default=1
    coordination_number : float, default=np.inf
    gain_function_type : {'sumsquares'}, default='sumsquares'
    store_precision : bool, default=True

    Attributes
    ----------
    best_max_clique_size_ : int
        Selected value for `max_clique_size`.
    covariance_ : ndarray of shape (n_features, n_features)
        Covariance estimated by refitting on the full dataset with the best
        `max_clique_size`.
    precision_ : ndarray of shape (n_features, n_features)
        Corresponding LOGO precision (inverse covariance).
    location_ : ndarray of shape (n_features,)
        Estimated mean (or 0 if `assume_centered=True`).
    n_features_in_ : int
        Number of features seen during fit.
    cliques_ : list of frozenset[int]
        Maximal cliques from the final refit.
    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities from the final refit.
    peo_ : list[int]
        Perfect elimination order from the final refit.
    cv_results_ : dict
        Aggregated CV diagnostics with keys:
          - 'params': list of {'max_clique_size': int}
          - 'mean_test_score': np.ndarray, shape (n_settings,)
          - 'std_test_score':  np.ndarray, shape (n_settings,)
          - 'split_test_score': list of per-split arrays
          - 'n_splits': int
    fold_scores_ : list[_FoldScore]
        Per-fold detailed records (including errors if any).
    estimator_ : MFCFCovariance
        Final refit estimator on the full data.

    Notes
    -----
    * `max_clique_size` is naturally bounded: min ≥ `min_clique_size`,
      max ≤ n_features. The grid is filtered accordingly (and deduplicated).
    * The objective matches the Gaussian log-likelihood up to an additive
      constant (independent of Theta), so it’s valid for model comparison.
    """

    def __init__(
        self,
        *,
        max_clique_size_grid: Union[Iterable[int], int] = 4,
        cv: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        # passthrough to base estimator
        threshold: float = 0.0,
        min_clique_size: int = 1,
        coordination_number: int = np.inf,
        gain_function_type: str = "sumsquares",
        assume_centered: bool = False,
        store_precision: bool = True,
        # scoring / robustness
        error_score: Union[str, float] = np.nan,
    ):
        self.max_clique_size_grid = max_clique_size_grid
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

        self.threshold = threshold
        self.min_clique_size = min_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type
        self.assume_centered = assume_centered
        self.store_precision = store_precision

        self.error_score = error_score

    # ------------------------------ utils ------------------------------
    def _iter_grid(self, n_features: int) -> List[int]:
        if _is_arraylike(self.max_clique_size_grid):
            grid = list({int(k) for k in self.max_clique_size_grid})
        else:
            lower = max(2, int(self.min_clique_size))
            upper = min(self.max_clique_size_grid, int(n_features))
            grid = list(range(lower, upper + 1))
        # filter feasibility
        grid = [k for k in grid if self.min_clique_size <= k <= n_features and k >= 1]
        if not grid:
            raise ValueError(
                "Empty `max_clique_size_grid` after feasibility filtering. "
                f"min_clique_size={self.min_clique_size}, n_features={n_features}."
            )
        grid.sort()
        return grid

    # ------------------------------ API ------------------------------

    def fit(self, X: np.ndarray, y=None):
        X = check_array(
            X,
            ensure_min_samples=max(self.cv, 2),
            ensure_min_features=1,
            dtype=[np.float64, np.float32],
        )
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.location_ = (
            np.zeros(n_features, dtype=X.dtype)
            if self.assume_centered
            else X.mean(axis=0)
        )

        grid = self._iter_grid(n_features)

        # CV splitter (KFold is standard for covariance estimation; y unused)
        kf = KFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)

        split_scores_per_param: List[List[float]] = [[] for _ in grid]
        fold_records: List[_FoldScore] = []

        for fold_idx, (tr, va) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr], X[va]
            S_val = empirical_covariance(X_va, assume_centered=self.assume_centered)

            for gi, k in enumerate(grid):
                try:
                    base = MFCFCovariance(
                        threshold=self.threshold,
                        min_clique_size=self.min_clique_size,
                        max_clique_size=k,
                        coordination_number=self.coordination_number,
                        gain_function_type=self.gain_function_type,
                        assume_centered=self.assume_centered,
                        store_precision=True,  # needed for scoring
                    )
                    base.fit(X_tr)
                    precision_ = base.precision_
                    # numerical safety: if precision has inf/nan, mark failure
                    if not np.all(np.isfinite(precision_)):
                        raise ValueError("Non-finite entries in precision_).")

                    score = log_likelihood(S_val, precision_)
                    split_scores_per_param[gi].append(score)
                    fold_records.append(
                        _FoldScore(k, fold_idx, len(tr), len(va), score, True)
                    )
                except Exception as e:
                    if self.error_score == "raise":
                        raise
                    # assign error_score, continue
                    es = float(self.error_score)
                    split_scores_per_param[gi].append(es)
                    fold_records.append(
                        _FoldScore(k, fold_idx, len(tr), len(va), es, False, str(e))
                    )
                    # Optional: warn for visibility
                    warnings.warn(
                        f"Fold {fold_idx}, max_clique_size={k} failed with: {e}. "
                        f"Assigned error_score={self.error_score}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Aggregate
        means = np.array([np.mean(s) for s in split_scores_per_param], dtype=float)
        stds = np.array([np.std(s, ddof=1) if len(s) > 1 else 0.0 for s in split_scores_per_param], dtype=float)

        # pick the best k (tie-breaker: larger k if equal mean? choose argmax, then prefer smaller std)
        best_idx = int(np.argmax(means))
        best_k = int(grid[best_idx])

        # Store CV artifacts
        self.fold_scores_ = fold_records
        self.cv_results_ = {
            "params": [{"max_clique_size": int(k)} for k in grid],
            "mean_test_score": means,
            "std_test_score": stds,
            "split_test_score": split_scores_per_param,
            "n_splits": self.cv,
        }
        self.best_max_clique_size_ = best_k

        # Refit on all data with best parameter
        final = MFCFCovariance(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=best_k,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
            assume_centered=self.assume_centered,
            store_precision=self.store_precision,
        )
        final.fit(X)

        # mirror attributes for sklearn-style access
        self.covariance_ = final.covariance_
        self.precision_ = final.precision_
        self.cliques_ = final.cliques_
        self.separators_count_ = final.separators_count_
        self.peo_ = final.peo_
        self.estimator_ = final
        return self

    # Convenience helpers to match sklearn covariance API
    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_
