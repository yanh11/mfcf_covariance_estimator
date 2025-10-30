import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union, Dict, Any, Set

import numpy as np
import numpy.linalg as LA
import optuna
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted, _is_arraylike

from fast_fast_mfcf import MFCF as _MFCFBuilder


class MFCFLoGo(EmpiricalCovariance):
    """
    A precision and covariance estimator with MFCF-LoGo algorithm. It constructs
    an MFCF first then use the LoGo algorithm to obtain a precision matrix.

    Precision matrix is the LoGo; estimated covariance is its inverse.

    Parameters
    ----------
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.

    min_clique_size : int, default=1
        The minimum size of cliques to form in the graph.

    max_clique_size : int, default=4
        The maximum size of cliques to form in the graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).

    gain_function_type : {'sumsquares'}, default='sumsquares'
        Gain function type to use in MFCF construction. The function is used to
        determine which vertex to add next to maximize the overall gain in the
        graph.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LoGo precision estimate.

    precision_ : ndarray of shape (n_features, n_features)
        The LoGo estimator (sparse inverse assembled from cliques and separators).

    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).

    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF.

    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in the constructed MFCF.

    peo_ : list[int]
        Perfect elimination order of vertices produced by the algorithm.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        min_clique_size: int = 1,
        max_clique_size: int = 4,
        coordination_number: int = np.inf,
        gain_function_type: str = "sumsquares",
        assume_centered: bool = False,
    ):
        super().__init__(assume_centered=assume_centered)
        self.threshold = threshold
        self.min_clique_size = min_clique_size
        self.max_clique_size = max_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type

    def fit(self, X: np.ndarray, y=None) -> "MFCFLoGo":
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(
            X,
            ensure_min_samples=2,
            ensure_min_features=1,
            dtype=[np.float64, np.float32],
        )
        # track feature count for sklearn API
        self.n_features_in_ = X.shape[1]
        self.location_ = (
            np.zeros(self.n_features_in_, dtype=X.dtype)
            if self.assume_centered
            else X.mean(axis=0)
        )

        C = np.corrcoef(X, rowvar=False)

        # run MFCF algorithm over a similarity/affinity matrix.
        builder = _MFCFBuilder(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
        )

        cliques, separators_count, peo, logo = builder.run(C=C)

        # store internals
        self.cliques_ = cliques
        self.separators_count_ = separators_count
        self.peo_ = peo

        # precision is LoGo; covariance is inverse of LoGo
        # Add small ridge if needed to avoid singularities.
        self.precision_ = logo
        if self.store_precision and not np.all(np.isfinite(self.precision_)):
            raise ValueError("Computed precision_ contains non-finite values.")

        # Robust inversion (fallback ridge if needed)
        try:
            cov = LA.inv(self.precision_)
        except LA.LinAlgError:
            # minimal Tikhonov regularization
            eps = 1e-8 * np.trace(self.precision_) / self.precision_.shape[0]
            cov = LA.inv(self.precision_ + eps * np.eye(self.precision_.shape[0]))
        self.covariance_ = cov

        return self

    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_


@dataclass(frozen=True)
class _FoldScore:
    max_clique_size: int
    fold: int
    train_n: int
    val_n: int
    score: float
    success: bool
    error: Optional[str] = None


class MFCFLoGoCV(EmpiricalCovariance):
    """
    Cross-validated MFCF-based covariance/precision estimator.

    This estimator selects `max_clique_size` via K-fold cross-validation by
    maximizing the validation Gaussian log-likelihood:
        score = logdet(Theta) - trace(S_val @ Theta)
    where Theta is the LoGo precision estimated on the training fold, and S_val
    is the empirical covariance of the validation fold.

    Parameters
    ----------
    # Hyper-parameter to tune
    max_clique_size_grid : Iterable[int] or int, default=4
        Candidate values for `max_clique_size`. When an array-like object is
        provided, it is used as the grid. When an int is provided, it defines
        the upper bound of a grid, and the grid starts from 2. For example,
        `max_clique_size_grid=6` results in the grid [2, 3, 4, 5, 6].

    # Cross-validation
    cv : int, default=5
        Number of KFold splits.

    shuffle : bool, default=False
        Whether to shuffle samples before splitting.

    random_state : Optional[int], default=None
        Random state for the KFold when shuffle=True.

    # Scoring / stability
    assume_centered : bool, default=False
        Passed to MFCFLoGo and empirical covariance computation.

    error_score : {'raise', float}, default=np.nan
        Score to assign if a fit/eval fails for a fold/setting. If 'raise',
        the exception is raised. If a float, the error is caught and this score
        is used for that fold.

    # Parameters forwarded to MFCFLoGo
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.

    min_clique_size : int, default=1
        The minimum size of cliques to form in the graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).

    gain_function_type : {'sumsquares'}, default='sumsquares'
        Gain function type to use in MFCF construction. The function is used to
        determine which vertext to add next to maximize the overall gain in the
        graph.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.


    Attributes
    ----------
    best_max_clique_size_ : int
        Selected value for `max_clique_size`.

    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LoGo precision estimate.

    precision_ : ndarray of shape (n_features, n_features)
        LoGo precision estimated by refitting on the full dataset with the best
        `max_clique_size`.

    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).

    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF.

    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in the constructed MFCF.

    peo_ : list[int]
        Perfect elimination order of vertices produced by the algorithm.

    n_features_in_ : int
        Number of features seen during fit.

    cv_results_ : dict
        Aggregated CV diagnostics with keys:
          - 'params': list of {'max_clique_size': int}
          - 'mean_test_score': np.ndarray, shape (n_settings,)
          - 'std_test_score':  np.ndarray, shape (n_settings,)
          - 'split_test_score': list of per-split arrays
          - 'n_splits': int

    fold_scores_ : list[_FoldScore]
        Per-fold detailed records (including errors if any).

    estimator_ : MFCFLoGo
        Final refit estimator on the full data.

    Notes
    -----
    * `max_clique_size` is naturally bounded: min ≥ `min_clique_size`,
      max ≤ n_features. The grid is filtered accordingly (and deduplicated).
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
        # scoring / robustness
        error_score: Union[str, float] = np.nan,
    ):
        super().__init__(assume_centered=assume_centered)
        self.max_clique_size_grid = max_clique_size_grid
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

        self.threshold = threshold
        self.min_clique_size = min_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type
        self.assume_centered = assume_centered

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
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

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

        # CV splitter
        kf = KFold(
            n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state
        )

        split_scores_per_param: List[List[float]] = [[] for _ in grid]
        fold_records: List[_FoldScore] = []

        for fold_idx, (tr, va) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr], X[va]
            S_val = empirical_covariance(X_va, assume_centered=self.assume_centered)

            for gi, k in enumerate(grid):
                try:
                    base = MFCFLoGo(
                        threshold=self.threshold,
                        min_clique_size=self.min_clique_size,
                        max_clique_size=k,
                        coordination_number=self.coordination_number,
                        gain_function_type=self.gain_function_type,
                        assume_centered=self.assume_centered,
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
        stds = np.array(
            [np.std(s, ddof=1) if len(s) > 1 else 0.0 for s in split_scores_per_param],
            dtype=float,
        )

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
        final = MFCFLoGo(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=best_k,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
            assume_centered=self.assume_centered,
        )
        final.fit(X)

        self.covariance_ = final.covariance_
        self.precision_ = final.precision_
        self.cliques_ = final.cliques_
        self.separators_count_ = final.separators_count_
        self.peo_ = final.peo_
        self.estimator_ = final
        return self

    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_


def log_likelihood_scorer(estimator, X, y=None) -> float:
    S_val = empirical_covariance(X, assume_centered=estimator.assume_centered)
    return log_likelihood(S_val, estimator.precision_)


class MFCFLoGoCVAll(EmpiricalCovariance):
    """
    Cross-validated MFCF-based covariance/precision estimator.

    This estimator selects `max_clique_size` via K-fold cross-validation by
    maximizing the validation Gaussian log-likelihood:
        score = logdet(Theta) - trace(S_val @ Theta)
    where Theta is the LoGo precision estimated on the training fold, and S_val
    is the empirical covariance of the validation fold.

    Parameters
    ----------
    # Hyperparameter tuning
    tunable_params : Iterable[str], default=()
        Names of parameters to tune with Optuna. Any subset of
        ``{"threshold", "min_clique_size", "max_clique_size", "coordination_number"}``.
        Parameters not listed here are kept fixed at the values passed to
        :class:`MFCFLoGoCVAll`.

    n_trials : int or None, default=100
        Number of Optuna trials to run. If ``None`` or ``0``, no tuning is
        performed and the estimator is refit once with the fixed parameters.

    n_jobs : int or None, default=None
        Number of parallel jobs for cross-validation scoring. Passed to
        :func:`sklearn.model_selection.cross_val_score`. ``None`` means 1;
        ``-1`` uses all processors.

    # Cross-validation
    cv : int, default=5
        Number of KFold splits.

    shuffle : bool, default=False
        Whether to shuffle samples before splitting.

    random_state : Optional[int], default=None
        Random state for the KFold when shuffle=True.

    # Scoring / stability
    assume_centered : bool, default=False
        Passed to MFCFLoGo and empirical covariance computation.

    error_score : {'raise', float}, default=np.nan
        Score to assign if a fit/eval fails for a fold/setting. If 'raise',
        the exception is raised. If a float, the error is caught and this score
        is used for that fold.

    # Parameters forwarded to MFCFLoGo
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.

    min_clique_size : int, default=1
        The minimum size of cliques to form in the graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).

    gain_function_type : {'sumsquares'}, default='sumsquares'
        Gain function type to use in MFCF construction. The function is used to
        determine which vertext to add next to maximize the overall gain in the
        graph.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.


    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LoGo precision estimate.

    precision_ : ndarray of shape (n_features, n_features)
        LoGo precision estimated by refitting on the full dataset with the best
        `max_clique_size`.

    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).

    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF.

    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in the constructed MFCF.

    peo_ : list[int]
        Perfect elimination order of vertices produced by the algorithm.

    n_features_in_ : int
        Number of features seen during fit.

    estimator_ : MFCFLoGo
        Final refit estimator on the full data.

    Notes
    -----
    * Optuna is used to optimize the mean CV log-likelihood. The sampler
        draws:
            - ``threshold`` via a continuous uniform search on ``[0, 1]`` when
              tuned.
            - ``max_clique_size`` and/or ``min_clique_size`` via integer
              log-scaled suggestions with consistency constraints
              (``min_clique_size <= max_clique_size``).
            - ``coordination_number`` via integer log-scaled suggestions.

    * ``max_clique_size`` is naturally bounded:
        ``min_clique_size <= max_clique_size <= n_features``. The search
        space is filtered/deduplicated accordingly on each trial.

    * The validation score equals the Gaussian log-likelihood up to an
        additive constant that does not depend on ``Theta``; maximizing the
        stated objective is therefore equivalent to maximizing the likelihood.

    """

    def __init__(
        self,
        *,
        tunable_params: Iterable[str] = (),
        cv: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        n_trials: Optional[int] = 100,
        # passthrough to base estimator
        threshold: float = 0.0,
        min_clique_size: int = 1,
        max_clique_size: int = 4,
        coordination_number: int = np.inf,
        gain_function_type: str = "sumsquares",
        assume_centered: bool = False,
        # scoring / robustness
        error_score: Union[str, float] = np.nan,
    ):
        super().__init__(assume_centered=assume_centered)
        self.tunable_params = tunable_params
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_trials = n_trials

        self.threshold = threshold
        self.max_clique_size = max_clique_size
        self.min_clique_size = min_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type
        self.assume_centered = assume_centered

        self.error_score = error_score

    def _process_params(self, trial: optuna.Trial, n_features: int) -> Dict[str, Any]:
        params = dict(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
        )
        if not self.tunable_params:
            return params

        if "threshold" in self.tunable_params:
            params["threshold"] = trial.suggest_float("threshold", 0.0, 1.0, step=None)

        if (
            "max_clique_size" in self.tunable_params
            and "min_clique_size" in self.tunable_params
        ):
            params["max_clique_size"] = trial.suggest_int(
                "max_clique_size", 2, min(32, n_features - 1), log=True
            )
            params["min_clique_size"] = trial.suggest_int(
                "min_clique_size", 1, params["max_clique_size"], log=True
            )
        elif "min_clique_size" in self.tunable_params:
            # Only tune min_clique_size, fixed max_clique_size
            params["min_clique_size"] = trial.suggest_int(
                "min_clique_size", 1, params["max_clique_size"], log=True
            )
        elif "max_clique_size" in self.tunable_params:
            # Only tune max_clique_size, fixed min_clique_size
            params["max_clique_size"] = trial.suggest_int(
                "max_clique_size", 2, min(32, n_features - 1), log=True
            )

        if "coordination_number" in self.tunable_params:
            params["coordination_number"] = trial.suggest_int(
                "coordination_number", 1, n_features, log=True
            )  # Needs to confirm the max coordination_number

        return params

    def _cross_validate(self, X: np.ndarray):
        def objective(trial: optuna.Trial):
            n_features = X.shape[1]
            params = self._process_params(trial, n_features)
            # Log sampled parameters before evaluating the trial
            # print(
            #     f"[Optuna] Trial {trial.number} starting with params: "
            #     + ", ".join(f"{k}={v}" for k, v in params.items()),
            #     flush=True,
            # )
            est = MFCFLoGo(**params, assume_centered=self.assume_centered)

            cv = KFold(n_splits=5, shuffle=self.shuffle, random_state=self.random_state)
            scores = cross_val_score(
                est, X, y=None, scoring=log_likelihood_scorer, cv=cv, n_jobs=self.n_jobs
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        final_params = dict(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
        )
        final_params.update(best_params)
        # print(best_params, flush=True)

        return final_params

    # ------------------------------ API ------------------------------

    def fit(self, X: np.ndarray, y=None):
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

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

        best_params = self._cross_validate(X)

        # Refit on all data with best parameter
        final = MFCFLoGo(**best_params, assume_centered=self.assume_centered)
        final.fit(X)

        self.covariance_ = final.covariance_
        self.precision_ = final.precision_
        self.cliques_ = final.cliques_
        self.separators_count_ = final.separators_count_
        self.peo_ = final.peo_
        self.estimator_ = final
        return self

    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_
