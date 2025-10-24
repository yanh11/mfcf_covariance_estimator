import numpy as np
import numpy.linalg as LA

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance._empirical_covariance import empirical_covariance
from sklearn.utils.validation import check_array, check_is_fitted

from fast_fast_mfcf import MFCF as _MFCFBuilder


class MFCFCovariance(EmpiricalCovariance):
    """
    A precision and covariance estimator with an MFCF-assembled sparse inverse ("logo").

    Precision matrix is the LOGO; estimated covariance is its inverse.

    Parameters
    ----------
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.
    min_clique_size : int, default=1
    max_clique_size : int, default=4
    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).
    gf_type : {'sumsquares'}, default='sumsquares'
    assume_centered : bool, default=False
        Passed-through behavior as in EmpiricalCovariance.
    store_precision : bool, default=True
        Whether to compute and store the precision_ attribute.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LOGO precision estimate.
    precision_ : ndarray of shape (n_features, n_features)
        The LOGO estimator (sparse inverse assembled from cliques and separators).
    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).
    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF/TMFG.
    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in LOGO assembly.
    peo_ : list[int]
        Perfect elimination order produced by the algorithm.
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
        store_precision: bool = True,
    ):
        super().__init__(
            assume_centered=assume_centered, store_precision=store_precision
        )
        self.threshold = threshold
        self.min_clique_size = min_clique_size
        self.max_clique_size = max_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type

    def fit(self, X: np.ndarray, y=None) -> "MFCFCovariance":
        """Fit the estimator from data X."""
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

        S = empirical_covariance(X, assume_centered=self.assume_centered)
        C = np.corrcoef(X, rowvar=False)

        # run MFCF algorithm over a similarity/affinity matrix.
        # Here we use the *empirical covariance* S as the input C.
        builder = _MFCFBuilder(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
        )

        cliques, separators_count, peo, logo = builder.run(C=C, cov_matrix=S)

        # store internals
        self.cliques_ = cliques
        self.separators_count_ = separators_count
        self.peo_ = peo

        # precision is LOGO; covariance is inverse of LOGO
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
