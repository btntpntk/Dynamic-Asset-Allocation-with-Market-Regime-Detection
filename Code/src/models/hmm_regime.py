import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM

class MarketRegimeHMM:
    """
    A Hidden Markov Model for identifying market regimes that automatically
    selects the optimal number of states (regimes) using BIC.
    """
    def __init__(self, max_states: int = 8, random_state: int = 42):
        self.max_states = max_states
        self.random_state = random_state
        self.model = None  # The best model will be stored here after fitting

    def fit(self, X: np.ndarray) -> bool:
        """
        Fits the HMM model to the data X by searching for the optimal number
        of states that minimizes the Bayesian Information Criterion (BIC).

        Returns True if a model is successfully fitted, False otherwise.
        """
        if not np.all(np.isfinite(X)):
            print("Warning: HMM training data contains NaN or Inf. Skipping fit.")
            return False

        best_bic = np.inf
        best_model = None

        # Test number of states from 2 up to max_states
        for n_components in range(2, self.max_states + 1):
            try:
                # Use n_init > 1 for more robust convergence.
                # covariance_type="full" allows capturing correlations between features.
                model = GaussianHMM(
                    n_components=n_components,
                    covariance_type="full",
                    n_iter=1000,
                    random_state=self.random_state,
                )
                model.fit(X)

                bic = model.bic(X)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model

            except ValueError as e:
                # This can happen if data is not suitable for a given n_components
                print(f"Warning: HMM with {n_components} states failed to fit. Error: {e}")
                continue
        
        if best_model is None:
            print("Warning: HMM fitting failed for all tested numbers of states.")
            return False

        self.model = best_model
        return True

    def predict_regimes(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def regime_probabilities(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)

    @property
    def n_states(self) -> int:
        """Returns the number of states of the fitted model."""
        if self.model:
            return self.model.n_components
        return 0

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)