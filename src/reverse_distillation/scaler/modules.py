import os
import warnings

import fbpca
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.base import clone
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from .. import logger
from .utils import johnstone_threshold, parse_model_id

# =============================================================================
# Wrapper Classes
# =============================================================================


class RegressionWrapper:
    """Wrapper class around sklearn's linear and ridge regressors"""

    def __init__(self, regressor="linear", **kwargs):
        if regressor == "linear":
            self.model = LinearRegression(**kwargs)
        elif regressor == "ridge":
            self.model = Ridge(**kwargs)
        else:
            raise ValueError(f"Regressor must be 'linear' or 'ridge', got '{regressor}'")

        self.regressor_type = regressor
        self.is_trained = False

    def predict(self, X):
        if not self.is_trained:
            warnings.warn("Applying an untrained regressor!", UserWarning)
        return self.model.predict(X=X)

    def fit(self, X, y, validate=True, **kwargs):
        self.model.fit(X=X, y=y, **kwargs)
        self.is_trained = True

        if validate:
            score = r2_score(y, self.predict(X))
            logger.info(f"r2 score(training): {score}")

        return self

    def get_state_dict(self):
        if not self.is_trained:
            raise ValueError("Regressor hasn't been trained, so no state_dict")

        state = {
            "coef_": self.model.coef_,
            "intercept_": self.model.intercept_,
            "n_features_in_": self.model.n_features_in_,
        }

        if hasattr(self.model, "rank_"):
            state["rank_"] = self.model.rank_
        if hasattr(self.model, "singular_"):
            state["singular_"] = self.model.singular_

        return state

    @classmethod
    def from_pretrained(cls, model_id):
        (
            scaler_type,
            plm_family,
            size_out,
            size_in,
            regressor_type,
            pca_type,
            n_pretrained_seqs,
        ) = parse_model_id(model_id)

        filename = f"{scaler_type}-scaler-{size_in}-{size_out}-{regressor_type}-{pca_type}-{n_pretrained_seqs}.npz"

        pretrained_file_path = hf_hub_download(
            repo_id="singhlab/plm_reverse_distillation", filename=f"weights/{filename}"
        )

        if not os.path.exists(pretrained_file_path):
            raise FileNotFoundError(
                f"Pretrained file path not found: {pretrained_file_path}"
            )

        loaded_data = np.load(pretrained_file_path, allow_pickle=True)
        state_dict = dict(loaded_data)

        instance = cls(regressor=regressor_type)

        def set_attr(model, attr_name, key_suffix):
            key = f"regressor__{key_suffix}"
            if key in state_dict:
                setattr(model, attr_name, state_dict[key])
            else:
                # Optional check for non-critical attributes, strictly logged
                pass

        set_attr(instance.model, "coef_", "coef_")
        set_attr(instance.model, "intercept_", "intercept_")
        set_attr(instance.model, "n_features_in_", "n_features_in_")
        set_attr(instance.model, "rank_", "rank_")
        set_attr(instance.model, "singular_", "singular_")

        instance.is_trained = True
        return instance


class fbpcaWrapper:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.is_trained = False

    def fit(self, res):
        self.mean_ = np.mean(res, axis=0)
        n_samples, _ = res.shape

        U, S, Vt = fbpca.pca(
            res - self.mean_,
            k=self.n_components,
            raw=True,
            n_iter=2,
            l=min(self.n_components + 10, n_samples),
        )

        self.components_ = Vt
        self.explained_variance_ = S**2 / (n_samples - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.var(res, axis=0).sum()
        )
        self.is_trained = True

        return self

    def get_state_dict(self):
        if not self.is_trained:
            raise ValueError("PCA hasn't been trained, no state_dict")

        return {
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "n_components": self.n_components,
        }

    def transform(self, res):
        transformed = np.dot(res - self.mean_, self.components_.T)
        if transformed.shape[1] < self.n_components:
            warnings.warn("Fewer components than requested", UserWarning)
        return transformed

    def fit_transform(self, res):
        return self.fit(res).transform(res)

    @classmethod
    def from_pretrained(cls, model_id):
        (
            scaler_type,
            plm_family,
            size_out,
            size_in,
            regressor_type,
            pca_type,
            n_pretrained_seqs,
        ) = parse_model_id(model_id)

        filename = f"{scaler_type}-scaler-{size_in}-{size_out}-{regressor_type}-{pca_type}-{n_pretrained_seqs}.npz"
        pretrained_file_path = hf_hub_download(
            repo_id="singhlab/plm_reverse_distillation", filename=f"weights/{filename}"
        )

        if not os.path.exists(pretrained_file_path):
            raise FileNotFoundError(
                f"Pretrained file path not found: {pretrained_file_path}"
            )

        loaded_data = np.load(pretrained_file_path, allow_pickle=True)
        state_dict = dict(loaded_data)

        comp_key = "pca__components_"
        if comp_key not in state_dict:
            raise KeyError(f"Could not find '{comp_key}' in {filename}")

        n_components = state_dict[comp_key].shape[0]
        instance = cls(n_components=n_components)

        def set_attr(obj, attr, key_suffix):
            key = f"pca__{key_suffix}"
            if key in state_dict:
                setattr(obj, attr, state_dict[key])
            else:
                logger.warning(f"Attribute {attr} (key: {key}) not found in state_dict.")

        set_attr(instance, "mean_", "mean_")
        set_attr(instance, "components_", "components_")
        set_attr(instance, "explained_variance_", "explained_variance_")
        set_attr(instance, "explained_variance_ratio_", "explained_variance_ratio_")

        instance.is_trained = True
        return instance


class IncrementalPCAWrapper:
    """
    Memory-efficient PCA using sklearn's IncrementalPCA.
    """

    def __init__(self, n_components, batch_size=131072):
        self.n_components = n_components
        self.batch_size = batch_size
        self.model = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.is_trained = False

    def fit(self, res):
        n_samples = res.shape[0]
        self.model = clone(self.model)

        if self.batch_size < self.n_components:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be >= n_components ({self.n_components})"
            )

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = res[start_idx:end_idx]
            self.model.partial_fit(batch)

        self.mean_ = self.model.mean_
        self.components_ = self.model.components_
        self.explained_variance_ = self.model.explained_variance_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.is_trained = True
        return self

    def transform(self, res):
        if not self.is_trained:
            warnings.warn("Applying untrained PCA!", UserWarning)
        return self.model.transform(res)

    def fit_transform(self, res):
        return self.fit(res).transform(res)

    def get_state_dict(self):
        if not self.is_trained:
            raise ValueError("PCA hasn't been trained, no state_dict")
        return {
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "n_components": self.n_components,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        batch_size=131072,
    ):
        (
            scaler_type,
            plm_family,
            size_out,
            size_in,
            regressor_type,
            pca_type,
            n_pretrained_seqs,
        ) = parse_model_id(model_id)

        filename = f"{scaler_type}-scaler-{size_in}-{size_out}-{regressor_type}-{pca_type}-{n_pretrained_seqs}.npz"
        pretrained_file_path = hf_hub_download(
            repo_id="singhlab/plm_reverse_distillation", filename=f"weights/{filename}"
        )

        if not os.path.exists(pretrained_file_path):
            raise FileNotFoundError(
                f"Pretrained file path not found: {pretrained_file_path}"
            )

        loaded_data = np.load(pretrained_file_path, allow_pickle=True)
        state_dict = dict(loaded_data)

        comp_key = "pca__components_"
        if comp_key not in state_dict:
            raise KeyError(f"Could not find '{comp_key}' in {filename}")

        n_components = state_dict[comp_key].shape[0]
        instance = cls(n_components=n_components, batch_size=batch_size)

        def set_attr(attr_name, key_suffix):
            key = f"pca__{key_suffix}"
            if key in state_dict:
                val = state_dict[key]
                setattr(instance, attr_name, val)
                setattr(instance.model, attr_name, val)
            else:
                logger.warning(
                    f"Attribute {attr_name} (key: {key}) not found in state_dict."
                )

        set_attr("mean_", "mean_")
        set_attr("components_", "components_")
        set_attr("explained_variance_", "explained_variance_")
        set_attr("explained_variance_ratio_", "explained_variance_ratio_")

        instance.is_trained = True
        return instance


class PCRegressionWrapper:
    """
    Principal Component Regression with automatic component selection
    using Johnstone's random matrix theory threshold.
    """

    def __init__(
        self, sigma_sq_: float = None, min_variance_explained: float = 0.95, **kwargs
    ):
        self.sigma_sq_ = sigma_sq_
        self.min_variance_explained = min_variance_explained
        self.regressor = LinearRegression(**kwargs)

        self.input_mean_ = None
        self.input_components_ = None
        self.input_eigenvalues_ = None
        self.n_significant_pcs_ = None
        self.johnstone_threshold_ = None
        self.is_trained = False

    def _estimate_sigma_sq_(
        self, eigenvalues: np.ndarray, n_samples: int, n_features: int
    ) -> float:
        lambda_ratio = n_features / n_samples
        initial_threshold = (1.0 + np.sqrt(lambda_ratio)) ** 2
        normalized_eigs = eigenvalues / np.mean(eigenvalues)
        noise_eigs = eigenvalues[normalized_eigs < initial_threshold]

        if len(noise_eigs) > 0:
            return np.median(noise_eigs)
        else:
            return np.median(eigenvalues[-max(1, n_features // 4) :])

    def fit(self, X: np.ndarray, y: np.ndarray, validate: bool = True, **kwargs):
        n_samples, n_features = X.shape
        self.input_mean_ = np.mean(X, axis=0)
        self.output_mean_ = np.mean(y, axis=0)

        X_centered = X - self.input_mean_
        y_centered = y - self.output_mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.input_components_ = Vt.T
        self.input_eigenvalues_ = (S**2) / (n_samples - 1)

        if self.sigma_sq_ is None:
            self.sigma_sq_ = self._estimate_sigma_sq_(
                self.input_eigenvalues_, n_samples, n_features
            )

        self.johnstone_threshold_ = johnstone_threshold(
            n_samples, n_features, self.sigma_sq_
        )
        significant_mask = self.input_eigenvalues_ > self.johnstone_threshold_
        self.n_significant_pcs_ = np.sum(significant_mask)

        if self.n_significant_pcs_ < 1:
            cumvar = np.cumsum(self.input_eigenvalues_) / np.sum(self.input_eigenvalues_)
            self.n_significant_pcs_ = (
                np.searchsorted(cumvar, self.min_variance_explained) + 1
            )
            logger.warning(
                f"No PCs exceeded Johnstone threshold. Using {self.n_significant_pcs_} PCs."
            )

        logger.info(
            f"PCRegression: {self.n_significant_pcs_}/{n_features} PCs selected (σ²={self.sigma_sq_:.4f})"
        )

        X_pc = X_centered @ self.input_components_[:, : self.n_significant_pcs_]
        self.regressor.fit(X_pc, y_centered, **kwargs)
        self.is_trained = True

        if validate:
            y_pred = self.predict(X)
            r2 = r2_score(y, y_pred)
            logger.info(f"PCRegression r2 score (training): {r2:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            warnings.warn("Applying untrained PC regressor!", UserWarning)

        X_centered = X - self.input_mean_
        X_pc = X_centered @ self.input_components_[:, : self.n_significant_pcs_]
        y_pred = self.regressor.predict(X_pc)
        return y_pred + self.output_mean_

    def get_state_dict(self):
        if not self.is_trained:
            raise ValueError("PC Regressor hasn't been trained, no state_dict")

        return {
            "coef_": self.regressor.coef_,
            "intercept_": self.regressor.intercept_,
            "input_mean_": self.input_mean_,
            "output_mean_": self.output_mean_,
            "input_components_": self.input_components_,
            "input_eigenvalues_": self.input_eigenvalues_,
            "n_significant_pcs_": self.n_significant_pcs_,
            "johnstone_threshold_": self.johnstone_threshold_,
            "sigma_sq_": self.sigma_sq_,
        }

    @classmethod
    def from_pretrained(cls, model_id):
        (
            scaler_type,
            plm_family,
            size_out,
            size_in,
            regressor_type,
            pca_type,
            n_pretrained_seqs,
        ) = parse_model_id(model_id)

        filename = f"{scaler_type}-scaler-{size_in}-{size_out}-{regressor_type}-{pca_type}-{n_pretrained_seqs}.npz"
        pretrained_file_path = hf_hub_download(
            repo_id="singhlab/plm_reverse_distillation", filename=f"weights/{filename}"
        )

        if not os.path.exists(pretrained_file_path):
            raise FileNotFoundError(
                f"Pretrained file path not found: {pretrained_file_path}"
            )

        loaded_data = np.load(pretrained_file_path, allow_pickle=True)
        state_dict = dict(loaded_data)

        instance = cls()

        def set_attr(obj, attr_name, key_suffix, required=True):
            key = f"regressor__{key_suffix}"
            if key in state_dict:
                setattr(obj, attr_name, state_dict[key])
            elif required:
                available = list(state_dict.keys())
                raise KeyError(
                    f"Missing required key '{key}' (checked '{key}'). Available: {available[:5]}..."
                )

        set_attr(instance, "input_mean_", "input_mean_")
        set_attr(instance, "output_mean_", "output_mean_")
        set_attr(instance, "input_components_", "input_components_")
        set_attr(instance, "input_eigenvalues_", "input_eigenvalues_")
        set_attr(instance, "n_significant_pcs_", "n_significant_pcs_")
        set_attr(instance, "johnstone_threshold_", "johnstone_threshold_")
        set_attr(instance, "sigma_sq_", "sigma_sq_", required=False)
        set_attr(instance.regressor, "coef_", "coef_")
        set_attr(instance.regressor, "intercept_", "intercept_")

        instance.is_trained = True
        return instance
