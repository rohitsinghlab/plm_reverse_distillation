import os

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .. import logger
from .modules import (
    IncrementalPCAWrapper,
    PCRegressionWrapper,
    RegressionWrapper,
    fbpcaWrapper,
)
from .utils import ESM2_CONFIGS, parse_model_id

# =============================================================================
# Reverse Distillation Scalers
# =============================================================================


class rdScaler:
    def __init__(
        self,
        plm_size_in,
        plm_size_out,
        regressor_type="linear",
        pca_type="fbpca",
        **kwargs,
    ):
        """
        Initialize the reverse distillation scaler.

        Args:
            plm_size_in: Size of input PLM (e.g., "8M", "35M")
            plm_size_out: Size of output PLM (e.g., "35M", "150M")
            regressor_type: Type of regressor ("linear", "ridge" for sklearn regressor, "pcr" for PC regression)
            pca_type: Type of pca, fbpca or incremental
            **kwargs: Additional arguments passed to regressor or pca
        """

        self.n_features_in = ESM2_CONFIGS[plm_size_in]["embed_dim"]
        self.n_features_out = ESM2_CONFIGS[plm_size_out]["embed_dim"]
        self.regressor_type = regressor_type
        self.pca_type = pca_type

        self.plm_size_in = plm_size_in
        self.plm_size_out = plm_size_out

        if regressor_type == "pcr":
            logger.info("Using PCRegressionWrapper with Johnstone threshold")
            self.regressor = PCRegressionWrapper(**kwargs)
        elif regressor_type == "linear" or regressor_type == "ridge":
            logger.info(f"Using {regressor_type} regressor")
            self.regressor = RegressionWrapper(regressor_type, **kwargs)
        else:
            raise ValueError(f"Got unexpected regressor type: {regressor_type}")

        # Choose PCA implementation
        n_pca_components = self.n_features_out - self.n_features_in
        if self.pca_type == "incremental":
            logger.info("Using IncrementalPCAWrapper (memory-efficient)")
            self.pca = IncrementalPCAWrapper(n_components=n_pca_components)
        elif self.pca_type == "fbpca":
            logger.info("Using fbPCA implementation")
            self.pca = fbpcaWrapper(n_components=n_pca_components)

        self.is_trained = False

    def fit_regressor(self, X, y, show_r2=True, **kwargs):
        self.regressor.fit(X=X, y=y, validate=show_r2, **kwargs)

    def fit_pca(self, X):
        self.pca.fit(X)

    def transform_pca(self, residuals):
        return self.pca.transform(residuals)

    def predict_regressor(self, xin):
        return self.regressor.predict(xin)

    def fit(self, xin, xout, show_r2=True, **kwargs):
        logger.info(f"Fitting rdScaler {self.plm_size_in} -> {self.plm_size_out} ...")
        self.fit_regressor(X=xin, y=xout, show_r2=show_r2, **kwargs)

        xin_transformed = self.predict_regressor(xin)
        residuals = xout - xin_transformed
        self.fit_pca(residuals)

        self.is_trained = True

    def step(
        self, xin: torch.Tensor, xout: torch.Tensor, batch_size=131072
    ) -> torch.Tensor:
        """
        Scale embeddings (xin) to match dimensions of xout using the trained scaler.
        """
        if (
            xin.size()[0] != xout.size()[0]
            or len(xin.size()) != 2
            or len(xout.size()) != 2
        ):
            raise ValueError(
                f"xin and xout must have same length. Got {len(xin)} and {len(xout)}"
            )

        assert xin.size()[1] + self.pca.n_components == xout.size()[1], (
            "input dimension + the pca components don't match with the embedding size of the output"
        )
        # Convert to numpy
        xin = xin.cpu().numpy()
        xout = xout.cpu().numpy()

        seq_len, _ = xin.shape

        scaled_embeddings = np.zeros_like(xout)
        # Process in batches
        for start_idx in range(0, seq_len, batch_size):
            end_idx = min(start_idx + batch_size, seq_len)
            batch_xin = xin[start_idx:end_idx]
            batch_xout = xout[start_idx:end_idx]

            # Predict with regressor
            xin_transformed = self.predict_regressor(batch_xin)

            # Calculate residuals
            residuals = batch_xout - xin_transformed

            # Apply PCA to residuals
            residuals_compressed = self.transform_pca(residuals)

            # Concatenate to get scaled embedding
            batch_scaled = np.concatenate((batch_xin, residuals_compressed), axis=-1)
            scaled_embeddings[start_idx:end_idx, :] = batch_scaled

        return torch.from_numpy(scaled_embeddings)

    def get_state_dict(self):
        if not self.is_trained:
            raise ValueError("Model not trained, no state_dict!")

        regressor_state_dict = self.regressor.get_state_dict()
        pca_state_dict = self.pca.get_state_dict()

        state_dict = {}

        state_dict.update(
            {"regressor__" + key: value for key, value in regressor_state_dict.items()}
        )
        state_dict.update({"pca__" + key: value for key, value in pca_state_dict.items()})

        return state_dict

    @classmethod
    def from_pretrained(cls, model_id):
        scaler_type, _, size_out, size_in, regressor_type, pca_type, _ = parse_model_id(
            model_id
        )

        if scaler_type != "rd":
            raise ValueError(
                f"Rd scaler has to saved with `rd` prefix, insted got: {scaler_type}"
            )

        instance = cls(
            plm_size_in=size_in,
            plm_size_out=size_out,
            regressor_type=regressor_type,
            pca_type=pca_type,
        )

        logger.info(f"Loading {regressor_type} regressor from {model_id}...")
        if regressor_type == "pcr":
            instance.regressor = PCRegressionWrapper.from_pretrained(model_id)
        elif regressor_type == "linear" or regressor_type == "ridge":
            instance.regressor = RegressionWrapper.from_pretrained(model_id)
        else:
            raise ValueError(f"Unknown regressor parsed from ID: {regressor_type}")

        logger.info(f"Loading {pca_type} PCA from {model_id}...")

        if pca_type == "incremental":
            instance.pca = IncrementalPCAWrapper.from_pretrained(model_id)
        elif pca_type == "fbpca":
            instance.pca = fbpcaWrapper.from_pretrained(model_id)
        else:
            raise ValueError(f"Unknown pca_type parsed from ID: {pca_type}")

        instance.is_trained = True

        return instance


class rdScalerGPUOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        # ridge/linear regressor
        self.register_buffer("coef_", None)
        self.register_buffer("intercept_", None)

        # pcr regressor
        self.register_buffer("input_mean_", None)
        self.register_buffer("output_mean_", None)
        self.register_buffer("input_components_", None)
        self.n_significant_pcs_ = None

        # pca
        self.register_buffer("mean_", None)
        self.register_buffer("components_", None)

        self.is_trained = False

        self.regressor_step = None  # predict_pcr or predict_regressor

    @torch.inference_mode()
    def predict_pcr(self, X):
        X_centered = X - self.input_mean_

        X_pc = X_centered @ self.input_components_[:, : self.n_significant_pcs_]

        y_centered_pred = self.predict_regressor(X_pc)

        return y_centered_pred + self.output_mean_

    @torch.inference_mode()
    def predict_regressor(self, X):
        return torch.nn.functional.linear(X, self.coef_, self.intercept_)

    @torch.inference_mode()
    def transform_pca(self, res):
        return (res - self.mean_) @ self.components_.T

    @torch.inference_mode()
    def step(self, xin, xout):
        xin_transformed = self.regressor_step(xin)

        if xout.ndim < xin_transformed.ndim:
            xout = xout.unsqueeze(-1)

        residuals = xout - xin_transformed

        residuals_compressed = self.transform_pca(residuals)

        xin_scaled = torch.cat((xin, residuals_compressed), dim=-1)

        return xin_scaled

    @classmethod
    def from_pretrained(cls, model_id):
        scaler_type, _, size_out, size_in, regressor_type, pca_type, n_pretrained_seqs = (
            parse_model_id(model_id)
        )

        filename = f"{scaler_type}-scaler-{size_in}-{size_out}-{regressor_type}-{pca_type}-{n_pretrained_seqs}.npz"
        pretrained_file_path = hf_hub_download(
            repo_id="singhlab/plm_reverse_distillation", filename=f"weights/{filename}"
        )

        if not os.path.exists(pretrained_file_path):
            raise FileNotFoundError(
                f"Pretrained file path not found: {pretrained_file_path}"
            )

        # Load Data
        loaded_data = np.load(pretrained_file_path, allow_pickle=True)
        state_dict = dict(loaded_data)

        instance = cls()
        instance.regressor_type = regressor_type

        # Helper to convert numpy -> torch float32 tensor
        def to_tensor(key):
            if key not in state_dict:
                raise KeyError(f"Key {key} missing in {filename}")
            val = state_dict[key]
            # Ensure at least 1D for intercept/scalars to match PyTorch broadcasting if needed
            if np.isscalar(val):
                val = np.array([val])
            return torch.from_numpy(val).float()

        # Load Regressor Buffers
        if regressor_type == "pcr":
            instance.regressor_step = instance.predict_pcr

            # Load PCR-specific stats
            instance.register_buffer("input_mean_", to_tensor("regressor__input_mean_"))
            instance.register_buffer("output_mean_", to_tensor("regressor__output_mean_"))
            instance.register_buffer(
                "input_components_", to_tensor("regressor__input_components_")
            )

            # Load scalar int (not a buffer)
            instance.n_significant_pcs_ = int(state_dict["regressor__n_significant_pcs_"])
        else:
            instance.regressor_step = instance.predict_regressor

        # Load Coefficients (Common to Linear, Ridge, and PCR)
        coef = to_tensor("regressor__coef_")
        if coef.ndim == 1:
            coef = coef.unsqueeze(0)

        instance.register_buffer("coef_", coef)
        instance.register_buffer("intercept_", to_tensor("regressor__intercept_"))

        # Load PCA Buffers (for Residuals)
        instance.register_buffer("mean_", to_tensor("pca__mean_"))
        instance.register_buffer("components_", to_tensor("pca__components_"))

        instance.is_trained = True
        instance.eval()

        return instance
