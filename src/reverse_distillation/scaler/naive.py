import os

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .. import logger
from .modules import IncrementalPCAWrapper, fbpcaWrapper
from .utils import ESM2_CONFIGS, parse_model_id

# =============================================================================
# Naive Scaler
# =============================================================================


class naiveScaler:
    def __init__(
        self,
        plm_size_in,
        plm_size_out,
        regressor_type="None",
        pca_type="incremental",
        **kwargs,
    ):
        self.n_features_in = ESM2_CONFIGS[plm_size_in]["embed_dim"]
        self.n_features_out = ESM2_CONFIGS[plm_size_out]["embed_dim"]

        self.pca_type = pca_type
        self.n_components = self.n_features_out

        if self.pca_type == "incremental":
            logger.info("Using IncrementalPCAWrapper (memory-efficient)")
            self.pca = IncrementalPCAWrapper(n_components=self.n_components, **kwargs)
        elif self.pca_type == "fbpca":
            logger.info("Using fbPCA implementation")
            self.pca = fbpcaWrapper(n_components=self.n_components, **kwargs)

        assert regressor_type == "None", (
            f"No regressor should be passed to the naive approch, but got: {regressor_type}"
        )
        self.is_trained = False

    def fit(self, xin, xout):
        xout_transformed = np.concatenate((xin, xout), axis=1)
        self.pca.fit(xout_transformed)

        self.is_trained = True

        return self

    def get_state_dict(self):
        if not self.is_trained:
            raise ValueError("Model not trained, no state_dict!")

        state_dict = {}
        state_dict.update(
            {"pca__" + key: value for key, value in self.pca.get_state_dict().items()}
        )

        return state_dict

    def step(self, xin, xout):
        xout_transformed = np.concatenate((xin, xout), axis=1)

        return self.pca.transform(xout_transformed)

    @classmethod
    def from_pretrained(cls, model_id):
        scaler_type, _, size_out, size_in, regressor_type, pca_type, n_pretrained_seqs = (
            parse_model_id(model_id)
        )

        if scaler_type != "naive":
            raise ValueError(
                f"Naive pca has to saved with `naive` prefix, insted got: {scaler_type}"
            )

        if regressor_type != "None":
            raise ValueError(
                f"Naive pca has to saved with `None` as regressor type, insted got: {scaler_type}"
            )

        instance = cls(plm_size_in=size_in, plm_size_out=size_out, pca_type=pca_type)

        logger.info(f"Loading {pca_type} PCA from {model_id}...")

        if pca_type == "incremental":
            instance.pca = IncrementalPCAWrapper.from_pretrained(model_id)
        elif pca_type == "fbpca":
            instance.pca = fbpcaWrapper.from_pretrained(model_id)
        else:
            raise ValueError(f"Unknown pca_type parsed from ID: {pca_type}")

        instance.is_trained = True
        instance.is_trained = True

        return instance


class naiveScalerGPUOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        # pca
        self.register_buffer("mean_", None)
        self.register_buffer("components_", None)

        self.is_trained = False

    @torch.inference_mode()
    def step(self, xin, xout):
        xin_transformed = torch.cat((xin, xout), dim=1)

        return (xin_transformed - self.mean_) @ self.components_.T

    @classmethod
    def from_pretrained(cls, model_id):
        scaler_type, _, size_out, size_in, regressor_type, pca_type, n_pretrained_seqs = (
            parse_model_id(model_id)
        )

        if scaler_type != "naive":
            raise ValueError(
                f"Naive pca has to saved with `naive` prefix, insted got: {scaler_type}"
            )

        if regressor_type != "None":
            raise ValueError(
                f"Naive pca has to saved with `None` as regressor type, insted got: {scaler_type}"
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

        # Helper to convert numpy -> torch float32 tensor
        def to_tensor(key):
            if key not in state_dict:
                raise KeyError(f"Key {key} missing in {filename}")
            val = state_dict[key]
            # Ensure at least 1D for intercept/scalars to match PyTorch broadcasting if needed
            if np.isscalar(val):
                val = np.array([val])
            return torch.from_numpy(val).float()

        # Load PCA Buffers
        instance.register_buffer("mean_", to_tensor("pca__mean_"))
        instance.register_buffer("components_", to_tensor("pca__components_"))

        instance.is_trained = True
        instance.eval()

        return instance
