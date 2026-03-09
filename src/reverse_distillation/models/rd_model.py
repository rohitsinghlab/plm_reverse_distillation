import argparse

import torch
import torch.nn as nn
from esm.pretrained import load_model_and_alphabet

from ..scaler.naive import naiveScaler, naiveScalerGPUOptimized
from ..scaler.rd import rdScaler, rdScalerGPUOptimized
from ..scaler.utils import ESM2_CONFIGS

"""
Naming convesion for the pretrained reversed distilled esm model:

    model_id = {scaler_type}.esm2/{size}

    - scaler_type:
        * rd: trained as reverse distillation process (see paper)
        * naive: embeddings are concatenated and then pca'ed to
                 the respective size (naive approch)

    - size: refers to the number of parameters of the esm2 family
            it will load all esm2 models in order from 8M to {size}
            where size could be any: 35M, 150M, 650M, 3B or 15B

    e.g. : rd.esm2/15M, naive.esm2/15B etc ...

    Also we provide different configurations for the rd scalers:

    scaler_options = {regressor_type}-{pca_type}-{n_pretrained_seqs}

    - regressor_type:
        * ridge
        * linear
        * pcr

    - pca_type:
        * fbpca (normal pca, see fbpca implementation)
        * incremental (incremental pca, see sklearn implementation)

    - n_pretrained_seqs: number of seqs the scaler was pretrained on,
                         could be any of the: 0.5k, 1k, 5k, 10k

"""


class RDESM(nn.Module):
    def __init__(
        self,
        sizes: list[str],
        esm_base_models: nn.ModuleDict = None,
        scalers: nn.ModuleDict | dict = None,
    ):
        super().__init__()
        self.sizes = sizes
        self.special_tokens = []

        self.esm_models = esm_base_models
        self.scalers = scalers

        if esm_base_models is None:
            raise ValueError(
                "esm_base_models has to be an nn.Module dict with the ESM2 pretrained models!"
            )

        if scalers is None:
            raise ValueError(
                "scalers has to be an nn.Module dict with the rdScalers pretrained models!"
            )

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "RDESM":
        ordered_sizes = ["8M", "35M", "150M", "650M", "3B", "15B"]

        # options for the scaler backend, i.e type of regressor, pca, etc.
        scaler_option = {
            "use_gpu_optimized_scaler": True,
            "scaler_type": "rd",
            "regressor_type": "pcr",  # pcr, linear, ridge
            "pca_type": "fbpca",  # default, incremental
            "n_pretrained_seqs": "1k",
        }

        for option in scaler_option:
            if option in kwargs:
                scaler_option[option] = kwargs[option]

        if scaler_option["scaler_type"] == "naive":
            scaler_option["regressor_type"] = "None"

        # model_id : "rd.esm2/SIZE"
        model_size = model_id.split("/")[1]
        size_idx = ordered_sizes.index(model_size)

        config = {
            **ESM2_CONFIGS[model_size],
            "sizes": ordered_sizes[: (size_idx + 1)],
            **scaler_option,
        }

        torch.serialization.add_safe_globals([argparse.Namespace])
        torch.set_float32_matmul_precision("high")

        esm_models = nn.ModuleDict()
        if config["use_gpu_optimized_scaler"]:
            scalers = nn.ModuleDict()
        else:
            scalers = {}

        scaler_config_str = f"{config['regressor_type']}-{config['pca_type']}-{config['n_pretrained_seqs']}"
        alphabet = None

        for i, size in enumerate(config["sizes"]):
            esm_model, alphabet = load_model_and_alphabet(f"{ESM2_CONFIGS[size]['name']}")

            if i > 0:
                scaler_name = f"{config['scaler_type']}.esm2/{size}:{scaler_config_str}"

                if config["use_gpu_optimized_scaler"]:
                    if config["scaler_type"] == "rd":
                        scaler = rdScalerGPUOptimized.from_pretrained(scaler_name)
                    elif config["scaler_type"] == "naive":
                        scaler = naiveScalerGPUOptimized.from_pretrained(scaler_name)
                    scalers.update({size: torch.compile(scaler, dynamic=True)})
                else:
                    if config["scaler_type"] == "rd":
                        scaler = rdScaler.from_pretrained(scaler_name)
                    elif config["scaler_type"] == "naive":
                        scaler = naiveScaler.from_pretrained(scaler_name)

                    scalers.update({size: scaler})

            for param in esm_model.parameters():
                param.requires_grad = False
            esm_model.eval()

            esm_models.update({size: torch.compile(esm_model, dynamic=True)})

        instance = cls(config["sizes"], esm_base_models=esm_models, scalers=scalers)

        instance.special_tokens = [
            alphabet.padding_idx,
            alphabet.mask_idx,
            alphabet.cls_idx,
            alphabet.eos_idx,
        ]
        return instance

    def forward(
        self,
        batch_tokens: torch.Tensor,
        **kwargs,
    ):
        size_in = "8M"
        repr_layers = ESM2_CONFIGS[size_in]["layers"]

        special_tokens_tensor = torch.tensor(
            self.special_tokens, device=batch_tokens.device
        )
        selection_mask = ~torch.isin(batch_tokens, special_tokens_tensor)

        sin = self.esm_step(batch_tokens, repr_layers, size_in, **kwargs)[selection_mask]

        rd_per_level = {}
        esm_per_level = {}

        for size_out in self.sizes[1:]:
            repr_layers = ESM2_CONFIGS[size_out]["layers"]
            sout_full = self.esm_step(
                batch_tokens, repr_layers, size_out, **kwargs
            )  # batch_size x max_seq_len x embed_size
            sout = sout_full[selection_mask]  # (batch_size * non_pad_aa) x embed_size

            scaler = self.scalers[size_out]
            sout_prime = scaler.step(sin, sout)

            reconstructed_sout_prime = sout_full
            reconstructed_sout_prime[selection_mask] = (
                sout_prime  # batch_size x max_seq_len x embed_size
            )

            rd_per_level[size_out] = reconstructed_sout_prime.detach().cpu()
            esm_per_level[size_out] = sout.detach().cpu()

            size_in = size_out
            sin = sout_prime.to(batch_tokens.device)

        results = {
            "logits": None,
            "representations": rd_per_level,
            "esm_representations": esm_per_level,
        }

        return results

    @torch.no_grad()
    def esm_step(self, batch_tokens, repr_layers, size, **kwargs):
        esm2 = self.esm_models[size]

        results = esm2(batch_tokens, repr_layers=[repr_layers], **kwargs)
        token_representations = results["representations"][repr_layers]

        return token_representations  # batch_size x max_seq_len x embed_size
