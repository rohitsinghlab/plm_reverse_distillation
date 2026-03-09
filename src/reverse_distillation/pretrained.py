import esm

from .models.rd_model import (
    RDESM,
)

# Registry for pretrained models
_MODEL_REGISTRY = {}


def register_model(model_name):
    """Decorator to register a pretrained model function."""

    def decorator(func):
        _MODEL_REGISTRY[model_name] = func
        return func

    return decorator


def load_model_and_alphabet(model_name, **kwargs):
    """Load a pretrained model and alphabet by name.

    Args:
        model_name: Name of the model (e.g., 'esm2_rd_35M', 'esm2_rd_150M')
        **kwargs: Additional arguments to pass to the model loading function

    Returns:
        model, alphabet: The loaded model and tokenizer alphabet
    """
    if model_name not in _MODEL_REGISTRY:
        available_models = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        )
    return _MODEL_REGISTRY[model_name](**kwargs)


def get_tokenizer_esm2():
    return esm.data.Alphabet.from_architecture("ESM-1b")


@register_model("esm2.rd/35M")
def esm2_rd_35M(
    use_gpu_optimized_scaler=True,
    regressor_type="pcr",
    pca_type="incremental",
    n_pretrained_seqs="1k",
):
    model = RDESM.from_pretrained(
        "rd.esm2/35M",
        use_gpu_optimized_scaler=use_gpu_optimized_scaler,
        regressor_type=regressor_type,
        pca_type=pca_type,
        n_pretrained_seqs=n_pretrained_seqs,
    )
    alphabet = get_tokenizer_esm2()
    return model, alphabet


@register_model("esm2.rd/150M")
def esm2_rd_150M(
    use_gpu_optimized_scaler=True,
    regressor_type="pcr",
    pca_type="incremental",
    n_pretrained_seqs="1k",
):
    model = RDESM.from_pretrained(
        "rd.esm2/150M",
        use_gpu_optimized_scaler=use_gpu_optimized_scaler,
        regressor_type=regressor_type,
        pca_type=pca_type,
        n_pretrained_seqs=n_pretrained_seqs,
    )
    alphabet = get_tokenizer_esm2()
    return model, alphabet


@register_model("esm2.rd/650M")
def esm2_rd_650M(
    use_gpu_optimized_scaler=True,
    regressor_type="pcr",
    pca_type="incremental",
    n_pretrained_seqs="1k",
):
    model = RDESM.from_pretrained(
        "rd.esm2/650M",
        use_gpu_optimized_scaler=use_gpu_optimized_scaler,
        regressor_type=regressor_type,
        pca_type=pca_type,
        n_pretrained_seqs=n_pretrained_seqs,
    )
    alphabet = get_tokenizer_esm2()
    return model, alphabet


@register_model("esm2.rd/3B")
def esm2_rd_3B(
    use_gpu_optimized_scaler=True,
    regressor_type="pcr",
    pca_type="incremental",
    n_pretrained_seqs="1k",
):
    model = RDESM.from_pretrained(
        "rd.esm2/3B",
        use_gpu_optimized_scaler=use_gpu_optimized_scaler,
        regressor_type=regressor_type,
        pca_type=pca_type,
        n_pretrained_seqs=n_pretrained_seqs,
    )
    alphabet = get_tokenizer_esm2()
    return model, alphabet


@register_model("esm2.rd/15B")
def esm2_rd_15B(
    use_gpu_optimized_scaler=True,
    regressor_type="pcr",
    pca_type="incremental",
    n_pretrained_seqs="1k",
):
    model = RDESM.from_pretrained(
        "rd.esm2/15B",
        use_gpu_optimized_scaler=use_gpu_optimized_scaler,
        regressor_type=regressor_type,
        pca_type=pca_type,
        n_pretrained_seqs=n_pretrained_seqs,
    )
    alphabet = get_tokenizer_esm2()
    return model, alphabet


# @register_model("esm2.naive/15B")
# def esm2_naive_15B(
#     use_gpu_optimized_scaler=True,
#     regressor_type="None",
#     pca_type="incremental",
#     n_pretrained_seqs="1k",
# ):
#     model = RDESM.from_pretrained(
#         "naive.esm2/15B",
#         use_gpu_optimized_scaler=use_gpu_optimized_scaler,
#         scaler_type="naive",
#         regressor_type=regressor_type,
#         pca_type=pca_type,
#         n_pretrained_seqs=n_pretrained_seqs,
#     )
#     alphabet = get_tokenizer_esm2()
#     return model, alphabet
