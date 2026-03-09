import numpy as np

ESM2_CONFIGS = {
    "8M": {"name": "esm2_t6_8M_UR50D", "layers": 6, "embed_dim": 320},
    "35M": {"name": "esm2_t12_35M_UR50D", "layers": 12, "embed_dim": 480},
    "150M": {"name": "esm2_t30_150M_UR50D", "layers": 30, "embed_dim": 640},
    "650M": {"name": "esm2_t33_650M_UR50D", "layers": 33, "embed_dim": 1280},
    "3B": {"name": "esm2_t36_3B_UR50D", "layers": 36, "embed_dim": 2560},
    "15B": {"name": "esm2_t48_15B_UR50D", "layers": 48, "embed_dim": 5120},
}


def read_fasta_to_dict(fasta_file):
    fasta_dict = []
    with open(fasta_file) as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id:
                    fasta_dict.append((sequence_id, "".join(sequence)))

                sequence_id = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)

        if sequence_id:
            fasta_dict.append((sequence_id, "".join(sequence)))

    return fasta_dict


def parse_model_id(model_id):
    ordered_sizes = ["8M", "35M", "150M", "650M", "3B", "15B"]
    # Format: {scaler_type}.{plm_family}/{size_out}:{regressor_type}-{pca_type}-{n_pretrained_seqs}
    try:
        scaler_type = model_id.split(".")[0]
        plm_family = model_id.split(".")[1].split("/")[0]

        size_out = model_id.split("/")[1].split(":")[0]
        size_in = ordered_sizes[ordered_sizes.index(size_out) - 1]

        options = model_id.split(":")[1]
        regressor_type = options.split("-")[0]
        pca_type = options.split("-")[1]
        n_pretrained_seqs = options.split("-")[2]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid model_id format: {model_id}") from e

    return (
        scaler_type,
        plm_family,
        size_out,
        size_in,
        regressor_type,
        pca_type,
        n_pretrained_seqs,
    )


def johnstone_threshold(n_samples: int, n_features: int, sigma_sq: float = 1.0) -> float:
    lambda_ratio = n_features / n_samples
    return sigma_sq * (1.0 + np.sqrt(lambda_ratio)) ** 2
