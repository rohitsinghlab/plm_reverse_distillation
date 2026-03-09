import argparse
import gc
import os

import numpy as np
import torch

from reverse_distillation import logger
from reverse_distillation.scaler.naive import naiveScaler
from reverse_distillation.scaler.rd import rdScaler
from reverse_distillation.scaler.utils import ESM2_CONFIGS, read_fasta_to_dict


def save_scaler(
    state_dict,
    scaler_type,
    plm_family,
    plm_size_in,
    plm_size_out,
    regressor_type,
    pca_type,
    n_pretrained_seqs,
    save_scalar_path,
):
    "{scaler_type}.{plm_family}/{size_out}:{regressor_type}-{pca_type}-{n_pretrained_seqs}"

    logger.info(f"Saving Scaler model: {plm_size_in} -> {plm_size_out} ...")
    base_dir = save_scalar_path
    os.makedirs(base_dir, exist_ok=True)

    pretrained_file = f"{scaler_type}-scaler-{plm_size_in}-{plm_size_out}-{regressor_type}-{pca_type}-{n_pretrained_seqs}.npz"

    filepath = os.path.join(base_dir, pretrained_file)
    np.savez_compressed(filepath, **state_dict)


def generate_embeddings(proteins_seqs, model, repr_layer, alphabet, batch_size):
    device = "cuda"
    model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    all_residue_embeddings = []
    with torch.no_grad():
        for i in range(0, len(proteins_seqs), batch_size):
            batch_data = proteins_seqs[i : i + batch_size]

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)  # batch_size x max_seq_len

            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_embeddings = results["representations"][repr_layer]

            lengths = torch.tensor(
                [len(protein) for _, protein in batch_data], device=device
            ).unsqueeze(1)
            max_seq_len = batch_tokens.size()[1]

            range_tensor = torch.arange(max_seq_len, device=device)

            selection_mask = (range_tensor >= 1) & (range_tensor <= lengths)

            valid_embeddings = token_embeddings[
                selection_mask
            ]  # n_amino_acids x embeding_size

            all_residue_embeddings.append(valid_embeddings.detach().cpu().numpy())
            del batch_tokens
            del results
            del token_embeddings

    model.cpu()
    torch.cuda.empty_cache()
    all_residue_embeddings = np.concatenate(all_residue_embeddings, axis=0)
    print(f"Processing complete. Final Tensor Shape: {all_residue_embeddings.shape}")

    return all_residue_embeddings


def load_esm2_models():
    torch.serialization.add_safe_globals([argparse.Namespace])
    torch.set_float32_matmul_precision("high")

    ordered_sizes = ["8M", "35M", "150M", "650M", "3B", "15B"]

    model_dict = {}
    for plm_size in ordered_sizes:
        esm2, alphabet = torch.hub.load(
            "facebookresearch/esm:main", f"{ESM2_CONFIGS[plm_size]['name']}"
        )

        esm2.eval()
        esm2.requires_grad_(False)

        model_dict.update({plm_size: (esm2, alphabet)})

    return model_dict


def train(
    model_dict,
    fasta_file,
    regressor_type,
    pca_type,
    n_pretrained_seqs,
    scaler,
    scaler_type,
    save_scalar_path,
):
    ordered_sizes = ["8M", "35M", "150M", "650M", "3B", "15B"]
    batch_sizes = {"8M": 256, "35M": 256, "150M": 128, "650M": 64, "3B": 32, "15B": 16}
    seqs_to_int = {"0.5k": 500, "1k": 1000, "5k": 5000, "10k": 10000}

    proteins_seqs = read_fasta_to_dict(fasta_file)

    proteins_seqs = [
        (label, seq.replace("\n", "").replace(" ", "").upper())
        for label, seq in proteins_seqs
    ][: seqs_to_int[n_pretrained_seqs]]

    plm_size_in = ordered_sizes[0]
    esm2_in, alphabet_in = model_dict[plm_size_in]
    sin = generate_embeddings(
        proteins_seqs,
        esm2_in,
        ESM2_CONFIGS[plm_size_in]["layers"],
        alphabet_in,
        batch_sizes[plm_size_in],
    )

    for plm_size_out in ordered_sizes[1:]:
        esm2_out, alphabet_out = model_dict[plm_size_out]
        sout = generate_embeddings(
            proteins_seqs,
            esm2_out,
            ESM2_CONFIGS[plm_size_out]["layers"],
            alphabet_out,
            batch_sizes[plm_size_out],
        )

        assert sin.shape[0] == sout.shape[0]

        # Train RD scaler
        logger.info(f"Training {scaler_type}: {plm_size_in} -> {plm_size_out}")
        rd_scaler = scaler(
            plm_size_in=plm_size_in,
            plm_size_out=plm_size_out,
            plm_family="esm2",
            regressor_type=regressor_type,
            pca_type=pca_type,
        )
        rd_scaler.fit(xin=sin, xout=sout)

        # Save RD scaler with correct naming convention
        state_dict = rd_scaler.get_state_dict()
        save_scaler(
            state_dict=state_dict,
            scaler_type=scaler_type,
            plm_family="esm2",
            plm_size_in=plm_size_in,
            plm_size_out=plm_size_out,
            regressor_type=regressor_type,
            pca_type=pca_type,
            n_pretrained_seqs=n_pretrained_seqs,
            save_scalar_path=save_scalar_path,
        )

        sin = sout
        plm_size_in = plm_size_out

        del sout

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Cleaned up. Moving to next pair...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mutation experiment")
    np.random.seed(42)

    parser.add_argument(
        "--dataset_path",
        help="Path to fasta dataset file",
        type=str,
    )

    parser.add_argument("--scalar_path", 
                        help="Path to where the scaler will be saved",
                        type=str)

    parser.add_argument(
        "--regressor_type",
        type=str,
        default="linear",
        help="Type of regressor",
        choices=["linear", "ridge", "pcr", "None"]  
    )

    parser.add_argument(
        "--scaler_type",
        help="Type of scaler",
        type=str,
        default="rd",
        choices=["rd", "naive"]  # rd, naive
    )

    parser.add_argument(
        "--pca_type",
        type=str,
        help="Type of PCA used",
        default="incremental",
        choices=["incremental", "fbpca"]  # incremental, fbpca
    )

    parser.add_argument(
        "--n_pretrained_seqs",
        type=str,
        default="1k",
        choices=["0.5k", "1k", "5k", "10k"]  # 0.5k, 1k, 5k 10k
    )

    args = parser.parse_args()
    model_dict = load_esm2_models()

    if args.scaler_type == "rd":
        logger.info("Reversed distilled scaler")
        scaler = rdScaler

    elif args.scaler_type == "naive":
        logger.info("Naive scaler")
        scaler = naiveScaler

    train(
        model_dict=model_dict,
        fasta_file=args.dataset_path,
        regressor_type=args.regressor_type,
        pca_type=args.pca_type,
        n_pretrained_seqs=args.n_pretrained_seqs,
        scaler=scaler,
        scaler_type=args.scaler_type,
        save_scalar_path=args.scalar_path,
    )
