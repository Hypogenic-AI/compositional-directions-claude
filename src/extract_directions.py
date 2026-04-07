"""
Extract concept directions from Gemma-2B's unembedding matrix using BATS 3.0 counterfactual pairs.
Implements the mean-difference method from Park et al. (2024).
"""
import os
import glob
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "EleutherAI/pythia-2.8b"
BATS_DIR = "datasets/bats3.0/BATS_3.0"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model():
    """Load model and extract the unembedding matrix."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    model.eval()

    # Get unembedding matrix - Pythia uses embed_out
    if hasattr(model, "embed_out"):
        W_U = model.embed_out.weight.detach().float()
    elif hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float()
    elif hasattr(model.gpt_neox, "embed_out"):
        # Some Pythia versions
        W_U = model.gpt_neox.embed_out.weight.detach().float()
    else:
        # Try to find the output embedding
        for name, param in model.named_parameters():
            if "embed_out" in name or "lm_head" in name:
                W_U = param.detach().float()
                break
        else:
            raise RuntimeError("Could not find unembedding matrix")

    print(f"Unembedding matrix shape: {W_U.shape}")
    return tokenizer, model, W_U


def load_bats_category(filepath, tokenizer):
    """Load a BATS category file and filter to single-token pairs."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        w1, w2 = parts
        # Handle BATS format where target may have multiple options separated by /
        w2 = w2.split("/")[0]

        # Check single-token encoding
        # Pythia/GPT-NeoX uses a BPE tokenizer with space prefix
        tok1 = tokenizer.encode(f" {w1}", add_special_tokens=False)
        tok2 = tokenizer.encode(f" {w2}", add_special_tokens=False)
        if len(tok1) == 1 and len(tok2) == 1 and tok1[0] != tok2[0]:
            pairs.append((w1, w2, tok1[0], tok2[0]))
    return pairs


def load_all_bats_categories(tokenizer):
    """Load all BATS 3.0 categories, returning those with sufficient single-token pairs."""
    categories = {}
    category_types = {}

    type_map = {
        "1_Inflectional_morphology": "morphological",
        "2_Derivational_morphology": "morphological",
        "3_Encyclopedic_semantics": "encyclopedic",
        "4_Lexicographic_semantics": "lexicographic",
    }

    for type_dir, type_name in type_map.items():
        dir_path = os.path.join(BATS_DIR, type_dir)
        if not os.path.isdir(dir_path):
            continue
        for fpath in sorted(glob.glob(os.path.join(dir_path, "*.txt"))):
            fname = os.path.basename(fpath)
            # Extract category name from filename like "E01 [country - capital].txt"
            cat_name = fname.replace(".txt", "").strip()
            pairs = load_bats_category(fpath, tokenizer)
            if len(pairs) >= 5:  # need at least 5 pairs
                categories[cat_name] = pairs
                category_types[cat_name] = type_name
                print(f"  {cat_name}: {len(pairs)} single-token pairs ({type_name})")

    return categories, category_types


def extract_concept_direction(pairs, W_U):
    """
    Extract concept direction using mean-difference in unembedding space.
    direction = mean(W_U[target] - W_U[base]) normalized to unit length.
    """
    diffs = []
    for _, _, tok_base, tok_target in pairs:
        diff = W_U[tok_target] - W_U[tok_base]
        diffs.append(diff)
    diffs = torch.stack(diffs)  # (n_pairs, d_model)
    mean_diff = diffs.mean(dim=0)
    direction = mean_diff / mean_diff.norm()
    return direction, diffs


def leave_one_out_consistency(pairs, W_U):
    """
    Compute LOO consistency: for each pair, extract direction from remaining pairs
    and measure cosine similarity with the left-out pair's difference vector.
    Returns mean and std of cosine similarities.
    """
    diffs = []
    for _, _, tok_base, tok_target in pairs:
        diff = W_U[tok_target] - W_U[tok_base]
        diffs.append(diff)
    diffs = torch.stack(diffs)

    cosines = []
    for i in range(len(diffs)):
        mask = torch.ones(len(diffs), dtype=torch.bool)
        mask[i] = False
        loo_mean = diffs[mask].mean(dim=0)
        loo_dir = loo_mean / loo_mean.norm()
        cos = torch.dot(loo_dir, diffs[i] / diffs[i].norm()).item()
        cosines.append(cos)
    return np.mean(cosines), np.std(cosines)


def compute_causal_inner_product_matrix(directions, diffs_dict, category_names):
    """
    Compute the causal inner product matrix.
    Uses the whitened inner product: ⟨d_A, d_B⟩_C = d_A^T Cov^{-1} d_B
    where Cov is estimated from all difference vectors.
    """
    # Collect all difference vectors for covariance estimation
    all_diffs = []
    for name in category_names:
        all_diffs.append(diffs_dict[name])
    all_diffs = torch.cat(all_diffs, dim=0)  # (N_total, d_model)

    # Estimate covariance
    mean = all_diffs.mean(dim=0, keepdim=True)
    centered = all_diffs - mean
    cov = (centered.T @ centered) / (centered.shape[0] - 1)

    # Regularized inverse (add small ridge for numerical stability)
    d = cov.shape[0]
    cov_reg = cov + 1e-4 * torch.eye(d, device=cov.device)
    cov_inv = torch.linalg.inv(cov_reg)

    # Compute inner product matrix
    n = len(category_names)
    dir_matrix = torch.stack([directions[name] for name in category_names])  # (n, d_model)

    # Causal inner product: d_A^T Cov^{-1} d_B
    whitened = dir_matrix @ cov_inv  # (n, d_model)
    causal_ip = whitened @ dir_matrix.T  # (n, n)

    # Also compute standard cosine similarity matrix
    norms = dir_matrix.norm(dim=1, keepdim=True)
    cosine_matrix = (dir_matrix @ dir_matrix.T) / (norms @ norms.T)

    return causal_ip.cpu().numpy(), cosine_matrix.cpu().numpy()


def composition_fidelity_score(d_A, d_B, W_U, pairs_A, pairs_B):
    """
    Compute Composition Fidelity Score:
    How well does d_A + d_B align with a direction extracted from tokens
    that satisfy both concepts?

    Since arbitrary joint concept pairs are hard to define, we use an
    alternative approach: measure whether the composed direction d_A + d_B
    preserves the ability to probe for both concept A and concept B.

    CFS = cosine(d_A + d_B, d_A) * cosine(d_A + d_B, d_B) — both components preserved
    Also return the raw cosine with each component.
    """
    composed = d_A + d_B
    composed_norm = composed / composed.norm()

    cos_A = torch.dot(composed_norm, d_A).item()
    cos_B = torch.dot(composed_norm, d_B).item()

    # CFS: geometric mean of preservation of both components
    cfs = np.sqrt(max(0, cos_A) * max(0, cos_B))

    return cfs, cos_A, cos_B


def interference_score(d_A, d_B, W_U, pairs_B):
    """
    Measure interference: how much does adding d_A affect the probing accuracy for concept B?

    For each pair in concept B, check if the difference vector (target - base)
    still aligns more with d_B than with d_B + interference from d_A.

    Returns: mean change in alignment when d_A is added as interference.
    """
    interferences = []
    for _, _, tok_base, tok_target in pairs_B:
        diff_B = W_U[tok_target] - W_U[tok_base]
        diff_B_norm = diff_B / diff_B.norm()

        # Original alignment with d_B
        orig_align = torch.dot(diff_B_norm, d_B).item()

        # Alignment of d_A with this pair's difference
        cross_align = torch.dot(diff_B_norm, d_A).item()

        interferences.append(abs(cross_align))

    return np.mean(interferences), np.std(interferences)


def run_direction_extraction(save=True):
    """Main pipeline: extract all directions and compute quality metrics."""
    tokenizer, model, W_U = load_model()

    print("\n=== Loading BATS 3.0 categories ===")
    categories, category_types = load_all_bats_categories(tokenizer)

    print(f"\n=== Extracting directions for {len(categories)} categories ===")
    directions = {}
    diffs_dict = {}
    quality_metrics = {}

    for name, pairs in tqdm(categories.items(), desc="Extracting directions"):
        direction, diffs = extract_concept_direction(pairs, W_U)
        directions[name] = direction
        diffs_dict[name] = diffs

        loo_mean, loo_std = leave_one_out_consistency(pairs, W_U)
        quality_metrics[name] = {
            "n_pairs": len(pairs),
            "type": category_types[name],
            "loo_consistency_mean": float(loo_mean),
            "loo_consistency_std": float(loo_std),
        }

    print("\n=== Direction Quality ===")
    for name, m in sorted(quality_metrics.items(), key=lambda x: x[1]["loo_consistency_mean"], reverse=True):
        print(f"  {name}: LOO={m['loo_consistency_mean']:.3f}±{m['loo_consistency_std']:.3f} (n={m['n_pairs']}, {m['type']})")

    # Filter to categories with reasonable quality (LOO > 0.1)
    good_cats = [n for n, m in quality_metrics.items() if m["loo_consistency_mean"] > 0.1]
    print(f"\n{len(good_cats)} categories with LOO consistency > 0.1")

    cat_names = sorted(good_cats)

    print("\n=== Computing inner product matrices ===")
    causal_ip, cosine_matrix = compute_causal_inner_product_matrix(
        directions, diffs_dict, cat_names
    )

    print("\n=== Computing composition fidelity scores ===")
    n = len(cat_names)
    cfs_matrix = np.zeros((n, n))
    interference_matrix = np.zeros((n, n))
    cos_A_matrix = np.zeros((n, n))
    cos_B_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                cfs_matrix[i, j] = 1.0
                continue
            d_A = directions[cat_names[i]]
            d_B = directions[cat_names[j]]
            cfs, cos_A, cos_B = composition_fidelity_score(
                d_A, d_B, W_U, categories[cat_names[i]], categories[cat_names[j]]
            )
            cfs_matrix[i, j] = cfs
            cos_A_matrix[i, j] = cos_A
            cos_B_matrix[i, j] = cos_B

            inter_mean, inter_std = interference_score(d_A, d_B, W_U, categories[cat_names[j]])
            interference_matrix[i, j] = inter_mean

    # Compute random baseline
    print("\n=== Computing random direction baselines ===")
    d_model = W_U.shape[1]
    n_random = 100
    random_cfs = []
    random_interference = []
    for _ in range(n_random):
        r1 = torch.randn(d_model, device=W_U.device)
        r1 = r1 / r1.norm()
        r2 = torch.randn(d_model, device=W_U.device)
        r2 = r2 / r2.norm()
        composed = r1 + r2
        composed = composed / composed.norm()
        random_cfs.append(np.sqrt(max(0, torch.dot(composed, r1).item()) * max(0, torch.dot(composed, r2).item())))

        # Random interference with real concepts
        idx = random.randint(0, n - 1)
        inter, _ = interference_score(r1, directions[cat_names[idx]], W_U, categories[cat_names[idx]])
        random_interference.append(inter)

    results = {
        "category_names": cat_names,
        "category_types": {n: category_types[n] for n in cat_names},
        "quality_metrics": {n: quality_metrics[n] for n in cat_names},
        "causal_inner_product": causal_ip.tolist(),
        "cosine_similarity": cosine_matrix.tolist(),
        "composition_fidelity": cfs_matrix.tolist(),
        "cos_A_preservation": cos_A_matrix.tolist(),
        "cos_B_preservation": cos_B_matrix.tolist(),
        "interference": interference_matrix.tolist(),
        "random_baseline": {
            "cfs_mean": float(np.mean(random_cfs)),
            "cfs_std": float(np.std(random_cfs)),
            "interference_mean": float(np.mean(random_interference)),
            "interference_std": float(np.std(random_interference)),
        },
        "model": MODEL_NAME,
        "seed": SEED,
    }

    if save:
        out_path = os.path.join(RESULTS_DIR, "direction_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        # Save tensors for later use
        torch.save(
            {name: directions[name].cpu() for name in cat_names},
            os.path.join(RESULTS_DIR, "directions.pt"),
        )
        print("Direction tensors saved.")

    return results


if __name__ == "__main__":
    results = run_direction_extraction()
    print("\n=== Summary ===")
    print(f"Categories analyzed: {len(results['category_names'])}")
    print(f"Random CFS baseline: {results['random_baseline']['cfs_mean']:.3f}±{results['random_baseline']['cfs_std']:.3f}")
