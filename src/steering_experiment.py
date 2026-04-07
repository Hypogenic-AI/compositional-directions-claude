"""
Steering composition experiment: test whether composed directions (d_A + d_B)
actually steer model outputs toward satisfying both concepts simultaneously.

This tests FUNCTIONAL compositionality, not just geometric alignment.
"""
import os
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

DEVICE = torch.device("cuda:0")
MODEL_NAME = "EleutherAI/pythia-2.8b"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model_and_directions():
    """Load model and previously extracted directions."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    model.eval()

    # Load directions
    directions = torch.load(os.path.join(RESULTS_DIR, "directions.pt"), weights_only=True)
    with open(os.path.join(RESULTS_DIR, "direction_results.json")) as f:
        results = json.load(f)

    return tokenizer, model, directions, results


def get_unembedding(model):
    """Get unembedding matrix from model."""
    if hasattr(model, "embed_out"):
        return model.embed_out.weight.detach().float()
    elif hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().float()
    for name, param in model.named_parameters():
        if "embed_out" in name or "lm_head" in name:
            return param.detach().float()
    raise RuntimeError("Cannot find unembedding matrix")


def steering_accuracy(model, tokenizer, direction, pairs, W_U, alpha=5.0):
    """
    Test if adding alpha * direction to the residual stream at the last layer
    steers model output toward the target concept.

    For each pair (base, target):
    - Feed base word to model
    - Add steering vector to final hidden state
    - Check if output logits shift toward target tokens

    Returns: fraction of pairs where target logit increases.
    """
    direction = direction.to(DEVICE)
    correct = 0
    total = 0

    for w_base, w_target, tok_base, tok_target in pairs[:30]:  # limit for speed
        # Encode base word
        input_ids = tokenizer.encode(f" {w_base}", return_tensors="pt", add_special_tokens=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            base_logits = outputs.logits[0, -1, :]  # logits at last position

            # Steer: add direction to last hidden state
            hidden = outputs.hidden_states[-1].clone()  # last layer hidden state
            hidden[0, -1, :] = hidden[0, -1, :] + alpha * direction.to(hidden.device)

            # Get steered logits by passing through the unembedding
            steered_logits = hidden[0, -1, :].to(W_U.device) @ W_U.T

        # Check if target logit increased relative to base logit
        base_target_logit = base_logits[tok_target].item()
        steered_target_logit = steered_logits[tok_target].item()

        if steered_target_logit > base_target_logit:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def composition_steering_test(model, tokenizer, dir_A, dir_B, pairs_A, pairs_B, W_U, alpha=5.0):
    """
    Test functional composition: does steering with d_A + d_B
    affect both concept A and concept B?

    Returns:
    - acc_A_single: steering accuracy of d_A alone for concept A
    - acc_B_single: steering accuracy of d_B alone for concept B
    - acc_A_composed: steering accuracy for concept A when using d_A + d_B
    - acc_B_composed: steering accuracy for concept B when using d_A + d_B
    - interference_A: drop in concept-A accuracy from adding d_B
    - interference_B: drop in concept-B accuracy from adding d_A
    """
    dir_composed = dir_A + dir_B
    dir_composed = dir_composed / dir_composed.norm() * np.sqrt(2)  # scale to same magnitude as sum

    acc_A_single = steering_accuracy(model, tokenizer, dir_A, pairs_A, W_U, alpha)
    acc_B_single = steering_accuracy(model, tokenizer, dir_B, pairs_B, W_U, alpha)
    acc_A_composed = steering_accuracy(model, tokenizer, dir_composed, pairs_A, W_U, alpha)
    acc_B_composed = steering_accuracy(model, tokenizer, dir_composed, pairs_B, W_U, alpha)

    return {
        "acc_A_single": acc_A_single,
        "acc_B_single": acc_B_single,
        "acc_A_composed": acc_A_composed,
        "acc_B_composed": acc_B_composed,
        "interference_A": acc_A_single - acc_A_composed,
        "interference_B": acc_B_single - acc_B_composed,
        "composition_success": (acc_A_composed + acc_B_composed) / 2,
    }


def run_steering_experiments():
    """Run steering composition tests for selected concept pairs."""
    print("Loading model and directions...")
    tokenizer, model, directions, prev_results = load_model_and_directions()
    W_U = get_unembedding(model)

    cat_names = prev_results["category_names"]
    cat_types = prev_results["category_types"]

    # Reload BATS pairs
    import glob
    BATS_DIR = "datasets/bats3.0/BATS_3.0"
    type_map = {
        "1_Inflectional_morphology": "morphological",
        "2_Derivational_morphology": "morphological",
        "3_Encyclopedic_semantics": "encyclopedic",
        "4_Lexicographic_semantics": "lexicographic",
    }
    all_pairs = {}
    for type_dir in type_map:
        dir_path = os.path.join(BATS_DIR, type_dir)
        if not os.path.isdir(dir_path):
            continue
        for fpath in sorted(glob.glob(os.path.join(dir_path, "*.txt"))):
            fname = os.path.basename(fpath).replace(".txt", "").strip()
            with open(fpath) as f:
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
                w2 = w2.split("/")[0]
                tok1 = tokenizer.encode(f" {w1}", add_special_tokens=False)
                tok2 = tokenizer.encode(f" {w2}", add_special_tokens=False)
                if len(tok1) == 1 and len(tok2) == 1 and tok1[0] != tok2[0]:
                    pairs.append((w1, w2, tok1[0], tok2[0]))
            if fname in cat_names:
                all_pairs[fname] = pairs

    # Select representative pairs for steering tests
    # We test: within-morphological, within-encyclopedic, cross-morph-encyc, cross-morph-lex
    test_pairs = []

    # Pick representative categories with high LOO consistency
    morph_cats = [c for c in cat_names if cat_types[c] == "morphological"
                  and prev_results["quality_metrics"][c]["loo_consistency_mean"] > 0.3]
    encyc_cats = [c for c in cat_names if cat_types[c] == "encyclopedic"
                  and prev_results["quality_metrics"][c]["loo_consistency_mean"] > 0.2]
    lex_cats = [c for c in cat_names if cat_types[c] == "lexicographic"
                and prev_results["quality_metrics"][c]["loo_consistency_mean"] > 0.1]

    print(f"High-quality morphological: {len(morph_cats)}")
    print(f"High-quality encyclopedic: {len(encyc_cats)}")
    print(f"High-quality lexicographic: {len(lex_cats)}")

    # Within-morphological pairs (up to 10)
    from itertools import combinations
    for a, b in list(combinations(morph_cats, 2))[:10]:
        test_pairs.append((a, b, "within-morphological"))

    # Within-encyclopedic
    for a, b in list(combinations(encyc_cats, 2))[:5]:
        test_pairs.append((a, b, "within-encyclopedic"))

    # Cross morph-encyc
    for a in morph_cats[:5]:
        for b in encyc_cats[:3]:
            test_pairs.append((a, b, "cross-morph-encyc"))

    # Cross morph-lex
    for a in morph_cats[:5]:
        for b in lex_cats[:3]:
            test_pairs.append((a, b, "cross-morph-lex"))

    print(f"\nTesting {len(test_pairs)} concept pairs for functional composition...")

    steering_results = []
    for cat_a, cat_b, pair_type in tqdm(test_pairs, desc="Steering tests"):
        dir_A = directions[cat_a].to(DEVICE)
        dir_B = directions[cat_b].to(DEVICE)

        result = composition_steering_test(
            model, tokenizer, dir_A, dir_B, all_pairs[cat_a], all_pairs[cat_b], W_U
        )
        result["cat_A"] = cat_a
        result["cat_B"] = cat_b
        result["pair_type"] = pair_type
        result["type_A"] = cat_types[cat_a]
        result["type_B"] = cat_types[cat_b]
        steering_results.append(result)

    # Save results
    with open(os.path.join(RESULTS_DIR, "steering_results.json"), "w") as f:
        json.dump(steering_results, f, indent=2)

    # Print summary
    print("\n=== Steering Composition Results ===")
    for ptype in ["within-morphological", "within-encyclopedic", "cross-morph-encyc", "cross-morph-lex"]:
        subset = [r for r in steering_results if r["pair_type"] == ptype]
        if not subset:
            continue
        acc_A_s = np.mean([r["acc_A_single"] for r in subset])
        acc_B_s = np.mean([r["acc_B_single"] for r in subset])
        acc_A_c = np.mean([r["acc_A_composed"] for r in subset])
        acc_B_c = np.mean([r["acc_B_composed"] for r in subset])
        inter_A = np.mean([r["interference_A"] for r in subset])
        inter_B = np.mean([r["interference_B"] for r in subset])
        comp_succ = np.mean([r["composition_success"] for r in subset])
        print(f"\n{ptype} (n={len(subset)}):")
        print(f"  Single A acc: {acc_A_s:.3f}, Single B acc: {acc_B_s:.3f}")
        print(f"  Composed A acc: {acc_A_c:.3f}, Composed B acc: {acc_B_c:.3f}")
        print(f"  Interference A: {inter_A:+.3f}, B: {inter_B:+.3f}")
        print(f"  Composition success: {comp_succ:.3f}")

    return steering_results


if __name__ == "__main__":
    run_steering_experiments()
