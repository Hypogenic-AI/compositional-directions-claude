"""
Analyze composition results: statistical tests, visualizations, and interpretation.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(context="paper", style="white", palette="colorblind", font_scale=1.4)


def load_results():
    with open(os.path.join(RESULTS_DIR, "direction_results.json")) as f:
        return json.load(f)


def classify_pairs(results):
    """Classify all (i,j) pairs as within-category or cross-category."""
    names = results["category_names"]
    types = results["category_types"]
    n = len(names)

    within_pairs = []  # same type (e.g., both morphological)
    cross_pairs = []   # different type

    for i in range(n):
        for j in range(i + 1, n):
            pair_info = {
                "i": i, "j": j,
                "name_i": names[i], "name_j": names[j],
                "type_i": types[names[i]], "type_j": types[names[j]],
                "cfs": results["composition_fidelity"][i][j],
                "causal_ip": abs(results["causal_inner_product"][i][j]),
                "cosine": abs(results["cosine_similarity"][i][j]),
                "interference_ij": results["interference"][i][j],
                "interference_ji": results["interference"][j][i],
                "cos_A": results["cos_A_preservation"][i][j],
                "cos_B": results["cos_B_preservation"][i][j],
            }
            if types[names[i]] == types[names[j]]:
                within_pairs.append(pair_info)
            else:
                cross_pairs.append(pair_info)

    return within_pairs, cross_pairs


def hypothesis_tests(within_pairs, cross_pairs, results):
    """Run statistical tests for each hypothesis."""
    print("=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    # H1: Related concepts share subspace (within-category have non-zero causal IP)
    print("\n--- H1: Within-category concepts have non-zero causal inner products ---")
    within_ips = [p["causal_ip"] for p in within_pairs]
    cross_ips = [p["causal_ip"] for p in cross_pairs]
    t_stat, p_val = stats.mannwhitneyu(within_ips, cross_ips, alternative="greater")
    print(f"  Within-category |causal IP|: {np.mean(within_ips):.4f} ± {np.std(within_ips):.4f}")
    print(f"  Cross-category  |causal IP|: {np.mean(cross_ips):.4f} ± {np.std(cross_ips):.4f}")
    print(f"  Mann-Whitney U test (within > cross): U={t_stat:.1f}, p={p_val:.2e}")
    d = (np.mean(within_ips) - np.mean(cross_ips)) / np.sqrt((np.std(within_ips)**2 + np.std(cross_ips)**2) / 2)
    print(f"  Cohen's d: {d:.3f}")

    # H2: Cross-category concepts are approximately orthogonal
    print("\n--- H2: Cross-category concepts are approximately orthogonal ---")
    cross_cosines = [p["cosine"] for p in cross_pairs]
    t_stat2, p_val2 = stats.ttest_1samp(cross_cosines, 0)
    print(f"  Cross-category |cosine|: {np.mean(cross_cosines):.4f} ± {np.std(cross_cosines):.4f}")
    print(f"  t-test vs 0: t={t_stat2:.2f}, p={p_val2:.2e}")
    # Compare with expected cosine for random unit vectors in d dimensions
    # E[|cos|] ≈ sqrt(2/(pi*d))
    d_model = len(results["causal_inner_product"][0])  # approximation
    # Actually we can compute expected random cosine from the data
    print(f"  (Note: for truly random vectors in high-d space, |cosine| → 0)")

    # H3: Cross-category pairs compose better than within-category
    print("\n--- H3: Cross-category pairs have higher composition fidelity ---")
    within_cfs = [p["cfs"] for p in within_pairs]
    cross_cfs = [p["cfs"] for p in cross_pairs]
    t_stat3, p_val3 = stats.mannwhitneyu(cross_cfs, within_cfs, alternative="greater")
    print(f"  Within-category CFS: {np.mean(within_cfs):.4f} ± {np.std(within_cfs):.4f}")
    print(f"  Cross-category  CFS: {np.mean(cross_cfs):.4f} ± {np.std(cross_cfs):.4f}")
    print(f"  Random baseline CFS: {results['random_baseline']['cfs_mean']:.4f} ± {results['random_baseline']['cfs_std']:.4f}")
    print(f"  Mann-Whitney U test (cross > within): U={t_stat3:.1f}, p={p_val3:.2e}")
    d3 = (np.mean(cross_cfs) - np.mean(within_cfs)) / np.sqrt((np.std(cross_cfs)**2 + np.std(within_cfs)**2) / 2)
    print(f"  Cohen's d: {d3:.3f}")

    # H4: Higher interference → lower composition fidelity
    print("\n--- H4: Correlation between interference and composition fidelity ---")
    all_pairs = within_pairs + cross_pairs
    all_interference = [(p["interference_ij"] + p["interference_ji"]) / 2 for p in all_pairs]
    all_cfs = [p["cfs"] for p in all_pairs]
    r, p_corr = stats.pearsonr(all_interference, all_cfs)
    rho, p_spearman = stats.spearmanr(all_interference, all_cfs)
    print(f"  Pearson r(interference, CFS): r={r:.4f}, p={p_corr:.2e}")
    print(f"  Spearman ρ(interference, CFS): ρ={rho:.4f}, p={p_spearman:.2e}")

    # Correlation between orthogonality and CFS
    print("\n--- Supplementary: Correlation between orthogonality and CFS ---")
    all_cosines = [p["cosine"] for p in all_pairs]
    r2, p2 = stats.pearsonr(all_cosines, all_cfs)
    rho2, ps2 = stats.spearmanr(all_cosines, all_cfs)
    print(f"  Pearson r(|cosine|, CFS): r={r2:.4f}, p={p2:.2e}")
    print(f"  Spearman ρ(|cosine|, CFS): ρ={rho2:.4f}, p={ps2:.2e}")

    return {
        "H1": {"within_ip_mean": np.mean(within_ips), "cross_ip_mean": np.mean(cross_ips),
                "p_value": float(p_val), "cohens_d": float(d)},
        "H2": {"cross_cosine_mean": np.mean(cross_cosines), "p_value": float(p_val2)},
        "H3": {"within_cfs_mean": np.mean(within_cfs), "cross_cfs_mean": np.mean(cross_cfs),
                "p_value": float(p_val3), "cohens_d": float(d3)},
        "H4": {"pearson_r": float(r), "p_value": float(p_corr),
                "spearman_rho": float(rho), "spearman_p": float(p_spearman)},
        "orthogonality_cfs_corr": {"pearson_r": float(r2), "p_value": float(p2)},
    }


def plot_causal_ip_heatmap(results):
    """Plot causal inner product heatmap with category type annotations."""
    names = results["category_names"]
    types = results["category_types"]
    causal_ip = np.array(results["causal_inner_product"])

    # Shorten names for display
    short_names = []
    for n in names:
        # Extract the bracketed part
        if "[" in n:
            short = n.split("[")[1].rstrip("]").strip()
        else:
            short = n
        short_names.append(short[:20])

    # Sort by type for block-diagonal visualization
    type_order = {"morphological": 0, "encyclopedic": 1, "lexicographic": 2}
    sorted_idx = sorted(range(len(names)), key=lambda i: (type_order.get(types[names[i]], 3), names[i]))
    causal_ip_sorted = causal_ip[np.ix_(sorted_idx, sorted_idx)]
    sorted_short_names = [short_names[i] for i in sorted_idx]
    sorted_types = [types[names[i]] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(causal_ip_sorted, cmap="RdBu_r", vmin=-np.percentile(np.abs(causal_ip), 95),
                    vmax=np.percentile(np.abs(causal_ip), 95))
    ax.set_xticks(range(len(sorted_short_names)))
    ax.set_yticks(range(len(sorted_short_names)))
    ax.set_xticklabels(sorted_short_names, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_short_names, fontsize=8)

    # Add type boundary lines
    prev_type = sorted_types[0]
    for idx, t in enumerate(sorted_types):
        if t != prev_type:
            ax.axhline(y=idx - 0.5, color="black", linewidth=2)
            ax.axvline(x=idx - 0.5, color="black", linewidth=2)
            prev_type = t

    plt.colorbar(im, ax=ax, label="Causal Inner Product")
    ax.set_title("Causal Inner Product Between Concept Directions\n(Block structure by concept type)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "causal_ip_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: causal_ip_heatmap.png")


def plot_cfs_heatmap(results):
    """Plot composition fidelity score heatmap."""
    names = results["category_names"]
    types = results["category_types"]
    cfs = np.array(results["composition_fidelity"])

    short_names = []
    for n in names:
        if "[" in n:
            short = n.split("[")[1].rstrip("]").strip()
        else:
            short = n
        short_names.append(short[:20])

    type_order = {"morphological": 0, "encyclopedic": 1, "lexicographic": 2}
    sorted_idx = sorted(range(len(names)), key=lambda i: (type_order.get(types[names[i]], 3), names[i]))
    cfs_sorted = cfs[np.ix_(sorted_idx, sorted_idx)]
    sorted_short_names = [short_names[i] for i in sorted_idx]
    sorted_types = [types[names[i]] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cfs_sorted, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(sorted_short_names)))
    ax.set_yticks(range(len(sorted_short_names)))
    ax.set_xticklabels(sorted_short_names, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_short_names, fontsize=8)

    prev_type = sorted_types[0]
    for idx, t in enumerate(sorted_types):
        if t != prev_type:
            ax.axhline(y=idx - 0.5, color="white", linewidth=2)
            ax.axvline(x=idx - 0.5, color="white", linewidth=2)
            prev_type = t

    plt.colorbar(im, ax=ax, label="Composition Fidelity Score")
    ax.set_title("Composition Fidelity Score (CFS) Between Concept Directions")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cfs_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cfs_heatmap.png")


def plot_orthogonality_vs_cfs(within_pairs, cross_pairs, results):
    """Scatter plot of orthogonality (|cosine|) vs CFS, colored by pair type."""
    fig, ax = plt.subplots(figsize=(10, 7))

    within_cos = [p["cosine"] for p in within_pairs]
    within_cfs = [p["cfs"] for p in within_pairs]
    cross_cos = [p["cosine"] for p in cross_pairs]
    cross_cfs = [p["cfs"] for p in cross_pairs]

    ax.scatter(within_cos, within_cfs, alpha=0.5, label="Within-category", s=30, c="tab:red")
    ax.scatter(cross_cos, cross_cfs, alpha=0.3, label="Cross-category", s=20, c="tab:blue")

    # Random baseline
    ax.axhline(y=results["random_baseline"]["cfs_mean"], color="gray", linestyle="--",
               label=f"Random baseline (CFS={results['random_baseline']['cfs_mean']:.3f})")

    ax.set_xlabel("|Cosine Similarity| Between Directions")
    ax.set_ylabel("Composition Fidelity Score (CFS)")
    ax.set_title("Orthogonality vs. Composition Fidelity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "orthogonality_vs_cfs.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: orthogonality_vs_cfs.png")


def plot_interference_vs_cfs(within_pairs, cross_pairs):
    """Scatter plot of interference vs CFS."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for pairs, label, color, alpha in [
        (within_pairs, "Within-category", "tab:red", 0.5),
        (cross_pairs, "Cross-category", "tab:blue", 0.3),
    ]:
        interference = [(p["interference_ij"] + p["interference_ji"]) / 2 for p in pairs]
        cfs = [p["cfs"] for p in pairs]
        ax.scatter(interference, cfs, alpha=alpha, label=label, s=30 if "Within" in label else 20, c=color)

    ax.set_xlabel("Mean Interference Score")
    ax.set_ylabel("Composition Fidelity Score (CFS)")
    ax.set_title("Interference vs. Composition Fidelity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "interference_vs_cfs.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: interference_vs_cfs.png")


def plot_cfs_by_type(within_pairs, cross_pairs, results):
    """Box plot of CFS by concept pair type."""
    # Group within-category by specific type
    type_groups = {}
    for p in within_pairs:
        t = p["type_i"]
        type_groups.setdefault(f"within-{t}", []).append(p["cfs"])

    # Group cross-category by type pair
    for p in cross_pairs:
        t1, t2 = sorted([p["type_i"], p["type_j"]])
        key = f"cross: {t1[:4]}-{t2[:4]}"
        type_groups.setdefault(key, []).append(p["cfs"])

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = sorted(type_groups.keys())
    data = [type_groups[l] for l in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = []
    for l in labels:
        if "within" in l:
            colors.append("salmon")
        else:
            colors.append("lightblue")
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    ax.axhline(y=results["random_baseline"]["cfs_mean"], color="gray", linestyle="--",
               label="Random baseline")
    ax.set_ylabel("Composition Fidelity Score")
    ax.set_title("CFS by Concept Pair Type")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cfs_by_type.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cfs_by_type.png")


def plot_direction_quality(results):
    """Bar plot of LOO consistency by category."""
    names = results["category_names"]
    metrics = results["quality_metrics"]
    types = results["category_types"]

    sorted_names = sorted(names, key=lambda n: metrics[n]["loo_consistency_mean"], reverse=True)
    short_names = []
    for n in sorted_names:
        if "[" in n:
            short = n.split("[")[1].rstrip("]").strip()
        else:
            short = n
        short_names.append(short[:25])

    means = [metrics[n]["loo_consistency_mean"] for n in sorted_names]
    stds = [metrics[n]["loo_consistency_std"] for n in sorted_names]
    type_colors = {"morphological": "tab:green", "encyclopedic": "tab:orange", "lexicographic": "tab:purple"}
    colors = [type_colors[types[n]] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(means)), means, yerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_ylabel("Leave-One-Out Consistency (cosine)")
    ax.set_title("Direction Quality: Leave-One-Out Consistency by Category")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "direction_quality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: direction_quality.png")


def generate_summary_table(within_pairs, cross_pairs, results, stats_results):
    """Generate a markdown summary table."""
    lines = []
    lines.append("| Metric | Within-Category | Cross-Category | Random Baseline |")
    lines.append("|--------|----------------|----------------|-----------------|")

    within_cfs = [p["cfs"] for p in within_pairs]
    cross_cfs = [p["cfs"] for p in cross_pairs]
    lines.append(f"| CFS (mean±std) | {np.mean(within_cfs):.3f}±{np.std(within_cfs):.3f} | "
                 f"{np.mean(cross_cfs):.3f}±{np.std(cross_cfs):.3f} | "
                 f"{results['random_baseline']['cfs_mean']:.3f}±{results['random_baseline']['cfs_std']:.3f} |")

    within_ips = [p["causal_ip"] for p in within_pairs]
    cross_ips = [p["causal_ip"] for p in cross_pairs]
    lines.append(f"| |Causal IP| (mean±std) | {np.mean(within_ips):.3f}±{np.std(within_ips):.3f} | "
                 f"{np.mean(cross_ips):.3f}±{np.std(cross_ips):.3f} | N/A |")

    within_inter = [(p["interference_ij"] + p["interference_ji"]) / 2 for p in within_pairs]
    cross_inter = [(p["interference_ij"] + p["interference_ji"]) / 2 for p in cross_pairs]
    lines.append(f"| Interference (mean±std) | {np.mean(within_inter):.3f}±{np.std(within_inter):.3f} | "
                 f"{np.mean(cross_inter):.3f}±{np.std(cross_inter):.3f} | "
                 f"{results['random_baseline']['interference_mean']:.3f}±{results['random_baseline']['interference_std']:.3f} |")

    return "\n".join(lines)


def run_analysis():
    """Main analysis pipeline."""
    results = load_results()
    within_pairs, cross_pairs = classify_pairs(results)

    print(f"Total categories: {len(results['category_names'])}")
    print(f"Within-category pairs: {len(within_pairs)}")
    print(f"Cross-category pairs: {len(cross_pairs)}")

    # Statistical tests
    stats_results = hypothesis_tests(within_pairs, cross_pairs, results)

    # Visualizations
    print("\n=== Generating Visualizations ===")
    plot_direction_quality(results)
    plot_causal_ip_heatmap(results)
    plot_cfs_heatmap(results)
    plot_orthogonality_vs_cfs(within_pairs, cross_pairs, results)
    plot_interference_vs_cfs(within_pairs, cross_pairs)
    plot_cfs_by_type(within_pairs, cross_pairs, results)

    # Summary table
    table = generate_summary_table(within_pairs, cross_pairs, results, stats_results)
    print("\n=== Summary Table ===")
    print(table)

    # Save analysis results
    analysis = {
        "stats": stats_results,
        "summary_table": table,
        "n_within_pairs": len(within_pairs),
        "n_cross_pairs": len(cross_pairs),
    }
    with open(os.path.join(RESULTS_DIR, "analysis_results.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print("\nAnalysis saved to results/analysis_results.json")

    return results, stats_results, within_pairs, cross_pairs


if __name__ == "__main__":
    run_analysis()
