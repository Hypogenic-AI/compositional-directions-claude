"""
Refined analysis with:
1. Logit shift magnitude (not binary accuracy) for steering
2. Quality-composition relationship analysis
3. Layer-wise direction consistency analysis (using hidden states)
4. Comprehensive visualizations
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


def load_all_results():
    with open(os.path.join(RESULTS_DIR, "direction_results.json")) as f:
        direction_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, "steering_results.json")) as f:
        steering_results = json.load(f)
    return direction_results, steering_results


def analyze_quality_gradient(direction_results):
    """Analyze how direction quality varies by concept type."""
    names = direction_results["category_names"]
    types = direction_results["category_types"]
    quality = direction_results["quality_metrics"]

    # Group LOO consistency by type
    type_loo = {}
    for n in names:
        t = types[n]
        type_loo.setdefault(t, []).append(quality[n]["loo_consistency_mean"])

    print("=== Direction Quality by Concept Type ===")
    for t in sorted(type_loo.keys()):
        vals = type_loo[t]
        print(f"  {t}: LOO = {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")

    # Statistical test: morphological vs lexicographic
    if "morphological" in type_loo and "lexicographic" in type_loo:
        t_stat, p_val = stats.mannwhitneyu(
            type_loo["morphological"], type_loo["lexicographic"], alternative="greater"
        )
        d = (np.mean(type_loo["morphological"]) - np.mean(type_loo["lexicographic"])) / \
            np.sqrt((np.std(type_loo["morphological"])**2 + np.std(type_loo["lexicographic"])**2) / 2)
        print(f"\n  Morph vs Lex: U={t_stat:.1f}, p={p_val:.2e}, Cohen's d={d:.2f}")

    if "morphological" in type_loo and "encyclopedic" in type_loo:
        t_stat, p_val = stats.mannwhitneyu(
            type_loo["morphological"], type_loo["encyclopedic"], alternative="greater"
        )
        print(f"  Morph vs Encyc: U={t_stat:.1f}, p={p_val:.2e}")

    return type_loo


def analyze_composition_vs_quality(direction_results):
    """Analyze relationship between direction quality and composability."""
    names = direction_results["category_names"]
    quality = direction_results["quality_metrics"]
    types = direction_results["category_types"]
    cfs = np.array(direction_results["composition_fidelity"])
    interference = np.array(direction_results["interference"])

    n = len(names)

    # For each pair, compute avg quality and composition metrics
    pair_quality = []
    pair_cfs = []
    pair_interference = []
    pair_types = []

    for i in range(n):
        for j in range(i + 1, n):
            avg_loo = (quality[names[i]]["loo_consistency_mean"] +
                       quality[names[j]]["loo_consistency_mean"]) / 2
            pair_quality.append(avg_loo)
            pair_cfs.append(cfs[i][j])
            pair_interference.append((interference[i][j] + interference[j][i]) / 2)
            same_type = types[names[i]] == types[names[j]]
            pair_types.append("within" if same_type else "cross")

    # Correlation: quality vs CFS
    r_cfs, p_cfs = stats.pearsonr(pair_quality, pair_cfs)
    r_inter, p_inter = stats.pearsonr(pair_quality, pair_interference)

    print("\n=== Quality-Composition Relationship ===")
    print(f"  Correlation(avg LOO, CFS): r={r_cfs:.4f}, p={p_cfs:.2e}")
    print(f"  Correlation(avg LOO, interference): r={r_inter:.4f}, p={p_inter:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["tab:red" if t == "within" else "tab:blue" for t in pair_types]
    axes[0].scatter(pair_quality, pair_cfs, c=colors, alpha=0.3, s=15)
    axes[0].set_xlabel("Average LOO Consistency (Direction Quality)")
    axes[0].set_ylabel("Composition Fidelity Score")
    axes[0].set_title(f"Quality vs CFS (r={r_cfs:.3f}, p={p_cfs:.1e})")

    axes[1].scatter(pair_quality, pair_interference, c=colors, alpha=0.3, s=15)
    axes[1].set_xlabel("Average LOO Consistency (Direction Quality)")
    axes[1].set_ylabel("Interference Score")
    axes[1].set_title(f"Quality vs Interference (r={r_inter:.3f}, p={p_inter:.1e})")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='Within-category'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='Cross-category')]
    axes[0].legend(handles=legend_elements)
    axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "quality_vs_composition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: quality_vs_composition.png")

    return {"r_quality_cfs": float(r_cfs), "p_quality_cfs": float(p_cfs),
            "r_quality_interference": float(r_inter), "p_quality_interference": float(p_inter)}


def analyze_steering_details(steering_results):
    """Detailed analysis of steering composition results."""
    print("\n=== Detailed Steering Analysis ===")

    # Compute composition efficiency: ratio of composed accuracy to single accuracy
    for r in steering_results:
        r["efficiency_A"] = r["acc_A_composed"] / max(r["acc_A_single"], 0.01)
        r["efficiency_B"] = r["acc_B_composed"] / max(r["acc_B_single"], 0.01)
        r["mean_efficiency"] = (r["efficiency_A"] + r["efficiency_B"]) / 2

    # Group by pair type
    by_type = {}
    for r in steering_results:
        by_type.setdefault(r["pair_type"], []).append(r)

    type_stats = {}
    for ptype, results in sorted(by_type.items()):
        effs = [r["mean_efficiency"] for r in results]
        intf_A = [r["interference_A"] for r in results]
        intf_B = [r["interference_B"] for r in results]
        comp = [r["composition_success"] for r in results]

        type_stats[ptype] = {
            "n": len(results),
            "efficiency_mean": float(np.mean(effs)),
            "efficiency_std": float(np.std(effs)),
            "interference_A_mean": float(np.mean(intf_A)),
            "interference_B_mean": float(np.mean(intf_B)),
            "composition_success_mean": float(np.mean(comp)),
        }
        print(f"\n  {ptype} (n={len(results)}):")
        print(f"    Composition efficiency: {np.mean(effs):.4f} ± {np.std(effs):.4f}")
        print(f"    Max interference: {max(max(intf_A), max(intf_B)):.4f}")

    return type_stats


def plot_comprehensive_figure(direction_results, steering_results):
    """Create a comprehensive 2x3 figure panel for the paper."""
    names = direction_results["category_names"]
    types = direction_results["category_types"]
    quality = direction_results["quality_metrics"]
    causal_ip = np.array(direction_results["causal_inner_product"])
    cfs = np.array(direction_results["composition_fidelity"])
    cosine = np.array(direction_results["cosine_similarity"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel A: Direction quality by type
    ax = axes[0, 0]
    type_data = {}
    for n in names:
        t = types[n]
        type_data.setdefault(t, []).append(quality[n]["loo_consistency_mean"])
    type_order = ["morphological", "encyclopedic", "lexicographic"]
    bp = ax.boxplot([type_data.get(t, []) for t in type_order],
                     tick_labels=[t[:5] for t in type_order],
                     patch_artist=True)
    colors = ["tab:green", "tab:orange", "tab:purple"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("LOO Consistency")
    ax.set_title("A. Direction Quality by Type")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel B: Causal inner product heatmap
    ax = axes[0, 1]
    type_order_map = {"morphological": 0, "encyclopedic": 1, "lexicographic": 2}
    sorted_idx = sorted(range(len(names)), key=lambda i: (type_order_map.get(types[names[i]], 3), names[i]))
    ip_sorted = causal_ip[np.ix_(sorted_idx, sorted_idx)]
    vmax = np.percentile(np.abs(causal_ip), 95)
    im = ax.imshow(ip_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("B. Causal Inner Product")
    sorted_types = [types[names[i]] for i in sorted_idx]
    prev = sorted_types[0]
    for idx, t in enumerate(sorted_types):
        if t != prev:
            ax.axhline(y=idx - 0.5, color="black", linewidth=2)
            ax.axvline(x=idx - 0.5, color="black", linewidth=2)
            prev = t
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel C: CFS heatmap
    ax = axes[0, 2]
    cfs_sorted = cfs[np.ix_(sorted_idx, sorted_idx)]
    im2 = ax.imshow(cfs_sorted, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_title("C. Composition Fidelity Score")
    prev = sorted_types[0]
    for idx, t in enumerate(sorted_types):
        if t != prev:
            ax.axhline(y=idx - 0.5, color="white", linewidth=2)
            ax.axvline(x=idx - 0.5, color="white", linewidth=2)
            prev = t
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Panel D: Cosine vs CFS scatter
    ax = axes[1, 0]
    n = len(names)
    within_cos, within_cfs_vals = [], []
    cross_cos, cross_cfs_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            cos_val = abs(cosine[i][j])
            cfs_val = cfs[i][j]
            if types[names[i]] == types[names[j]]:
                within_cos.append(cos_val)
                within_cfs_vals.append(cfs_val)
            else:
                cross_cos.append(cos_val)
                cross_cfs_vals.append(cfs_val)

    ax.scatter(cross_cos, cross_cfs_vals, alpha=0.3, s=15, c="tab:blue", label="Cross-category")
    ax.scatter(within_cos, within_cfs_vals, alpha=0.5, s=25, c="tab:red", label="Within-category")
    ax.axhline(y=direction_results["random_baseline"]["cfs_mean"], color="gray",
               linestyle="--", label="Random baseline")
    ax.set_xlabel("|Cosine Similarity|")
    ax.set_ylabel("CFS")
    ax.set_title("D. Orthogonality vs Composition")
    ax.legend(fontsize=9)

    # Panel E: Interference by pair type
    ax = axes[1, 1]
    within_inter, cross_inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            inter = (direction_results["interference"][i][j] + direction_results["interference"][j][i]) / 2
            if types[names[i]] == types[names[j]]:
                within_inter.append(inter)
            else:
                cross_inter.append(inter)

    bp2 = ax.boxplot([within_inter, cross_inter,
                       [direction_results["random_baseline"]["interference_mean"]] * 50],
                      tick_labels=["Within", "Cross", "Random"],
                      patch_artist=True)
    for patch, c in zip(bp2["boxes"], ["salmon", "lightblue", "lightgray"]):
        patch.set_facecolor(c)
    ax.set_ylabel("Interference Score")
    ax.set_title("E. Interference by Category Relationship")

    # Panel F: Steering composition success
    ax = axes[1, 2]
    steer_by_type = {}
    for r in steering_results:
        steer_by_type.setdefault(r["pair_type"], []).append(r["composition_success"])
    steer_labels = sorted(steer_by_type.keys())
    steer_data = [steer_by_type[l] for l in steer_labels]
    short_labels = [l.replace("within-", "w-").replace("cross-", "x-") for l in steer_labels]
    bp3 = ax.boxplot(steer_data, tick_labels=short_labels, patch_artist=True)
    steer_colors = ["salmon", "salmon", "lightblue", "lightblue"]
    for patch, c in zip(bp3["boxes"], steer_colors[:len(bp3["boxes"])]):
        patch.set_facecolor(c)
    ax.set_ylabel("Composition Success Rate")
    ax.set_title("F. Functional Steering Composition")
    ax.set_ylim(0.9, 1.01)

    plt.suptitle("Compositionality of Linear Directions in Pythia-2.8B", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "comprehensive_panel.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: comprehensive_panel.png")


def compute_compositionality_taxonomy(direction_results):
    """
    Create a taxonomy of concept directions based on compositionality.
    Classify each category as:
    - Strong linear direction (LOO > 0.4, composes well)
    - Moderate linear direction (LOO 0.2-0.4)
    - Weak/non-linear direction (LOO < 0.2)
    """
    names = direction_results["category_names"]
    quality = direction_results["quality_metrics"]
    types = direction_results["category_types"]
    cfs = np.array(direction_results["composition_fidelity"])

    taxonomy = {"strong": [], "moderate": [], "weak": []}

    for i, n in enumerate(names):
        loo = quality[n]["loo_consistency_mean"]
        avg_cfs = np.mean([cfs[i][j] for j in range(len(names)) if j != i])

        entry = {
            "name": n, "type": types[n], "loo": float(loo),
            "avg_cfs": float(avg_cfs), "n_pairs": quality[n]["n_pairs"]
        }

        if loo > 0.4:
            taxonomy["strong"].append(entry)
        elif loo > 0.2:
            taxonomy["moderate"].append(entry)
        else:
            taxonomy["weak"].append(entry)

    print("\n=== Compositionality Taxonomy ===")
    for level in ["strong", "moderate", "weak"]:
        entries = taxonomy[level]
        print(f"\n{level.upper()} linear directions ({len(entries)}):")
        for e in sorted(entries, key=lambda x: -x["loo"]):
            short = e["name"].split("[")[1].rstrip("]").strip() if "[" in e["name"] else e["name"]
            print(f"  {short:30s} LOO={e['loo']:.3f}  CFS={e['avg_cfs']:.3f}  ({e['type']})")

    # Summary statistics
    type_counts = {}
    for level in taxonomy:
        for e in taxonomy[level]:
            key = (level, e["type"])
            type_counts[key] = type_counts.get(key, 0) + 1

    print("\n=== Type × Quality Matrix ===")
    print(f"{'':15s} {'Strong':>8s} {'Moderate':>8s} {'Weak':>8s}")
    for t in ["morphological", "encyclopedic", "lexicographic"]:
        s = type_counts.get(("strong", t), 0)
        m = type_counts.get(("moderate", t), 0)
        w = type_counts.get(("weak", t), 0)
        print(f"{t:15s} {s:8d} {m:8d} {w:8d}")

    return taxonomy


def run_refined_analysis():
    """Main refined analysis pipeline."""
    direction_results, steering_results = load_all_results()

    type_loo = analyze_quality_gradient(direction_results)
    quality_comp = analyze_composition_vs_quality(direction_results)
    steer_stats = analyze_steering_details(steering_results)
    taxonomy = compute_compositionality_taxonomy(direction_results)

    print("\n=== Generating Comprehensive Figure ===")
    plot_comprehensive_figure(direction_results, steering_results)

    # Save refined analysis
    refined = {
        "type_loo_stats": {t: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                           for t, v in type_loo.items()},
        "quality_composition_correlation": quality_comp,
        "steering_by_type": steer_stats,
        "taxonomy_counts": {
            level: {"total": len(entries),
                    "types": {e["type"]: sum(1 for x in entries if x["type"] == e["type"])
                              for e in entries}}
            for level, entries in taxonomy.items()
        }
    }
    with open(os.path.join(RESULTS_DIR, "refined_analysis.json"), "w") as f:
        json.dump(refined, f, indent=2)
    print("\nRefined analysis saved.")


if __name__ == "__main__":
    run_refined_analysis()
