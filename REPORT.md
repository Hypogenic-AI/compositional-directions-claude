# Which Linear Directions Are Compositional?

## 1. Executive Summary

We systematically tested whether linear directions extracted from the residual stream of a Transformer LLM (Pythia-2.8B) form compositional subspaces. Using 25 concept categories from BATS 3.0 spanning morphological, encyclopedic, and lexicographic relations, we extracted concept directions via the mean-difference method and tested their compositionality through geometric analysis (causal inner products, composition fidelity scores) and functional steering experiments. Our key finding is that **compositionality is not a property of direction pairs but of direction quality**: morphological concepts form strong linear directions (LOO consistency 0.38 ± 0.12) that compose well both geometrically and functionally, encyclopedic concepts are moderate (LOO 0.35 ± 0.09), and lexicographic relations (synonyms, antonyms, meronyms) mostly fail to form coherent linear directions at all (many LOO < 0.1, excluded from analysis). Within the space of well-formed directions, composition via vector addition is remarkably robust — functional steering with composed directions d_A + d_B achieves >96% accuracy across all tested pair types, with minimal cross-concept interference.

## 2. Research Question & Motivation

**Hypothesis**: Not all linear directions in the residual stream form coherent subspaces that can be composed; related vectors may compose naturally while others may only appear linear in limited contexts.

**Why this matters**: The linear representation hypothesis underpins key interpretability techniques (linear probing, steering vectors, activation editing). If directions don't compose reliably, these tools have fundamental limits that practitioners must understand. Prior work showed block-diagonal structure (Park et al. 2024), some function vectors compose while others don't (Todd et al. 2024), and superposition can interfere with clean representation (Elhage et al. 2022). However, no systematic map existed of *which* types of directions are compositional.

**Gap**: No prior work tested composition across a broad taxonomy of concept types with both geometric and functional metrics.

## 3. Methodology

### Model
- **Pythia-2.8B** (EleutherAI), 2560-dimensional residual stream, 50304-token vocabulary
- Chosen for: open access (no gated repo), well-studied, similar scale to models in prior work
- Hardware: NVIDIA RTX A6000 (49GB), CUDA 12.5

### Data
- **BATS 3.0** (Gladkova et al. 2016): 40 concept categories with counterfactual word pairs
- After filtering to single-token pairs in Pythia's tokenizer: 32 categories with ≥5 pairs
- After quality filtering (LOO > 0.1): 25 categories (17 morphological, 5 encyclopedic, 3 lexicographic)
- Total unique concept direction pairs tested: 300 (149 within-category, 151 cross-category)

### Direction Extraction
Following Park et al. (2024), concept directions are extracted from the unembedding matrix W_U ∈ ℝ^{50304 × 2560}:

```
d_concept = normalize(mean(W_U[target_i] - W_U[base_i]))
```

where (base_i, target_i) are counterfactual token pairs (e.g., "cat"→"cats" for noun-plural).

### Metrics
1. **Leave-One-Out (LOO) Consistency**: Cosine similarity between a held-out pair's difference vector and the direction from remaining pairs. Measures whether the concept has a single coherent direction.
2. **Causal Inner Product**: d_A^T Cov^{-1} d_B using the whitened inner product from Park et al. (2024). Measures causal relationship between concept directions.
3. **Composition Fidelity Score (CFS)**: geometric_mean(cos(d_A+d_B, d_A), cos(d_A+d_B, d_B)). Measures whether both components are preserved in composition.
4. **Interference Score**: Mean |cos(diff_B, d_A)| over B's pairs. Measures cross-concept contamination.
5. **Functional Steering Accuracy**: Whether adding α·d to a token's final hidden state increases the target token's logit. Tests actual behavioral effect.

### Statistical Tests
- Mann-Whitney U tests for group comparisons
- Pearson and Spearman correlations
- Cohen's d for effect sizes
- Significance level: α = 0.05

### Reproducibility
- Random seed: 42 (all sources)
- Python 3.12.8, PyTorch 2.4.0+cu121, transformers 5.5.0
- Full results saved in `results/` directory

## 4. Results

### 4.1 Direction Quality Varies Dramatically by Concept Type

| Concept Type | LOO Consistency (mean±std) | N categories | Categories surviving filter |
|---|---|---|---|
| Morphological | 0.383 ± 0.117 | 17/18 | 17 |
| Encyclopedic | 0.346 ± 0.092 | 5/5 | 5 |
| Lexicographic | 0.194 ± 0.081 | 3/10 | 3 (7 excluded) |

**Key finding**: Morphological relations (verb tenses, noun plurals, derivational morphology) form the strongest linear directions. Lexicographic relations (synonyms, antonyms, meronyms, hyponyms) mostly *fail* to form coherent linear directions — 7 of 10 categories had LOO < 0.1 and were excluded. This is the most important result: the question "which directions are compositional?" is largely answered by "which concepts have coherent linear representations in the first place?"

Statistical test (morphological vs. lexicographic): Mann-Whitney U = 48.0, p = 0.006, Cohen's d = 1.88 (large effect).

### 4.2 Block-Diagonal Structure in Causal Inner Products

| Metric | Within-Category | Cross-Category | p-value | Cohen's d |
|---|---|---|---|---|
| \|Causal IP\| | 89.07 ± 36.84 | 75.68 ± 31.56 | 3.3×10⁻⁴ | 0.39 |

Within-category concept pairs (e.g., two morphological relations) show significantly higher causal inner products than cross-category pairs, confirming the block-diagonal structure observed by Park et al. (2024). This supports **H1**: related concepts share subspace structure.

### 4.3 Composition Fidelity and the Orthogonality Paradox

| Metric | Within-Category | Cross-Category | Random Baseline |
|---|---|---|---|
| CFS (mean±std) | 0.735 ± 0.083 | 0.706 ± 0.031 | 0.707 ± 0.007 |
| Interference | 0.102 �� 0.070 | 0.044 ± 0.022 | 0.016 ± 0.003 |

**Surprising finding**: Within-category CFS is *higher* than cross-category, contradicting **H3** (which predicted orthogonal pairs would compose better). The explanation is that CFS measures geometric preservation in the sum, and aligned vectors reinforce each other (cos(d_A+d_B, d_A) > 1/√2 when d_A·d_B > 0). However, within-category pairs also show 2.3× higher interference — the tradeoff between geometric CFS and functional interference reveals two distinct senses of "compositionality."

Correlations:
- |Cosine similarity| vs CFS: r = 0.37, p = 3.7×10⁻¹¹ (more aligned → higher CFS)
- Interference vs CFS: r = 0.33, p = 3.2×10⁻⁹ (more interference → higher CFS)
- Quality (LOO) vs interference: r = 0.36, p = 1.9×10⁻¹⁰ (stronger directions → more interference)
- Quality (LOO) vs CFS: r = 0.01, p = 0.80 (no relationship)

### 4.4 Functional Steering Composition

| Pair Type | Single A Acc | Single B Acc | Composed A Acc | Composed B Acc | Composition Success |
|---|---|---|---|---|---|
| Within-morphological | 1.000 | 0.997 | 1.000 | 0.993 | 0.997 |
| Within-encyclopedic | 1.000 | 1.000 | 0.965 | 1.000 | 0.982 |
| Cross-morph-encyc | 1.000 | 1.000 | 0.998 | 0.994 | 0.996 |
| Cross-morph-lex | 1.000 | 0.976 | 0.996 | 0.976 | 0.986 |

**Key finding**: Functional composition via steering is remarkably robust. Adding the composed direction d_A + d_B achieves >96% accuracy on both concepts across all pair types. The maximum interference observed (within-encyclopedic: 0.12) is small. This suggests that in the unembedding space, the directions that survive quality filtering are all effectively compositional for steering purposes.

### 4.5 Compositionality Taxonomy

| Quality Level | Criteria | Morphological | Encyclopedic | Lexicographic | Total |
|---|---|---|---|---|---|
| **Strong** | LOO > 0.4 | 6 | 1 | 0 | 7 |
| **Moderate** | LOO 0.2–0.4 | 10 | 4 | 1 | 15 |
| **Weak** | LOO 0.1–0.2 | 1 | 0 | 2 | 3 |
| **Non-linear** | LOO < 0.1 | 0 | 0 | 7 | 7 (excluded) |

The strongest linear directions are verb tense transformations (inf→3pSg: LOO=0.63, inf→Ving: LOO=0.47, inf→Ved: LOO=0.41) and color associations (things→color: LOO=0.49). Lexicographic relations like synonyms (LOO=0.02), antonyms (LOO=-0.04), and hyponyms (LOO=0.01) are essentially non-linear.

## 5. Analysis & Discussion

### Answer to the Research Question

**Which linear directions are compositional?** The answer is simpler than expected: the directions that form *coherent linear representations in the first place* are all compositional. The real boundary is not between "compositional" and "non-compositional" linear directions — it is between concepts that have linear representations and concepts that don't.

Specifically:
- **Morphological transformations** (verb tenses, pluralization, derivational affixes) are the most reliably linear and compose freely via vector addition
- **Encyclopedic associations** (color, gender, animal properties) are moderately linear and compose adequately
- **Lexicographic semantic relations** (synonymy, antonymy, meronymy, hyponymy) are *not well-represented as linear directions* in the unembedding space — the concept of composing them doesn't arise because the individual directions are not coherent

### Why Lexicographic Relations Fail

Lexicographic relations (synonyms, antonyms, etc.) have near-zero LOO consistency, meaning different word pairs in the same category point in completely different directions. This is likely because:
1. **These are not single transformations**: "synonym" is not a single function like "make plural" — the transformation from "big"→"large" is fundamentally different from "fast"→"quick"
2. **Superposition**: Semantic relationships may be encoded in superposed, distributed representations rather than single directions
3. **Context-dependence**: As Opiełka et al. (2026) showed, the "direction" for an abstract concept may vary with format/context

### Two Senses of Compositionality

Our results reveal a tension between two senses of "compositional":
1. **Geometric composition**: d_A + d_B preserves both component directions (CFS). This is *easier* for aligned directions (within-category) but comes with interference.
2. **Functional composition**: Steering with d_A + d_B achieves both effects independently. This works well for both aligned and orthogonal directions, as long as both individual directions are coherent.

The practical implication: **steering vector composition works well** in the regime where individual directions are well-formed (LOO > 0.2), regardless of whether the directions are aligned or orthogonal.

### Comparison with Prior Work

- **Park et al. (2024)**: Our causal inner product heatmap confirms their block-diagonal structure. We extend their finding by showing this structure holds specifically for morphological and encyclopedic concepts but breaks down for lexicographic relations.
- **Todd et al. (2024)**: Their finding that "some function vectors compose and some don't" aligns with our taxonomy — the compositions that work involve well-defined functional transformations (like our morphological categories), while those that fail likely involve poorly-defined directions.
- **Elhage et al. (2022)**: Superposition likely explains why lexicographic relations don't form clean directions — these are encoded in distributed, superposed representations.

## 6. Limitations

1. **Model**: We tested only Pythia-2.8B. Results may differ for larger models (where superposition may resolve) or architecturally different models.
2. **Unembedding space only**: We extracted directions from the unembedding matrix, not intermediate layers. Compositional structure may differ across layers.
3. **BATS 3.0 coverage**: The dataset covers morphological and basic semantic relations but not abstract behavioral concepts (honesty, sycophancy) tested in steering literature.
4. **Ceiling effect in steering**: Steering accuracy was near-ceiling (>96% everywhere), making it hard to differentiate pair types. Future work should use harder steering tasks or finer-grained metrics (logit shift magnitude, rank changes).
5. **Single-token constraint**: We required both base and target words to be single tokens, limiting the available pairs. Multi-token analysis could capture more concepts.
6. **No ground-truth for "compositional"**: We define compositionality operationally (CFS, steering accuracy) but there is no ground-truth benchmark for direction compositionality.

## 7. Conclusions & Next Steps

### Clear Answer
Not all conceptual relations form linear directions in the residual stream. **Morphological transformations form strong, compositional linear directions; encyclopedic associations form moderate ones; lexicographic semantic relations generally fail to form coherent linear directions at all.** Among well-formed linear directions, composition via vector addition is remarkably robust for both geometric and functional purposes.

### Implications
- **For steering**: Practitioners can compose steering vectors for well-defined transformations (morphological, simple semantic) with confidence. Abstract semantic relations require caution.
- **For interpretability**: The linear representation hypothesis holds strongly for morphological structure and weakly for semantic structure. This should inform where linear probing is expected to work.
- **For theory**: The boundary of linearity appears to be between *regular functional transformations* (which have consistent direction) and *family-resemblance categories* (which don't).

### Recommended Follow-Up
1. Test on larger models (LLaMA-3-8B, Gemma-2-9B) to see if lexicographic relations become more linear
2. Analyze intermediate layers, not just unembedding space
3. Test behavioral concepts (honesty, helpfulness) for compositionality
4. Develop harder steering benchmarks that avoid ceiling effects
5. Study the relationship between superposition and direction quality using SAE-based decomposition

## References

1. Park, K., Choe, Y. J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. ICML 2024.
2. Park, K., Choe, Y. J., Jiang, Y., & Veitch, V. (2025). The Geometry of Categorical and Hierarchical Concepts in Large Language Models. ICLR 2025.
3. Todd, E., et al. (2024). Function Vectors in Large Language Models. ICLR 2024.
4. Opiełka, J., et al. (2026). Causality ≠ Invariance: Function and Concept Vectors in LLMs. ICLR 2026.
5. Elhage, N., et al. (2022). Toy Models of Superposition. Anthropic.
6. Gladkova, A., Drozd, A., & Matsuoka, S. (2016). Analogy-based detection of morphological and semantic relations with word embeddings. NAACL-HLT 2016 (BATS 3.0).
7. Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML 2023.
