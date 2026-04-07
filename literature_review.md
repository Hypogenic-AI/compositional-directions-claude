# Literature Review: Which Linear Directions Are Compositional?

## Research Area Overview

The "linear representation hypothesis" posits that high-level semantic concepts are encoded as linear directions in the representation spaces of Transformer LLMs. This literature review surveys the theoretical foundations, empirical evidence, and open questions around whether these linear directions form coherent subspaces that can be composed—i.e., which directions are compositional and which are singular.

## Key Papers

### 1. Park et al. (2024a) — "The Linear Representation Hypothesis and the Geometry of Large Language Models" (ICML 2024)

**Key Contribution:** Formalizes the linear representation hypothesis using counterfactuals. Shows three notions of "linear representation" (subspace, measurement, intervention) are unified under a *causal inner product*.

**Methodology:**
- Defines concepts as latent variables with counterfactual outputs (e.g., male⇒female yields {"man","woman"}, {"king","queen"}, ...)
- *Unembedding representation* γ̄_W: concept direction in token output space → connected to linear probing (Theorem 2.2)
- *Embedding representation* λ̄_W: concept direction in context space → connected to steering/intervention (Theorem 2.5)
- *Causal inner product*: an inner product where causally separable concepts are orthogonal. Estimated as `⟨γ̄, γ̄'⟩_C = γ̄ᵀ Cov(γ)⁻¹ γ̄'`

**Key Results:**
- 27 concepts tested on LLaMA-2-7B using BATS 3.0 counterfactual pairs
- Block-diagonal structure in inner-product heatmap: related concepts (verb tenses, language pairs) have non-zero inner products; unrelated concepts are orthogonal
- Concept directions act as linear probes AND intervention vectors
- **Critical for our research:** Causally separable concepts (e.g., gender and language) are orthogonal → they live in independent subspaces. But *related* concepts (e.g., verb→3pSg and verb→Ving) have non-zero inner products → they share subspace structure.

**Datasets Used:** BATS 3.0, bilingual lexicons (word2word), Wikipedia
**Code:** github.com/KihoPark/linear_rep_geometry

---

### 2. Park et al. (2024b) — "The Geometry of Categorical and Hierarchical Concepts in LLMs" (ICLR 2025)

**Key Contribution:** Extends linear representation from binary concepts (directions) to categorical concepts (polytopes) and shows hierarchical relationships are encoded as orthogonality.

**Methodology:**
- Binary features (e.g., `is_animal`) get *vector representations* (with magnitude), not just directions
- Categorical concepts ({mammal, bird, reptile, fish}) are polytopes (convex hulls of feature vectors)
- **Theorem 8 (Hierarchical Orthogonality):** If concept Z is subordinate to W (e.g., mammal⊂animal), then ℓ̄_animal ⊥ (ℓ̄_mammal - ℓ̄_animal) — the "refining" direction is orthogonal to the "coarser" concept

**Key Results:**
- Validated on Gemma-2B and LLaMA-3-8B using 900+ concepts from WordNet
- Hierarchical concepts decompose as direct sums of polytopes
- **Critical for our research:** This is the clearest evidence for *compositional subspaces*. Related concepts (animal → mammal → dog) live in orthogonal *sub*-directions that compose additively. The paper proves this structure follows from the definition of linear representation.

**Datasets Used:** WordNet hierarchy, model vocabularies
**Code:** github.com/KihoPark/LLM_Categorical_Hierarchical_Representations

---

### 3. Todd et al. (2024) — "Function Vectors in Large Language Models" (ICLR 2024)

**Key Contribution:** Discovers that ICL tasks are represented as compact *function vectors* (FVs) in transformer hidden states, extractable by summing outputs of ~10 causal attention heads.

**Methodology:**
- Causal mediation analysis identifies attention heads that transport task information during ICL
- FV = sum of mean activations of top causal heads across ICL prompts
- Tests 40+ ICL tasks across GPT-J, GPT-NeoX, LLaMA-2 (7B–70B)

**Key Results on Compositionality (Section 3.3):**
- **FVs obey vector algebra over functions**: v*_BD = v_AD + v_BC - v_AC (parallelogram rule)
- E.g., Last-Capital* = Last-Copy + First-Capital - First-Copy
- Some compositions work well (Last-Country-Capital: 0.60 vs ICL 0.32), others fail (Last-Antonym: 0.07 vs ICL 0.25)
- **Critical for our research:** This is direct evidence that *some* linear directions compose and others don't. The composable tasks appear to involve complementary mechanisms (word selection + word transformation). Tasks that resist composition may involve single entangled mechanisms.

**Also important:** FVs encode more than just output vocabulary — they carry "procedural" information beyond decoded tokens.

**Datasets Used:** Custom ICL task datasets (40+ tasks)
**Code:** functions.baulab.info

---

### 4. Opiełka et al. (2026) — "Causality ≠ Invariance: Function and Concept Vectors in LLMs" (ICLR 2026)

**Key Contribution:** Shows that FVs are NOT format-invariant—same concept extracted from different formats (open-ended vs multiple-choice) yields nearly orthogonal FVs. Introduces *Concept Vectors* (CVs) using RSA that ARE format-invariant.

**Methodology:**
- Compares activation patching (AP, selects causal heads → FVs) with Representational Similarity Analysis (RSA, selects format-invariant heads → CVs)
- 7 concepts × 3 formats × 4 models (Llama 3.1 8B/70B, Qwen 2.5 7B/72B)

**Key Results:**
- FV heads and CV heads are largely disjoint despite appearing in similar layers
- FVs excel in-distribution; CVs generalize better out-of-distribution
- **Critical for our research:** This reveals that "linear directions" are not monolithic. There are at least two kinds: (1) causal/functional directions that drive task performance but mix concept with format, and (2) invariant/conceptual directions that encode abstract concepts stably. Only the latter are truly "compositional" in the sense of being reusable across contexts.

---

### 5. Elhage et al. (2022) — "Toy Models of Superposition" (Anthropic)

**Key Contribution:** Shows that neural networks represent more features than they have dimensions through *superposition*—features are encoded as nearly-orthogonal directions in a lower-dimensional space.

**Key Results:**
- When features are sparse, networks pack them into fewer dimensions using near-orthogonal directions
- Feature importance and sparsity determine whether a feature gets a dedicated dimension vs. being superposed
- Phase transitions: as sparsity increases, features transition from dedicated neurons → superposed directions
- **Critical for our research:** Superposition means many "linear directions" are NOT independent subspaces—they are approximately orthogonal but interfere. This is a key mechanism by which directions FAIL to be compositional: superposed features cannot be cleanly composed because adding one direction partially activates others.

---

### 6. Zou et al. (2023) — "Representation Engineering" (arXiv)

**Key Contribution:** Introduces RepE, a top-down approach using population-level representations (not individual neurons) to monitor and manipulate cognitive phenomena in LLMs.

**Relevance:** Shows that contrastive pairs can extract "reading vectors" for high-level concepts (honesty, fairness, etc.) that work as steering vectors. Demonstrates that even abstract behavioral concepts have approximately linear representations, but their compositionality is not systematically studied.

---

### 7. Panickssery et al. (2023) — "Steering Llama 2 via Contrastive Activation Addition" (arXiv)

**Key Contribution:** CAA computes steering vectors by averaging residual stream activation differences between positive/negative behavioral examples.

**Relevance:** Shows steering vectors for behavioral concepts (sycophancy, hallucination, etc.) can be extracted and applied. But steering multiple behaviors simultaneously (composing vectors) is not systematically validated—a gap our research addresses.

---

### 8. Bereska et al. (2025) — "Superposition as Lossy Compression"

**Key Contribution:** Information-theoretic framework measuring superposition via sparse autoencoders. Connects superposition to adversarial vulnerability.

**Relevance:** Provides tools to *measure* how much superposition exists, which directly relates to how compositional directions can be—more superposition means less clean composability.

---

### 9. Hänni et al. (2024) — "Mathematical Models of Computation in Superposition"

**Key Contribution:** Studies *computational* superposition (not just representational), where superposition is used during computation, not just storage.

**Relevance:** Even if features are cleanly stored, computation on superposed features introduces interference. This suggests composition may fail at the computation stage even if representations are clean.

---

### 10. Marks et al. (2024) — "Sparse Feature Circuits"

**Key Contribution:** Discovers causally implicated subnetworks of SAE-identified features. Features form interpretable circuits.

**Relevance:** Feature circuits provide a mechanism-level understanding of how features interact—essential for understanding when composition succeeds or fails.

---

## Common Methodologies

1. **Counterfactual pairs** (Park et al.): Define concepts via word pairs that differ on exactly one attribute. Used for both probing and steering.
2. **Causal mediation / activation patching** (Todd et al., Opiełka et al.): Identify model components causally responsible for specific behaviors.
3. **Contrastive activation differences** (Zou et al., Panickssery et al.): Extract concept directions by averaging activation differences between positive/negative examples.
4. **Sparse autoencoders** (Bricken et al., Marks et al., Bereska et al.): Decompose activations into interpretable features to study superposition.
5. **Causal inner product** (Park et al.): Transform representation space so Euclidean operations respect causal structure: `g(y) = Cov(γ)^{-1/2}(γ(y) - E[γ])`

## Standard Baselines

- **Euclidean inner product**: Baseline comparison for the causal inner product
- **Random directions**: Baseline for verifying concept directions are meaningful
- **Linear probes**: Standard method for testing whether information is linearly accessible
- **ICL performance**: Baseline for function vector effectiveness
- **PCA / mean difference**: Simple baselines for extracting concept directions

## Evaluation Metrics

- **Cosine similarity** between concept directions (with appropriate inner product)
- **Linear probing accuracy**: How well a direction separates concept values
- **Steering accuracy**: How well adding a direction changes model behavior on target concept without affecting off-target concepts
- **Orthogonality measures**: Inner products between representations of related/unrelated concepts
- **Composition accuracy**: Performance of composed vectors (v_A + v_B) vs. individual vectors

## Datasets in the Literature

| Dataset | Used By | Purpose |
|---------|---------|---------|
| BATS 3.0 | Park et al. 2024a | Counterfactual word pairs for 22 morphological/semantic concepts |
| WordNet | Park et al. 2024b | Hierarchical concept taxonomy (900+ concepts) |
| word2word bilingual lexicons | Park et al. 2024a | Cross-lingual counterfactual pairs |
| Custom ICL tasks (40+) | Todd et al. 2024 | Function vector extraction and composition |
| Google analogy dataset | Mikolov et al. 2013 | Classic word analogy benchmarks |
| Wikipedia (multilingual) | Park et al. 2024a | Natural text for probing experiments |

## Gaps and Opportunities

1. **No systematic taxonomy of which directions compose.** Park et al. show hierarchical concepts compose; Todd et al. show some FVs compose. But no one has mapped the *boundary* between compositional and non-compositional directions.

2. **Superposition vs. composition trade-off is unexplored.** Elhage et al. show features superpose; Park et al. show clean concepts are orthogonal. The relationship between degree of superposition and composability is not studied.

3. **Format-dependence of composition.** Opiełka et al. show FVs mix concept with format. Whether *compositional* structure is preserved across formats is unknown.

4. **Layer-wise composition analysis.** Most work studies the final layer (unembedding space) or a single intermediate layer. How composability varies across layers is unstudied.

5. **Quantitative metrics for compositionality.** No standard metric exists for measuring "how compositional" a set of directions is.

## Recommendations for Our Experiment

### Primary Datasets
1. **BATS 3.0** — provides counterfactual pairs for extracting concept directions across semantic and morphological categories
2. **WordNet via NLTK** — provides hierarchical concept structure for testing compositional orthogonality
3. **Custom ICL task data** from Todd et al. — for testing function vector composition

### Recommended Baselines
1. Euclidean vs. causal inner product for measuring orthogonality
2. Individual concept directions vs. composed directions for measuring composition quality
3. Random directions as null baseline

### Recommended Metrics
1. Cosine similarity (under causal inner product) between related and unrelated concept pairs
2. Steering accuracy (single concept vs. composed concepts)
3. Interference measure: does adding direction A affect concept B?
4. Composition fidelity: does v_A + v_B achieve the same effect as v_{A+B} extracted directly?

### Recommended Models
- **Gemma-2B** or **LLaMA-3-8B** (used in Park et al. 2024b, manageable size)
- Larger models (LLaMA-2-7B, LLaMA-2-13B) for validation

### Methodological Considerations
- Always use the causal inner product (Cov(γ)^{-1/2} whitening) rather than Euclidean
- Extract directions from both unembedding space (clean, theory-grounded) and intermediate layers (more practical)
- Test composition at multiple granularities: word-level concepts, categorical concepts, behavioral concepts
- Control for superposition effects by measuring feature sparsity
