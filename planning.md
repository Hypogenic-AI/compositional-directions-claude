# Research Plan: Which Linear Directions Are Compositional?

## Motivation & Novelty Assessment

### Why This Research Matters
The linear representation hypothesis — that high-level concepts are encoded as linear directions in LLM residual streams — is foundational to mechanistic interpretability. If these directions compose reliably (v_A + v_B yields the joint concept A∧B), then representation engineering becomes a principled tool for understanding and steering LLMs. If they don't compose, the field needs to understand *when and why* composition fails, so that practitioners know the limits of linear probing, steering vectors, and activation editing.

### Gap in Existing Work
Park et al. (2024) proved that causally separable concepts yield orthogonal directions and showed block-diagonal structure among related concepts. Todd et al. (2024) demonstrated that some function vectors compose via vector algebra (e.g., Last-Capital = Last-Copy + First-Capital - First-Copy) while others fail badly. However, **no one has systematically mapped the boundary between compositional and non-compositional directions**. Specifically:
- Park et al. tested orthogonality but not additive composition of directions
- Todd et al. tested composition of function vectors but only for ~10 composite tasks
- No work measures *degree of compositionality* across a broad taxonomy of concept types

### Our Novel Contribution
We systematically test compositionality of linear directions across 4 concept categories (morphological, encyclopedic, lexicographic, functional) using a model's unembedding space. We introduce a **Composition Fidelity Score (CFS)** that measures how well the sum of two independently-extracted directions matches a direction extracted from jointly-defined concepts. We also measure **interference** — whether composing directions degrades performance on individual concepts. This creates the first empirical map of which types of directions compose and which don't.

### Experiment Justification
- **Experiment 1 (Direction Quality)**: Verify that extracted concept directions are meaningful (high leave-one-out consistency, above-random probing accuracy). Without this, composition tests are meaningless.
- **Experiment 2 (Within-Category Composition)**: Test whether related directions (e.g., noun-plural + verb-past within morphology) compose. Literature suggests they should (Park's block-diagonal structure).
- **Experiment 3 (Cross-Category Composition)**: Test whether unrelated directions (e.g., morphological + encyclopedic) compose. Literature suggests they should compose better (orthogonal → no interference) or fail (superposition interference).
- **Experiment 4 (Steering Composition)**: Test whether composed steering vectors achieve their intended effect without side effects. This is the practical test of compositionality.

## Research Question
Which linear directions in the residual stream of Transformer LLMs form coherent compositional subspaces (where direction addition preserves meaning), and which are merely approximately linear in isolation?

## Hypothesis Decomposition
1. **H1 (Related concepts share subspace)**: Directions for related concepts within the same semantic category (e.g., country→capital and country→language) will have non-zero causal inner products and moderate composition fidelity.
2. **H2 (Unrelated concepts are orthogonal)**: Directions for unrelated concepts across categories (e.g., morphological vs. encyclopedic) will be approximately orthogonal under the causal inner product.
3. **H3 (Orthogonal directions compose better)**: Cross-category direction pairs (orthogonal) will show higher composition fidelity than within-category pairs (non-orthogonal), because interference is lower.
4. **H4 (Superposition degrades composition)**: Concept directions that are more superposed (higher interference with random directions) will show lower composition fidelity.

## Proposed Methodology

### Approach
We use **google/gemma-2-2b** as our primary model (well-studied, fits on single GPU, used in Park et al. 2024b). We extract concept directions from the unembedding matrix using the counterfactual pairs approach from Park et al. We then test composition by:
1. Extracting individual concept directions d_A, d_B
2. Extracting a "joint" direction d_{A,B} from tokens that satisfy both concepts
3. Measuring how well d_A + d_B aligns with d_{A,B} (Composition Fidelity Score)
4. Measuring interference: does applying d_A as a steering vector affect concept B?

### Experimental Steps
1. Load Gemma-2B, extract unembedding matrix W_U
2. Load BATS 3.0 counterfactual pairs for ~30 concept categories
3. Extract concept directions using mean-difference method (Park et al.)
4. Compute causal inner product matrix across all concept pairs
5. For each pair of concepts: compute CFS and interference scores
6. Categorize results by concept type (morphological, encyclopedic, lexicographic)
7. Statistical analysis: compare within-category vs. cross-category composition
8. Visualize: heatmaps, scatter plots of orthogonality vs. composition fidelity

### Baselines
- **Random directions**: Null baseline for all metrics
- **Euclidean inner product**: Compare against causal inner product
- **Individual direction performance**: Baseline for steering experiments

### Evaluation Metrics
1. **Leave-one-out consistency**: How stable is the extracted direction? (validates quality)
2. **Causal inner product**: |⟨d_A, d_B⟩_C| — measures orthogonality
3. **Composition Fidelity Score (CFS)**: cosine_similarity(d_A + d_B, d_{A,B}) — measures additive composition
4. **Interference Score**: Change in concept-B probing accuracy when d_A is added
5. **Steering Accuracy**: Does adding d_A change the model's output on concept A?

### Statistical Analysis Plan
- Paired t-tests / Wilcoxon signed-rank for within-category vs. cross-category comparisons
- Pearson/Spearman correlation between orthogonality and composition fidelity
- Bootstrap confidence intervals for all aggregate metrics
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1 supported: Within-category concept pairs show moderate CFS (0.3-0.7) and non-zero causal inner products
- H2 supported: Cross-category pairs show near-zero causal inner products (<0.1)
- H3: Cross-category pairs show higher CFS than within-category pairs (less interference)
- H4: Negative correlation between superposition degree and CFS

## Timeline and Milestones
1. Direction extraction + quality validation: 30 min
2. Inner product matrix computation: 20 min
3. Composition experiments: 40 min
4. Steering experiments: 30 min
5. Analysis + visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- Some BATS categories may have too few single-token pairs for Gemma's tokenizer → fall back to categories with sufficient pairs
- Joint concept directions (d_{A,B}) may be hard to define for arbitrary concept pairs → use the intersection of token sets
- Gemma-2B may be too small to show clean linear structure → validate direction quality first

## Success Criteria
- At least 15 concept categories with ≥10 valid single-token pairs each
- Clear separation in CFS between within-category and cross-category pairs
- Statistical significance (p < 0.05) for at least one hypothesis test
- Interpretable visualization showing the compositionality landscape
