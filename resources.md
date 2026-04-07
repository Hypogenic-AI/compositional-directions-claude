# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Which Linear Directions Are Compositional?" including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 19

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| The Linear Representation Hypothesis | Park, Choe, Veitch | 2024 | papers/park2023_linear_representation_hypothesis.pdf | Formalizes LRH, causal inner product |
| Geometry of Categorical and Hierarchical Concepts | Park, Choe, Jiang, Veitch | 2025 | papers/park2024_geometry_categorical_hierarchical_concepts.pdf | Polytopes, hierarchical orthogonality |
| Function Vectors in LLMs | Todd et al. | 2024 | papers/todd2023_function_vectors.pdf | FV extraction, composition experiments |
| Causality ≠ Invariance | Opiełka et al. | 2026 | papers/opielka2026_causality_invariance_concept_vectors.pdf | FVs vs CVs, format invariance |
| Toy Models of Superposition | Elhage et al. | 2022 | papers/elhage2022_toy_models_superposition.pdf | Superposition theory |
| Representation Engineering | Zou et al. | 2023 | papers/zou2023_representation_engineering.pdf | RepE framework |
| Contrastive Activation Addition | Panickssery et al. | 2023 | papers/panickssery2023_contrastive_activation_addition.pdf | CAA steering vectors |
| Sparse Feature Circuits | Marks et al. | 2024 | papers/marks2024_sparse_feature_circuits.pdf | SAE-based feature circuits |
| Superposition as Lossy Compression | Bereska et al. | 2025 | papers/bereska2025_superposition_lossy_compression.pdf | Information-theoretic superposition |
| Mathematical Models of Computation in Superposition | Hänni et al. | 2024 | papers/hanni2024_mathematical_models_computation_superposition.pdf | Computational superposition |
| Belief State Geometry in Residual Stream | Shai et al. | 2024 | papers/shai2024_belief_state_geometry_residual_stream.pdf | Belief state geometry |
| PaCE: Parsimonious Concept Engineering | Luo et al. | 2024 | papers/luo2024_pace_concept_engineering.pdf | Sparse concept engineering |
| Activation Steering Broad Skills | van der Weij et al. | 2024 | papers/vanderweij2024_activation_steering_broad_skills.pdf | Steering for broad skills |
| Unforgettable Generalization | Zhang et al. | 2024 | papers/zhang2024_unforgettable_generalization.pdf | Persistent task representations |
| Summing Up the Facts | Chughtai et al. | 2024 | papers/chughtai2024_summing_up_facts.pdf | Additive factual recall |
| Towards Monosemanticity (SAEs) | Bricken et al. | 2023 | papers/bricken2023_monosemanticity_sparse_autoencoders.pdf | Sparse autoencoders |
| Emergent Linear Representations | Nanda et al. | 2023 | papers/nanda2023_emergent_linear_representations.pdf | Linear reps in world models |
| Residual Stream Duality | Zhang | 2026 | papers/zhang2026_residual_stream_duality.pdf | Two-axis transformer view |
| Hierarchical Semantics in SAEs | Muchane et al. | 2025 | papers/muchane2025_hierarchical_semantics_sae.pdf | Hierarchical SAE architecture |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| BATS 3.0 | Gladkova et al. 2016 | ~50 pairs × 40 categories | Analogy / counterfactual pairs | datasets/bats3.0/ | Used by Park et al. 2024 |
| WordNet (via NLTK) | Princeton | 117K+ synsets | Hierarchical concept taxonomy | datasets/wordnet/ | Used by Park et al. 2025 |
| Function Vectors Tasks | Todd et al. 2024 | 40+ tasks, ~200-5000 pairs each | ICL task composition | datasets/function_vectors/ | 29 abstractive + 28 extractive tasks |
| Google Analogy | Mikolov et al. 2013 | ~19K questions | Word analogy benchmark | datasets/google_analogy/ | Classic analogy test set |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| linear_rep_geometry | github.com/KihoPark/linear_rep_geometry | Causal inner product, concept directions | code/linear_rep_geometry/ | LLaMA-2 experiments |
| LLM_Categorical_Hierarchical_Representations | github.com/KihoPark/LLM_Categorical_Hierarchical_Representations | Hierarchical concept experiments | code/hierarchical_representations/ | Gemma-2B, LLaMA-3 experiments |
| function_vectors | github.com/ericwtodd/function_vectors | Function vector extraction + composition | code/function_vectors/ | GPT-J, LLaMA-2 experiments |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Paper-finder service for initial relevance-ranked results
2. arXiv API searches across 10+ query formulations targeting the intersection of: linear representations, compositionality, superposition, steering vectors, and mechanistic interpretability
3. Targeted searches for specific foundational papers (Park, Todd, Elhage, Zou, etc.)

### Selection Criteria
- Direct relevance to the research question of compositional linear directions
- Papers that provide theoretical frameworks (Park et al.), empirical evidence (Todd et al.), or important context (Elhage et al.)
- Preference for recent work (2023-2026) with established benchmarks

### Challenges Encountered
- One arxiv ID (2305.18703) resolved to wrong paper — corrected with proper Nanda 2023 paper
- Paper-finder service had parsing issues on some queries; fell back to arXiv API successfully

### Gaps and Workarounds
- No single dataset designed specifically to test compositionality of linear directions → combine BATS 3.0 (counterfactual pairs), WordNet (hierarchical structure), and FV tasks (composition tests)
- No pre-existing "compositionality benchmark" for representation directions → our experiment will create one

## Recommendations for Experiment Design

### Primary Datasets
1. **BATS 3.0** — extract concept directions via counterfactual pairs, test compositionality within and across semantic categories
2. **WordNet hierarchy** — test hierarchical orthogonality (the strongest form of compositional structure)
3. **Function Vectors task data** — test FV composition using the parallelogram method from Todd et al.

### Baseline Methods
1. **Euclidean vs. causal inner product** — measure how much orthogonality/composition improves under the right metric
2. **Random directions** — null baseline for all composition tests
3. **PCA directions** — compare with causally-derived directions

### Evaluation Metrics
1. **Orthogonality under causal inner product** — |⟨γ̄_W, γ̄_Z⟩_C| for related vs. unrelated concept pairs
2. **Steering accuracy** — does adding direction A change concept A without affecting concept B?
3. **Composition fidelity** — does v_A + v_B work as well as v_{A∩B} extracted directly?
4. **Interference measure** — quantify how much adding direction A affects unrelated concepts

### Code to Adapt/Reuse
1. `code/linear_rep_geometry/` — causal inner product estimation, concept direction extraction, probing/intervention code
2. `code/hierarchical_representations/` — WordNet taxonomy extraction, hierarchical orthogonality tests
3. `code/function_vectors/` — FV extraction via causal mediation, FV composition experiments

### Recommended Models
- **Gemma-2B** (compact, well-studied in Park et al. 2025)
- **LLaMA-3-8B** (good balance of capability and compute)
- Optional: LLaMA-2-7B for comparison with Park et al. 2024
