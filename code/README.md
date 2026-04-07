# Reference Code Repositories

External codebases cloned for reference in this project. Each subdirectory is
an independent Git repository.

---

## 1. `linear_rep_geometry/`

**Paper:** Park et al. 2023 — "The Linear Representation Hypothesis and the
Geometry of Large Language Models" ([arXiv:2311.03658](https://arxiv.org/abs/2311.03658))

**Source:** https://github.com/KihoPark/linear_rep_geometry

Formalises the linear representation hypothesis and defines a *causal inner
product* that respects linguistic semantic structure. Experiments use LLaMA-2.

### Key files
| File | Purpose |
|---|---|
| `store_matrices.py` | Pre-compute weight matrices (run first) |
| `linear_rep_geometry.py` | Core library (causal inner product, projections) |
| `1_subspace.ipynb` | Counterfactual-pair projection onto concept directions |
| `2_heatmap.ipynb` | Orthogonality of unembedding representations |
| `3_measurement.ipynb` | Concept direction as a linear probe |
| `4_intervention.ipynb` | Targeted concept editing via embeddings |
| `5_sanity_check.ipynb` | Verifies uncorrelatedness (Assumption 3.3) |
| `word_pairs/` | Counterfactual word pairs per concept |
| `paired_contexts/` | Wikipedia context samples for measurement |

### Dependencies
`transformers`, `torch`, `numpy`, `seaborn`, `matplotlib`, `tqdm`. Requires GPU.

---

## 2. `hierarchical_representations/`

**Paper:** Park et al. 2024 — "LLM Categorical and Hierarchical
Representations" ([arXiv:2406.01506](https://arxiv.org/abs/2406.01506))

**Source:** https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations

Shows that LLMs represent categorical concepts as polytopes and hierarchical
relations as orthogonality. Experiments use Gemma-2B and LLaMA-3-8B.

### Key files
| File | Purpose |
|---|---|
| `01_eval_noun.ipynb` | WordNet noun hierarchy analysis (Figures 3-5, 10-11) |
| `02_eval_verb.ipynb` | WordNet verb hierarchy analysis |
| `03_eval_noun_llama.ipynb` | Noun hierarchy with LLaMA-3 |
| `04_intervention.ipynb` | Validates representation definitions (Table 1) |
| `05_visualization.ipynb` | 2D and 3D polytope plots |
| `06_subgraph.ipynb` | Zoomed-in tree and heatmaps |
| `07_eval_gamma.ipynb` | Euclidean vs causal inner product comparison |
| `get_wordnet_hypernym_*.ipynb` | WordNet data extraction for each model |
| `data/` | Animal/plant word sets (ChatGPT-4 generated) |
| `hierarchical/` | Hierarchical concept data |
| `requirements.txt` | Python dependencies |

### Dependencies
`torch`, `numpy`, `transformers`, `accelerate`, `bitsandbytes`,
`huggingface-hub`, `matplotlib`, `inflect`, `seaborn`, `networkx`,
`scikit-learn`, `python-dotenv`, `nltk`. See `requirements.txt`.

---

## 3. `function_vectors/`

**Paper:** Todd et al. 2024 — "Function Vectors in Large Language Models"
(ICLR 2024, [arXiv:2310.15213](https://arxiv.org/abs/2310.15213))

**Source:** https://github.com/ericwtodd/function_vectors

**Project site:** https://functions.baulab.info

Extracts *function vectors* from in-context learning demonstrations and shows
they can be added to model activations to induce specific functional behaviour.

### Key files
| File | Purpose |
|---|---|
| `notebooks/fv_demo.ipynb` | Demo: create and apply a function vector |
| `src/evaluate_function_vector.py` | Main FV evaluation pipeline |
| `src/compute_indirect_effect.py` | Indirect-effect computation |
| `src/compute_average_activations.py` | Average activation extraction |
| `src/utils/` | Utility modules (see below) |
| `src/eval_scripts/` | Sample evaluation script wrappers |
| `dataset_files/` | Task datasets |
| `fv_environment.yml` | Conda environment specification |

#### Utility modules (`src/utils/`)
- `eval_utils.py` — evaluate function vectors in various contexts
- `extract_utils.py` — extract function vectors and model activations
- `intervention_utils.py` — intervene with FVs during inference
- `model_utils.py` — load models and tokenizers from HuggingFace
- `prompt_utils.py` — data loading and prompt construction

### Dependencies
Python 3.10, PyTorch 1.13, `transformers`, `datasets`, `baukit`
(github.com/davidbau/baukit), `bitsandbytes`, `accelerate`, `scikit-learn`,
`pandas`, `plotly`, `seaborn`, `matplotlib`, `sentencepiece`, `tqdm`.
Full spec in `fv_environment.yml`.
