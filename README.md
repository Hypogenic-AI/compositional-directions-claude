# Which Linear Directions Are Compositional?

Systematic analysis of which linear directions in Transformer LLM residual streams form compositional subspaces, tested on Pythia-2.8B using 25 concept categories from BATS 3.0.

## Key Findings

- **Morphological transformations** (verb tenses, plurals, affixes) form the strongest linear directions (LOO consistency 0.3–0.6) and compose reliably via vector addition
- **Encyclopedic associations** (color, gender, animal properties) are moderately linear (LOO 0.2–0.5)
- **Lexicographic relations** (synonyms, antonyms, meronyms) mostly fail to form coherent linear directions at all — 7 of 10 categories had LOO < 0.1
- Among well-formed directions, **functional composition is remarkably robust**: steering with d_A + d_B achieves >96% accuracy across all tested pair types
- The compositionality boundary is not between pairs of directions — it's between concepts that have coherent linear representations and those that don't

## Project Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and methodology
├── src/
│   ├── extract_directions.py  # Direction extraction + composition metrics
│   ├── steering_experiment.py # Functional steering composition tests
│   ├── analyze_results.py     # Statistical analysis + visualizations
│   └── refined_analysis.py    # Taxonomy + comprehensive figure
├── results/
│   ├── direction_results.json # All direction extraction results
│   ├── steering_results.json  # Steering experiment results
│   ├── analysis_results.json  # Statistical test results
│   └── refined_analysis.json  # Taxonomy and refined stats
├── figures/
│   ├── comprehensive_panel.png   # 6-panel summary figure
│   ├── causal_ip_heatmap.png     # Block-diagonal structure
│   ├── cfs_heatmap.png           # Composition fidelity
│   ├── direction_quality.png     # LOO consistency by category
│   ├── orthogonality_vs_cfs.png  # Scatter: cosine vs CFS
│   └── ...
├── datasets/                  # BATS 3.0, WordNet, etc.
├── papers/                    # 19 reference papers
└── code/                      # Baseline code repositories
```

## Reproducing Results

```bash
uv venv && source .venv/bin/activate
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv add transformers accelerate numpy scipy scikit-learn matplotlib seaborn tqdm nltk

python src/extract_directions.py    # ~2 min, downloads Pythia-2.8B
python src/steering_experiment.py   # ~2 min
python src/analyze_results.py       # ~10 sec
python src/refined_analysis.py      # ~10 sec
```

Requires: CUDA GPU (tested on RTX A6000), Python 3.10+, ~6GB disk for model weights.

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, statistical tests, and discussion.
