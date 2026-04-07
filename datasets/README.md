# Datasets for "Which Linear Directions Are Compositional?"

This directory contains datasets for studying compositional linear directions in transformer LLMs.

## 1. BATS 3.0 (Big Analogy Test Set)

**Location:** `bats3.0/BATS_3.0/`

**Source:** Gladkova et al. 2016; used by Park et al. 2023 for defining concepts via counterfactual pairs.

**Download:**
```bash
# Already included. To re-download:
wget https://vecto-data.s3-us-west-1.amazonaws.com/BATS_3.0.zip
unzip BATS_3.0.zip -d bats3.0/
```

**Format:** Plain text, one word pair per line. Multiple valid answers separated by `/`.
```
actor    actress
boy      girl
king     queen
```

**Structure:** 4 categories x 10 relations = 40 relation types, ~50 pairs each:
- `1_Inflectional_morphology/` (I01-I10): plural, comparative, verb forms
- `2_Derivational_morphology/` (D01-D10): un+adj, verb+tion, etc.
- `3_Encyclopedic_semantics/` (E01-E10): country-capital, male-female, animal-young
- `4_Lexicographic_semantics/` (L01-L10): hypernyms, synonyms, antonyms, meronyms

**Key files for compositional direction research:**
- `E10 [male - female].txt` - Gender concept pairs
- `E01 [country - capital].txt` - Geographic concept pairs
- `E02 [country - language].txt` - Language concept pairs
- `L09 [antonyms - gradable].txt`, `L10 [antonyms - binary].txt` - Antonym pairs

**Loading code:**
```python
from pathlib import Path

def load_bats_relation(filepath):
    """Load a BATS relation file as list of (word1, [word2_options]) tuples."""
    pairs = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word1 = parts[0]
                word2_options = parts[1].split('/')
                pairs.append((word1, word2_options))
    return pairs

# Example: load all encyclopedic semantics
bats_dir = Path("datasets/bats3.0/BATS_3.0")
for fpath in sorted((bats_dir / "3_Encyclopedic_semantics").glob("*.txt")):
    pairs = load_bats_relation(fpath)
    print(f"{fpath.name}: {len(pairs)} pairs")
```

---

## 2. WordNet (via NLTK)

**Location:** `wordnet/` (sample files); raw NLTK data in `wordnet/nltk_data/` (gitignored)

**Source:** Miller 1995; used by Park et al. 2024 for hierarchical concept taxonomies (900+ concepts).

**Download:**
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Pre-generated samples:**
- `wordnet/sample_taxonomies.json` - Hierarchical subtrees for animal, plant, vehicle, food, color (depth=3)
- `wordnet/hierarchical_pairs_sample.json` - Parent-child concept pairs from the top of the hierarchy

**Stats:** 117,659 total synsets; 82,115 noun synsets.

**Loading code:**
```python
from nltk.corpus import wordnet as wn

# Get a concept hierarchy (e.g., animals)
animal = wn.synset('animal.n.01')
print(f"Definition: {animal.definition()}")
print(f"Hyponyms: {[h.name() for h in animal.hyponyms()]}")
print(f"Hypernyms: {[h.name() for h in animal.hypernyms()]}")

# Get all hypernym paths (for hierarchy depth analysis)
dog = wn.synset('dog.n.01')
for path in dog.hypernym_paths():
    print(" -> ".join(s.name() for s in path))

# Compute semantic similarity
print(wn.path_similarity(wn.synset('dog.n.01'), wn.synset('cat.n.01')))

# Build concept pairs for a taxonomy branch (Park et al. 2024 style)
def get_taxonomy_pairs(root_synset, max_depth=4):
    """Extract (parent, child) pairs from a WordNet subtree."""
    pairs = []
    queue = [(root_synset, 0)]
    while queue:
        synset, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for hypo in synset.hyponyms():
            pairs.append({
                "parent": synset.lemmas()[0].name(),
                "child": hypo.lemmas()[0].name(),
                "parent_synset": synset.name(),
                "child_synset": hypo.name(),
                "depth": depth + 1,
            })
            queue.append((hypo, depth + 1))
    return pairs
```

---

## 3. Function Vectors Task Data (Todd et al. 2024)

**Location:** `function_vectors/`

**Source:** Todd et al. 2024. "Function Vectors in Large Language Models." GitHub: https://github.com/ericwtodd/function_vectors

**Download:**
```bash
# Clone the full dataset:
git clone https://github.com/ericwtodd/function_vectors.git /tmp/fv
cp -r /tmp/fv/dataset_files/abstractive function_vectors/
cp -r /tmp/fv/dataset_files/extractive function_vectors/
```

**Format:** JSON arrays of `{"input": "...", "output": "..."}` objects.
```json
[
  {"input": "flawed", "output": "perfect"},
  {"input": "orthodox", "output": "unorthodox"}
]
```

**Available tasks (57 total):**

Abstractive (29 tasks) -- require knowledge beyond the prompt:
- Language: `antonym`, `synonym`, `english-french`, `english-german`, `english-spanish`
- Knowledge: `country-capital`, `country-currency`, `landmark-country`, `park-country`, `national_parks`
- People: `person-instrument`, `person-occupation`, `person-sport`
- Grammar: `singular-plural`, `present-past`
- String: `capitalize`, `capitalize_first_letter`, `capitalize_last_letter`, `capitalize_second_letter`, `lowercase_first_letter`, `lowercase_last_letter`
- Classification: `sentiment`, `ag_news`, `commonsense_qa`, `word_length`
- Sequence: `next_item`, `prev_item`, `next_capital_letter`, `product-company`

Extractive (28 tasks) -- answer extractable from the prompt:
- Classification: `adjective_v_verb`, `animal_v_object`, `color_v_animal`, `concept_v_object`, `fruit_v_animal`, `object_v_concept`, `verb_v_adjective` (each with 3 and 5 variants)
- Selection: `choose_first_of`, `choose_last_of`, `choose_middle_of`, `alphabetically_first`, `alphabetically_last` (each with 3 and 5 variants)
- NER: `conll2003_location`, `conll2003_organization`, `conll2003_person`
- QA: `squad_val`

**Samples included in repo (not gitignored):**
- `abstractive/antonym.json` (2,398 pairs)
- `abstractive/english-french.json` (4,698 pairs)
- `abstractive/country-capital.json` (197 pairs)
- `extractive/color_v_animal_3.json` (1,000 examples)

**Loading code:**
```python
import json

def load_fv_task(task_path):
    """Load a Function Vectors task file."""
    with open(task_path) as f:
        data = json.load(f)
    return data  # List of {"input": str, "output": str}

# Build ICL prompts (as in Todd et al.)
def make_icl_prompt(data, n_shots=5, query_idx=None):
    """Create an in-context learning prompt from task data."""
    import random
    if query_idx is None:
        query_idx = random.randint(n_shots, len(data) - 1)
    shots = data[:n_shots]
    query = data[query_idx]
    prompt = ""
    for ex in shots:
        prompt += f"Q: {ex['input']}\nA: {ex['output']}\n\n"
    prompt += f"Q: {query['input']}\nA:"
    return prompt, query['output']
```

---

## 4. Google Analogy Test Set (Mikolov et al. 2013)

**Location:** `google_analogy/questions-words.txt`

**Source:** Mikolov et al. 2013. "Efficient Estimation of Word Representations in Vector Space."

**Format:** Category headers prefixed with `:`, then lines of 4 words (A is to B as C is to D):
```
: capital-common-countries
Athens Greece Baghdad Iraq
Athens Greece Bangkok Thailand
```

**Categories (14 total, ~19,558 questions):**

Semantic (5):
- `capital-common-countries` - Common country capitals
- `capital-world` - World capitals
- `currency` - Country-currency pairs
- `city-in-state` - US city-state pairs
- `family` - Family relationships (brother/sister, king/queen)

Syntactic (9):
- `gram1-adjective-to-adverb`
- `gram2-opposite`
- `gram3-comparative`
- `gram4-superlative`
- `gram5-present-participle`
- `gram6-nationality-adjective`
- `gram7-past-tense`
- `gram8-plural`
- `gram9-plural-verbs`

**Loading code:**
```python
def load_google_analogies(filepath):
    """Load Google analogy test set, returns dict of category -> list of (a,b,c,d) tuples."""
    categories = {}
    current = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                current = line[2:]
                categories[current] = []
            elif line and current:
                words = line.split()
                if len(words) == 4:
                    categories[current].append(tuple(words))
    return categories

# Usage
cats = load_google_analogies("datasets/google_analogy/questions-words.txt")
for name, pairs in cats.items():
    print(f"{name}: {len(pairs)} analogies")
```

---

## Relevance to Research Questions

| Dataset | Research Use | Key Paper |
|---------|-------------|-----------|
| BATS 3.0 | Counterfactual word pairs define concept directions (e.g., male-female direction from word pair differences) | Park et al. 2023 |
| WordNet | Hierarchical concept taxonomy; test whether linear directions respect IS-A hierarchy | Park et al. 2024 |
| Function Vectors | ICL tasks define functional directions; test compositionality of task vectors | Todd et al. 2024 |
| Google Analogy | Classic analogy benchmark; tests whether vector arithmetic captures relational composition | Mikolov et al. 2013 |

### Compositionality experiments these datasets enable:
1. **Direction addition:** Does `male-female` + `English-French` compose meaningfully? (BATS, Function Vectors)
2. **Hierarchical composition:** Do directions at different taxonomy levels compose? (WordNet)
3. **Task composition:** Can function vectors for related tasks be combined? (Function Vectors)
4. **Analogy as composition:** Is `king - man + woman = queen` a form of direction composition? (Google Analogy, BATS)
