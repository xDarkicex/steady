# Research: Text Classification Architecture

## Sources Evaluated (2026-06-17)

| Library | Author | License | Approach | Adopted? |
|---------|--------|---------|----------|----------|
| fastText | Meta | MIT (archived) | Learned subword embeddings + linear classifier | No — archived, GB-scale models |
| Floret | Explosion | MIT | fastText successor, Bloom embeddings (50MB) | No — C++/Python, too heavy for pure Go |
| text (R) | Oscar Kjell | MIT | BERT embeddings + ridge/RF classifier | No — requires Python/HuggingFace runtime |
| Owl | Liang Wang | MIT | TF-IDF + regression suite | Partial — sparse dot product pattern, vocabulary trimming |
| byteSteady | Zhang & Drouin (2021) | Academic | Raw byte n-gram hashing, collision-tolerant | **Yes** — core encoder |
| Conformal Prediction | Vovk et al. | Unpatented math | Prediction sets with guaranteed coverage | **Yes** — noise/OOD rejection |
| Platt Scaling | Platt (1999) | Unpatented math | Logistic calibration of raw scores | **Yes** — replaces hard-coded confidence |
| Hogwild! SGD | Niu, Recht, Ré, Wright (2011) | Academic | Lock-free parallel SGD for sparse objectives | **Yes** — training algorithm |

## Architecture Selection

### Why byteSteady (not fastText/Floret)

fastText and its successor Floret learn word/subword embeddings from scratch. This
requires a vocabulary, tokenizer, and dedicated word-index rows in the embedding
table. For our scale (6 classes, 60K training examples, chat text), the complexity
is unnecessary.

byteSteady operates on raw bytes with no tokenizer, no vocabulary, and no
per-word embeddings. Multi-scale byte n-grams (4, 8, 12, 16 bytes) are FNV-1a
hashed directly into a fixed-size embedding table. Hash collisions are tolerated
(the averaging over hundreds of windows per text smooths them out). This is:

- **Simpler** — no tokenizer, no vocabulary, no UTF-8 boundary logic
- **Smaller** — no per-word embedding rows, just bucket × dim float32
- **Faster** — O(text_bytes × num_window_sizes) with zero allocations
- **Language-agnostic** — raw bytes work for any encoding including CJK and Cyrillic

The key insight from the research phase: at our accuracy requirements and scale,
the embedding table size matters more than the embedding learning algorithm.
Floret's Bloom embedding innovation (multiple hashes per word into a shared table)
is clever but adds complexity without benefit for 6-class classification.

### Why OVA Logistic (not Softmax)

Softmax forces mutual exclusivity (sum of probabilities = 1). A text cannot be
both 0.9 identity and 0.7 constraint. OVA (One-vs-All) logistic regression gives
each label an independent sigmoid probability. A text CAN be both identity and
constraint simultaneously — "I am a Go developer and API keys must never be
committed" genuinely belongs to both classes.

### Why Platt Scaling (not raw sigmoid)

Raw sigmoid outputs are not calibrated probabilities. A score of 0.8 does not mean
"80% chance of being correct." Platt scaling fits `P(calibrated) = 1/(1+exp(-(A×score+B)))`
on a held-out calibration set, mapping raw scores to empirical correctness
frequencies. Two float32s per label, evaluated in nanoseconds.

### Why Conformal Prediction (not threshold-based)

A fixed confidence threshold (e.g., "accept if score > 0.7") is brittle — different
classes have different score distributions. Conformal prediction replaces this with
a single quantile Q computed from calibration non-conformity scores. At inference,
labels with `1 - calibrated_prob ≤ Q` are included. Empty set → noise/OOD.

The Q value is the only tunable parameter and has a clear interpretation: Q=0.08
means the prediction set is guaranteed to contain the true label 92% of the time.

### Why Hogwild! SGD (not serial SGD)

The embedding table update is sparse — each example touches ~200 rows out of 2M.
Collision probability between goroutines is ~0.01%. Lock-free CAS atomic float32
addition gives near-linear speedup on multi-core. The convergence guarantee
(Hogwild! Theorem, Niu et al. 2011) holds for our objective: sparse, bounded
gradients, strongly convex per-coordinate.

## What We Evaluated and Rejected

| Technique | Reason Rejected |
|-----------|----------------|
| BPE tokenizer + linear probe | Adds tokenizer complexity, no accuracy gain over byte n-grams at our scale |
| TF-IDF + L1-Logistic (Owl-style) | Sparse vectors are memory-efficient but OOV handling is poor for chat text |
| BERT embeddings + ridge (text-style) | Requires Python/HuggingFace runtime, violates pure Go constraint |
| Wazero WASM (Floret C++ compiled) | Adds build complexity, WASM overhead erodes latency budget |
| Negative sampling loss | Only needed for >100 labels; we have 6 |
| Hierarchical softmax | Same — for 10K+ labels |
| Product quantization | Model size is already acceptable (488MB mmap'd, OS page cache) |
| Multi-layer perceptron head | Linear classifier is sufficient for bag-of-ngram features |

## Key Design Decisions

1. **Byte-level encoding over word-level** — language-agnostic, no tokenizer, handles typos and slang naturally
2. **Collision-tolerant hashing** — FNV-1a with 2M buckets. Collisions exist but are averaged out by hundreds of windows per text
3. **OVA over softmax** — multi-label semantics (text can be identity + constraint)
4. **Conformal over threshold** — mathematical coverage guarantee, empty set for noise
5. **Mmap over loading** — 488MB table is mmap'd read-only, OS page cache manages residency, GC-invisible
6. **Off-heap over GC** — all inference allocations are `xDarkicex/memory` Pool-backed, Reset() between calls, zero GC pressure

## Results

Trained with bucket=2M, dim=64, epochs=20 on Apple M2 (8 cores):

- Conformal Q: 0.08 (92% coverage guarantee)
- Calibrated confidence on correct label: >95%
- Inference latency: ~260µs per classification
- Zero heap allocations on hot path
- Model file: 488MB (mmap'd, OS page cache)
