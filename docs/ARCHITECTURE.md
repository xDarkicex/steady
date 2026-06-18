# Architecture: steady

## Pipeline

```
Raw text bytes
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ byte_steady.go: Encode()                            │
│                                                     │
│ Multi-scale byte n-gram windows (4, 8, 12, 16)     │
│   ↓ FNV-1a hash → modulo bucket → row lookup        │
│   ↓ Accumulate embedding rows into hidden vector    │
│   ↓ Divide by window count (average)                │
│                                                     │
│ Input:  text []byte, table []float32, bucket, dim   │
│ Output: hidden []float32 (Pool-allocated, dim)      │
│ Latency: ~250µs (2M bucket, 64 dim, 50-word text)   │
│ Allocations: 0 (Pool-backed, Reset between calls)    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ logistic.go: PredictLogits()                        │
│                                                     │
│ OVA logistic regression over hidden vector.         │
│   For each label i:                                 │
│     logit[i] = sigmoid(dot(hidden, weights[i]) + bias[i]) │
│                                                     │
│ 6 labels × 64 dim = 384 multiply-adds + 6 sigmoids  │
│ Latency: ~3µs                                       │
│ Allocations: 0 (output slice pre-allocated)         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ platt.go: ApplyPlatt()                              │
│                                                     │
│ Platt scaling per label:                            │
│   P = 1 / (1 + exp(-(A × logit + B)))               │
│                                                     │
│ Two float32 params per label (A slope, B intercept) │
│ Fit offline on held-out calibration set              │
│ Latency: ~65ns per label                            │
│ Allocations: 0 (pure math)                          │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ conformal.go: PredictSet()                          │
│                                                     │
│ Split conformal prediction:                         │
│   Include label if (1 - calibrated_prob) ≤ Q        │
│   Empty set → noise / out-of-distribution           │
│                                                     │
│ Q computed offline from calibration non-conformity   │
│ scores at desired coverage level (e.g., 95%)         │
│ Latency: ~300ns                                     │
│ Allocations: 0 (Pool-backed output slices)          │
└─────────────────────────────────────────────────────┘
    │
    ▼
PredictionSet {Kinds, Confidences}
```

## Module Map

```
steady/
├── byte_steady.go       — Encoder: byte n-gram hashing + embedding lookup
├── logistic.go          — OVA logistic regression head
├── platt.go             — Platt scaling calibration + inference
├── conformal.go         — Split conformal prediction sets
├── model.go             — Binary artifact loader (mmap)
├── steady.go            — Top-level Classify() API + debug
├── train.go             — Hogwild! SGD training loop
├── cmd/steady/main.go   — CLI entry point
└── docs/
    ├── QUICKSTART.md    — Usage guide + preferred settings
    ├── RESEARCH.md      — Literature review + design rationale
    └── ARCHITECTURE.md  — This document
```

## Data Flow: Training

```
Labeled data (__label__kind text)
    │
    ▼
parseExamples() → []example{{text, label}, ...}
    │
    ▼
initTable() → mmap'd float32[bucket×dim] (writable)
    │  Initialized via XorShift32 PRNG in [-1/dim, +1/dim]
    ▼
trainEmbeddings() → Hogwild! workers
    │
    ├─ hogwildWorker (per goroutine):
    │    for each example:
    │      Encode → hidden
    │      PredictLogits → logits
    │      Backprop correct label + random negative label:
    │        - Update weights[correct] toward hidden
    │        - Update weights[negative] away from hidden
    │        - Update embedding rows via CAS atomic float32
    │        - Center gradient → prevent bilinear drift
    │        - Weight decay 1e-4 → prevent unbounded growth
    │
    ▼
calibratePlatt() → Platt A, B parameters on held-out set
    │
    ▼
ComputeQuantile() → conformal Q
    │
    ▼
writeModel() → model.bin (header + table + weights + bias + plattA + plattB + Q)
```

## Memory Strategy

### Inference (Classify)

| Structure | Allocator | Lifetime |
|-----------|-----------|----------|
| Embedding table (read-only) | `memory.MmapFileReadOnly` | Startup → shutdown |
| Model weights, bias, platt | `memory.Pool` (modelPool) | Startup → shutdown |
| Hidden vector, logits, calibrated | `memory.Pool` (scratchPool) | Per-classification → Reset() |
| Output PredictionSet slices | `memory.Pool` (scratchPool) | Per-classification → Reset() |

Total inference footprint: ~2MB for pools + OS page cache for model file.

### Training (Train)

| Structure | Allocator | Why |
|-----------|-----------|-----|
| Embedding table (writable) | `memory.MmapFile(..., true)` | Shared across goroutines, survives crash |
| OVA weights, bias | `memory.Pool` (shared) | Single-owner, written by Hogwild! CAS |
| Per-worker hidden vector | `memory.Pool` (per goroutine) | No sharing needed |
| Per-worker n-gram hash buffer | `memory.Pool` (per goroutine) | slices.Sort dedup, Reset between examples |
| Calibration scores | `memory.Pool` (shared) | Single-goroutine phase |

## Concurrency Model

### Inference: single-goroutine per Model

Each Model has its own scratchPool. `Classify()` calls `scratchPool.Reset()` at
start, allocates scratch slices, runs pipeline, returns. No locking, no sharing.
Multiple goroutines can each have their own Model (separate scratchPools).

### Training: Hogwild! (lock-free multi-goroutine)

Embedding table is mmap'd writable, shared across goroutines. Updates use
CAS-based `atomicAddFloat32()`. OVA weights and bias also updated via CAS.
Per-worker scratch is isolated (each goroutine has its own Pool).

Sparsity: each example touches ~200 rows out of 2M. Collision probability
between goroutines is ~0.01%. The Hogwild! convergence theorem (Niu et al.
2011) guarantees convergence under these conditions.

## Model Artifact Format (Binary)

```
Offset  Size  Field
0       4     Magic (0x42595445 = "BYTE")
4       4     Version (1)
8       4     Bucket (embedding table rows)
12      4     Dim (embedding dimension)
16      4     MinN (smallest byte n-gram, always 4)
20      4     MaxN (largest byte n-gram, always 16)
24      4     NumLabels
28      4     Reserved (0)
32      B*D*4 Embedding table (float32, row-major)
...     L*D*4 OVA weights (float32, row-major)
...     L*4   OVA bias (float32)
...     L*4   Platt A (float32)
...     L*4   Platt B (float32)
...     4     Conformal quantile Q (float32)

B = Bucket, D = Dim, L = NumLabels
```

The embedding table is loaded via `memory.MmapFileReadOnly` — zero-copy, OS
page cache, GC-invisible. All other fields are small (<2KB total) and loaded
into Pool-backed slices.
