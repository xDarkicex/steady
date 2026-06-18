# steady — Zero-Allocation Text Classification in Go

[![Go Reference](https://pkg.go.dev/badge/github.com/xDarkicex/steady.svg)](https://pkg.go.dev/github.com/xDarkicex/steady)
[![Go Report Card](https://goreportcard.com/badge/github.com/xDarkicex/steady)](https://goreportcard.com/report/github.com/xDarkicex/steady)
[![Coverage](https://img.shields.io/badge/coverage-88%25-green.svg)](https://github.com/xDarkicex/steady/actions)
[![Vulnerabilities](https://img.shields.io/badge/vuln-go%20vulncheck-blue.svg)](https://pkg.go.dev/golang.org/x/vuln)
[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pure Go text classification. Raw bytes in, calibrated prediction sets out.
Sub-millisecond, zero GC pressure, all memory backed by off-heap allocators.

## Why

Text classification in Go typically requires either CGo bindings to C++ libraries
or HTTP calls to Python model servers. Neither is acceptable for latency-sensitive,
deterministic, single-binary deployments. steady provides a self-contained engine:
train a model offline, mmap it at startup, classify with zero heap allocations.

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [When to Use](#when-to-use)
- [When Not to Use](#when-not-to-use)
- [How It Works](#how-it-works)
- [API](#api)
- [Training](#training)
- [Performance](#performance)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Research](#research)
- [License](#license)

## Install

```
go get github.com/xDarkicex/steady
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/xDarkicex/steady"
)

func main() {
    m, _ := steady.Load("model.bin")
    defer m.Close()
    m.SetLabelNames([]string{"identity", "constraint", "decision", "fact", "preference", "episode"})

    result := m.Classify("I am a Go developer working in Berlin")
    if result.IsEmpty() {
        fmt.Println("noise / out-of-distribution")
    } else {
        for i, k := range result.Kinds {
            fmt.Printf("%s: %.2f\n", k, result.Confidences[i])
        }
    }
}
```

## When to Use

- You need sub-millisecond text classification in a pure Go binary
- You want calibrated confidence scores, not just a label
- You need noise/out-of-distribution rejection (empty prediction set)
- Your text spans multiple languages (byte-level encoding is language-agnostic)
- You want deterministic inference (same input → same output, every time)
- You can train offline and ship a model artifact

## When Not to Use

- You need state-of-the-art accuracy on long-form document classification (use a transformer)
- You can't train a model offline (steady does not do online learning)
- You need per-token or sequence-labeling output (steady classifies whole texts)
- Your label space is >1000 classes (OVA logistic regression is O(L×D))

## How It Works

```
Raw text bytes
    │
    ▼
byteSteady encoder — multi-scale byte n-grams (4,8,12,16) hashed via FNV-1a
into a fixed embedding table. Rows are averaged into a dim-dimensional hidden
vector. No tokenizer. No vocabulary. Works on any UTF-8 text.
    │
    ▼
OVA logistic head — one independent sigmoid per label over the hidden vector.
A text can be both identity AND constraint simultaneously (no forced mutual
exclusion like softmax).
    │
    ▼
Platt scaling — raw sigmoid outputs are calibrated via logistic regression
(A×score + B) fit on a held-out set. Two float32s per label.
    │
    ▼
Conformal prediction — labels with non-conformity ≤ Q are included in the
prediction set. Q is a single float32 computed from calibration errors.
Empty set = noise / out-of-distribution.
```

## API

```go
// Load a trained model artifact (mmap'd, zero-copy).
func Load(path string) (*Model, error)

// SetLabelNames sets the human-readable label names used in prediction sets.
func (m *Model) SetLabelNames(names []string)

// Classify runs the full pipeline and returns a calibrated prediction set.
// Empty set = noise or out-of-distribution input.
func (m *Model) Classify(text string) PredictionSet

// ClassifyDebug returns raw logits, calibrated probabilities, and Platt
// parameters for debugging model behavior.
func (m *Model) ClassifyDebug(text string) DebugResult

// Close releases mmap'd memory and off-heap pools.
func (m *Model) Close() error
```

## Training

```
go run ./cmd/steady -input data.txt -output model.bin -epochs 20 -bucket 2000000 -dim 64
```

Input format: `__label__classname Text here\n`

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for preferred settings and tuning guide.

## Performance

| Operation | Latency | Allocations |
|-----------|---------|-------------|
| Encode (50-word text, 2M bucket, dim=64) | ~250µs | 0 B, 0 allocs |
| Full Classify pipeline | ~260µs | 0 B, 0 allocs |

Apple M2, Go 1.25, bucket=2M, dim=64, 6 labels.

## Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) — Usage guide, preferred settings, tuning
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Pipeline diagrams, module map, data flow
- [RESEARCH.md](docs/RESEARCH.md) — Literature review, design rationale, rejected alternatives

## Dependencies

### xDarkicex/memory

Off-heap allocators providing GC-isolated, lock-free memory management.
The embedding table is mmap'd read-only via `MmapFileReadOnly`. Per-classification
scratch buffers are backed by Pool with bulk `Reset()` between calls. Training
uses writable mmap and ShardedFreeList for concurrent gradient accumulation.

https://github.com/xDarkicex/memory

## Research

steady builds on published algorithms and open-source reference implementations.
Key influences:

| Algorithm | Source | Role in steady |
|-----------|--------|----------------|
| byteSteady | Zhang & Drouin (2021) | Byte-level n-gram hashing encoder |
| OVA Logistic Regression | Bishop (2006), standard ML | Multi-label classification head |
| Platt Scaling | Platt (1999), *Advances in Large Margin Classifiers* | Confidence calibration |
| Conformal Prediction | Vovk, Gammerman, Shafer (2005) | Prediction sets with coverage guarantee |
| Hogwild! SGD | Niu, Recht, Ré, Wright (2011), arXiv:1106.5730 | Lock-free parallel training |
| fastText | Joulin et al. (2016), Meta | Architectural inspiration (subword embeddings + linear classifier) |
| Owl | Wang (2016–2022), MIT | TF-IDF pipeline, sparse vector operations |
| SetFit | Tunstall et al. (2022), HuggingFace | Few-shot contrastive learning methodology |

See [docs/RESEARCH.md](docs/RESEARCH.md) for the full literature review and
design rationale.

## License

MIT © 2026 xDarkicex
