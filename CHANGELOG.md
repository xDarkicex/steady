# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tests for truncated and corrupt model file loading.

## [0.1.2] — 2026-06-18

### Fixed

- `Load()` now returns an error on truncated or corrupt model files instead
  of silently loading bad weights.
- Temp training mmap files are cleaned up immediately after creation.

## [0.1.1] — 2026-06-18

### Changed

- Run `gofmt` across all source files.

## [0.1.0] — 2026-06-18

### Added

- Initial release.
- byteSteady encoder: multi-scale byte n-gram hashing (4, 8, 12, 16) into a fixed
  embedding table with FNV-1a hashing and zero-allocation inference.
- OVA logistic regression head: independent sigmoid per label with Pool-backed
  hidden vector and logits.
- Platt scaling: confidence calibration via logistic regression on held-out set.
- Conformal prediction: split conformal prediction sets with coverage guarantee
  and noise/out-of-distribution rejection via empty set.
- Model artifact format: single flat binary file with 32-byte header, mmap'd
  embedding table, and small Pool-backed parameter arrays.
- Hogwild! lock-free parallel SGD training with gradient centering, weight decay,
  and negative sampling.
- Training CLI via `cmd/steady` with configurable bucket, dim, epochs, and lr.
- Off-heap memory model: all allocations backed by `xDarkicex/memory` (Pool,
  mmap, ShardedFreeList). Zero GC pressure on inference hot path.
- Documentation: README, QUICKSTART, ARCHITECTURE, RESEARCH.
- 87.8% test coverage, race-clean, staticcheck-clean.

[0.1.2]: https://github.com/xDarkicex/steady/releases/tag/v0.1.2
[0.1.1]: https://github.com/xDarkicex/steady/releases/tag/v0.1.1
[0.1.0]: https://github.com/xDarkicex/steady/releases/tag/v0.1.0
