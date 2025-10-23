# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.1] - UNRELEASED

- Basic tooling for memory profiling (see `memprof` feature flag).


## [1.0.0] - 2025-06-22

### Fixed 

- Pin the `rc.9` version of `ort-sys` (`ort` was not doing so before `rc.10`, which would break builds of crates that depend on earlier versions, see [pykeio/ort#399](https://github.com/pykeio/ort/issues/399)).


## [0.9.4] - 2025-03-30

### Fixed

- Fix words mask when the first non-label word is encoded as multiple tokens (PR [#6](https://github.com/fbilhaut/gline-rs/pull/6)).

### Added

- Add the ability to load a model from memory ([PR#9](https://github.com/fbilhaut/gline-rs/pull/9)).
- Basic github CI workflow (PR [#7](https://github.com/fbilhaut/gline-rs/pull/7)).
- The token and span pipelines now expose the expected input/output tensors so `orp` can check if they are consistent with the actual model.

### Changed

- Switch to `orp` version `0.9.2`.
- Move the `examples` directory from `src` to `.` (standard location Cargo-wise).


## [0.9.3] - 2025-03-08

Two important fixes in this release (see "fixed" below). It also comes with very minor changes in the public API:
* Some imports might need to be adapted due to externalization of the pipeline system in another crate.
* The (optional) `dup_label` flag has been added to the parameters to allow/disallow overlapping spans with the same label.

### Fixed

- Fix word mask on words with sub-tokens (PR [#3](https://github.com/fbilhaut/gline-rs/pull/3)) which prevented some (multi-token) entities to be recognized.
- Fix the handling of overlapping spans (the `flat_ner` and associated flags were not correctly honored). This fix solves the problem described in PR [#4](https://github.com/fbilhaut/gline-rs/pull/4).

### Changed

- The generic pipeline framework has been externalized, see the [`composable`](https://github.com/fbilhaut/composable) and [`orp`](https://github.com/fbilhaut/orp) crates. The public API is left unchanged, beside minor import adaptations (see examples).


## [0.9.2] - 2025-01-26

### Fixed

- Fixed issue with **multi-word labels** (issue [#1](https://github.com/fbilhaut/gline-rs/issues/1)).

### Added

- Pipeline for **Relation Extraction**, with related example.
- Matrix-level documentation of pre- and post-processing steps (see `doc/Processing.typ` or `doc/Processing.pdf`).
- More unit-tests.

### Changed

- The `Pipeline` trait is now fully generic wrt. input, output and context types.
- The `Model` struct is more opaque and parametrized by a pipeline which it handles by itself.
- The `GLiNER` struct is now a light convenience wrapper around `Model`, `Pipeline` and `Parameters`.
- The `Composable` trait is now implemented for `Model`+`Pipeline`+`Parameters` combos, to facilitate re-use and combination of pipelines. See `Pipeline::to_composable()` or `Model::to_composable()`.
- Drop `num_traits` dependency (in favor of `ndarray`'s `NdFloat`).


## [0.9.1] - 2025-01-13

### Added

- GPU support: public API (`RuntimeParameters`), new example (`benchmark-gpu`), cargo features (mirroring `ort` backends), and documentation.

### Changed

- Minor source code linting and documentation refactoring.


## [0.9.0] - 2025-01-07

First public release.