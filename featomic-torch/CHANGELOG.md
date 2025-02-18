# Changelog

All notable changes to featomic are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/featomic/)

<!-- Possible sections for each package:

### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.6.1](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.6.1) - 2025-02-18\

## Added

- Add Support for Python 3.13 and PyTorch 2.6


## [Version 0.6.0](https://github.com/metatensor/featomic/releases/tag/featomic-torch-v0.6.0) - 2025-01-07

### Added

- C++ and Python TorchScript bindings to `featomic`, making all calculators
  accessible from TorchScript models.

- Integration of the TorchScript calculators with
  [metatensor-torch-atomistic](https://docs.metatensor.org/latest/atomistic/index.html),
  using the `System` class from this package as a system provider, and
  integrating with neighbor lists provided by the simulation engine through
  metatensor.

- Automatic integration with (Py)Torch automatic differentiation system. If any
  of the inputs requires gradients, then `featomic-torch` will compute them,
  store them, and integrate them with a custom backward function on the
  calculator output.

- Re-export of Python tools for Clebsch-Gordan tensor products from `featomic`,
  in a TorchScript-compatible way.
