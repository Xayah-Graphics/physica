![Physica](https://github.com/Xayah-Graphics/imagebed/blob/570306e838efa21581a5c06a10cd9ff753f495ad/physica-banner.png)

# Physica

[![Windows Build](https://github.com/Xayah-Graphics/physica/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/physica/actions/workflows/windows-build.yml)
[![Arch Build](https://github.com/Xayah-Graphics/physica/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/physica/actions/workflows/arch-build.yml)
[![Docker](https://github.com/Xayah-Graphics/physica/actions/workflows/docker.yml/badge.svg)](https://github.com/Xayah-Graphics/physica/actions/workflows/docker.yml)
[![License](https://img.shields.io/github/license/Xayah-Graphics/physica)](LICENSE)

Physica is a C++23 and CUDA computer graphics research workspace for modern implementations of classic reconstruction,
simulation, and differentiable graphics algorithms. Core algorithms are organized as reusable C++ modules, while
standalone examples can expose live visualization providers to [Spectra](https://github.com/Xayah-Graphics/spectra).

## Algorithms

- Reconstruction: Instant Neural Graphics Primitives and shared multi-view datasets
- Gas simulation: differentiable smoke simulation
- Particle liquids: WCSPH, PCISPH, IISPH, DFSPH, and PBF
- Deformables: differentiable cloth simulation

## Build Instruction

### Requirements

- CMake 4.4
- CUDA Toolkit 13.3
- C++23 compiler with standard-library module support
- Ninja
- Spectra SDK when building visualization providers

### Build

```bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --parallel 30
```

All core algorithms are always built. Examples and Spectra providers are controlled independently:

```bash
cmake -S . -B cmake-build-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPHYSICA_EXAMPLES=ON \
    -DPHYSICA_BUILD_SPECTRA=OFF
```

Build the Linux CUDA image locally with Docker:

```bash
docker build -t physica .
```
