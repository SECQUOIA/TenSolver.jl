# TenSolver.jl

Tensor Network-based solver for **Q**uadratic **U**nconstrained **B**inary **O**ptimization (QUBO) problems.

## Overview

TenSolver.jl provides an efficient solver for QUBO problems using tensor network methods, specifically leveraging the Density Matrix Renormalization Group (DMRG) algorithm. The package is particularly useful for:

- Solving large-scale binary optimization problems
- Finding approximate solutions to NP-hard problems
- Exploring the solution space of combinatorial optimization problems
- GPU-accelerated optimization

## Installation

This package is currently not registered. Install it directly from the git url:

```julia
using Pkg
Pkg.add(url="https://github.com/SECQUOIA/TenSolver.jl.git")
```

## Quick Start

The simplest way to use this package is passing a matrix to the solver:

```julia
using TenSolver

Q = randn(40, 40)
E, psi = TenSolver.minimize(Q)
```

The returned argument `E` is the calculated estimate for the minimum value,
while `psi` is a probability distribution over all possible solutions to the problem.
You can sample Boolean vectors from it:

```julia
x = TenSolver.sample(psi)
```

## Features

- **Tensor Network Optimization**: Uses advanced tensor network methods (DMRG) for efficient optimization
- **Probability Distribution**: Returns a probability distribution over optimal solutions
- **JuMP Integration**: Works seamlessly with the JuMP modeling language
- **GPU Support**: Supports GPU acceleration via CUDA.jl, Metal.jl, and other accelerators
- **Flexible Configuration**: Numerous parameters to control accuracy vs. performance trade-off

## Contents

```@contents
Pages = ["api.md", "examples.md"]
Depth = 2
```
