# TenSolver.jl

Tensor Network-based solver for discrete polynomial optimization.

## Overview

TenSolver.jl provides an efficient solver for QUBO problems using tensor network methods,
leveraging the Density Matrix Renormalization Group (DMRG) algorithm.
The package is particularly useful for:

- Solving large-scale discrete optimization problems
- Finding approximate solutions to NP-hard problems
- Exploring the solution space of combinatorial optimization problems
- Enforcing hard constraints exactly through projection, without penalty terms
- GPU-accelerated optimization

## Installation

```julia
using Pkg
Pkg.add("TenSolver")
```

## Quick Start

The simplest way to use this package is passing a matrix to the solver:

```jldoctest quickstart
using TenSolver

assets = ["Wind", "Solar", "Battery"]

# Wind and solar overlap, while storage pairs well with either source.
Q = [0.0 1.0 -1.5;
     1.0 0.0 -0.5;
     -1.5 -0.5 0.0]
E, psi = TenSolver.minimize(Q; verbosity=0)

# Verify we found the minimum
E ≈ -3.0

# output

true
```

The returned argument `E` is the calculated estimate for the minimum value,
while `psi` is a probability distribution over all possible solutions to the problem.
You can sample from it:

```jldoctest quickstart
x = TenSolver.sample(psi)

# Verify the sampled solution achieves the minimum and inspect the chosen assets
(x' * Q * x ≈ E, join(assets[findall(==(1), x)], ", "))

# output

(true, "Wind, Battery")
```

## Features

- **Tensor Network Optimization**: Uses advanced tensor network methods (DMRG) for efficient optimization
- **Probability Distribution**: Returns a probability distribution over optimal solutions
- **Hard Constraints**: Enforces constraints exactly via CoTenN-style projection MPOs, so sampled solutions are always feasible (see [Constrained Optimization](@ref))
- **JuMP Integration**: Works seamlessly with the JuMP modeling language
- **GPU Support**: Supports GPU acceleration via CUDA.jl, Metal.jl, and other accelerators
- **Flexible Configuration**: Numerous parameters to control accuracy vs. performance trade-off

## Citing TenSolver

If you use TenSolver in your research, please cite our [NeurIPS 2025 Workshop paper](https://openreview.net/pdf?id=EL002DTBRA):

```bibtex
@inproceedings{tensolver2025,
  title     = {Quantum-Inspired Tensor Network Methods for Quadratic Unconstrained Binary Optimization},
  author    = {Iago {Leal de Freitas} and Jo{\~a}o Victor {Paim de Cerqueira Melo Souza} and David E. {Bernal Neira}},
  booktitle = {NeurIPS Workshop on GPU-Accelerated and Scalable Optimization},
  year      = {2025},
  url       = {https://openreview.net/forum?id=EL002DTBRA}
}
```

## Contents

```@contents
Pages = ["examples.md", "constraints.md", "api.md"]
Depth = 2
```
