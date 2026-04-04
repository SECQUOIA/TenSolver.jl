# TenSolver.jl

Tensor Network-based solver for **Q**uadratic **U**nconstrained **B**inary **O**ptimization (QUBO) problems.

## Overview

TenSolver.jl provides an efficient solver for QUBO problems using tensor network methods, specifically leveraging the Density Matrix Renormalization Group (DMRG) algorithm. The package is particularly useful for:

- Solving large-scale binary optimization problems
- Finding approximate solutions to NP-hard problems
- Exploring the solution space of combinatorial optimization problems
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

Q = [0.0 -1.0; -1.0 0.0]
E, psi = TenSolver.minimize(Q; verbosity=0)

# Verify we found the minimum
E ≈ -2.0

# output

true
```

The returned argument `E` is the calculated estimate for the minimum value,
while `psi` is a probability distribution over all possible solutions to the problem.
You can sample Boolean vectors from it:

```jldoctest quickstart
x = TenSolver.sample(psi)

# Verify the sampled solution achieves the minimum
x' * Q * x ≈ E

# output

true
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
