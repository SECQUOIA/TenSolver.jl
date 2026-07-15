# TenSolver.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://SECQUOIA.github.io/TenSolver.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://SECQUOIA.github.io/TenSolver.jl/dev/)

Tensor Network-based solver for **Q**uadratic **U**nconstrained **B**inary **O**ptimization (QUBO) problems.

$$\begin{array}{rl}
  \min_x      & x' Q x \\
  \text{s.t.} & x \in \mathbb{B}^{n}
\end{array}$$

## Installation

This package is currently not registered. Install it directly from the git url:

```julia
using Pkg

Pkg.add(url="https://github.com/SECQUOIA/TenSolver.jl.git")
```

## Usage

The simplest way to use this package is passing a matrix to the solver

```julia
using TenSolver

Q = randn(40, 40)

E, psi = TenSolver.minimize(Q)
```

The returned argument `E` is the calculated estimate for the minimum value,
while `psi` is a probability distribution over all possible solutions to the problem.
You can sample Boolean vectors from it.
These vectors are the (approximate) optimal solutions to the original optimization problem.

```julia
x = TenSolver.sample(psi)
```

### Constraints

TenSolver can also enforce **hard constraints** on the binary variables.
Rather than adding penalty terms to the objective, each constraint is lowered to
a projection MPO that removes the infeasible subspace exactly, so every sampled
solution is guaranteed feasible.

```julia
using TenSolver

# Maximize value while selecting at most two of the three assets.
values = [3.0, 2.0, 4.0]
budget = SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=))

E, psi = TenSolver.minimize(zeros(3, 3), -values; constraints = [budget])
x = TenSolver.sample(psi)   # always satisfies the budget
```

The available constraint types are `SumConstraint`, `NotEqualsConstraint`,
`ExactlyOneConstraint`, and `RelationConstraint`. This projection design is
adapted from CoTenN (Sharma, Peng, Dangwal, and Achour, *"CoTenN: Constrained
Optimization with Tensor Networks,"* PLDI 2026). Hard constraints are
experimental; see the [documentation](https://SECQUOIA.github.io/TenSolver.jl)
for details.

### JuMP interface

Alternatively, we also provide an `Optimizer`
for solving QUBOs described as [JuMP](https://jump.dev) models.

```julia
using JuMP, TenSolver

dim = 40
Q   = randn(dim, dim)

begin
  m = Model(TenSolver.Optimizer) # <-- The important line
  @variable(m, x[1:dim], Bin)
  @objective(m, Min, x'Q*x)

  optimize!(m)                   # <-- Equivalent to running TenSolver.minimize(Q)
end
```

### Running on GPU

The solver uses the tensor networks machinery from [ITensors.jl](https://itensor.org/)
which comes with GPU support for tensor contractions.

To run the code in a GPU, all you have to do is passing the appropriate accelerator
as a keyword to the solver.
For example, the code below optimizes the QUBO using `CUDA.jl`.

```julia
using TenSolver
import CUDA: cu

Q = randn(4, 4)

E, psi = minimize(Q; device = CUDA.cu)
```

Since ITensor's GPU platform support is always improving,
be sure to check out their [documentation](https://itensor.github.io/ITensors.jl/stable/)
to know which GPUs are accepted.
