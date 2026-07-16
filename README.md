# TenSolver.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://SECQUOIA.github.io/TenSolver.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://SECQUOIA.github.io/TenSolver.jl/dev/)

Tensor Network-based solver for binary optimization problems:
quadratic (QUBO), higher-order polynomial (PUBO), and constrained.

$$\begin{array}{rl}
  \min_x      & p(x) \\
  \text{s.t.} & Ax = b \\
   P x \le q \\
   x \text{ satisfies other constraints} \\
              & x \in \mathbb{B}^{n}
\end{array}$$

Here `p` may be a quadratic form `x' Q x + l' x + c` or an arbitrary polynomial.

## Installation

The package is registered in the General registry:

```julia
using Pkg

Pkg.add("TenSolver")
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

TenSolver can also enforce hard constraints on the binary variables. Every
sampled solution is guaranteed feasible, with no penalty terms involved:

```julia
budget = SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=))
E, psi = TenSolver.maximize(zeros(3, 3), values; constraints = [budget])
```

The constraint API is experimental and subject to change; see the
[constraints documentation](https://SECQUOIA.github.io/TenSolver.jl/dev/constraints/)
for the available types, the projection method behind them, and worked examples.

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
