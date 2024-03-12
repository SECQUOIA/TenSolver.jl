# TenSolver.jl

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

E, psi = solve_qubo(Q)
```

The returned argument `E` is the calculated estimate for the minimum value,
while `psi` is a probability distribution over all possible solutions to the problem.
You can sample a Boolean vector from it.

```julia
x = sample_solution(psi)
```

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

  optimize!(m)                   # <-- Equivalent to running solve_qubo(Q)
end
```

### Running on GPU

The solver uses the tensor networks machinery from [ITensors.jl](https://itensor.org/)
which comes with GPU support for tensor contractions.

To run the code in a GPU, all you have to do is passing the appropriate accelerator
to the `solve_qubo` method.
For example, the code below optimizes the QUBO using `CUDA.jl`.

```julia
using TenSolver
import CUDA: cu

Q = randn(4, 4)

E, psi = solve_qubo(Q; accelerator = cu)
```

Since ITensor's GPU platform support is always improving,
be sure to check out their [documentation](https://itensor.github.io/ITensors.jl/stable/)
to know which GPUs are accepted.
