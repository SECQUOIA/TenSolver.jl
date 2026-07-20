# TenSolver.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://SECQUOIA.github.io/TenSolver.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://SECQUOIA.github.io/TenSolver.jl/dev/)
[![NeurIPS 2025 ScaleOPT paper](https://img.shields.io/badge/paper-NeurIPS_ScaleOPT_2025-orange)](https://openreview.net/pdf?id=EL002DTBRA)
[![JuMP-dev 2026](https://img.shields.io/badge/JuMP--dev-2026-8A2BE2)](https://www.youtube.com/watch?v=nvbv1NzMRMg&feature=youtu.be)


Tensor Network-based solver for discrete polynomial optimization problems,

$$\begin{array}{rl}
  \min_x      & p(x) \\
  \text{s.t.} & Ax = b \\
              & P x \le q \\
              & x \in \mathtt{Constraints} \\
              & x_i \in \{u_0,\ldots,u_{d-1}\} \subset \mathbb{Z}
\end{array}$$

Additionally, TenSolver provides special support
for special classes of optimization problems,
with particular emphasis put in _Quadratic Unconstrained Binary Optimization_ (QUBO).


## Installation

The package is registered in the General registry:

```julia
using Pkg

Pkg.add("TenSolver")
```

## Usage

The simplest way to use this package is passing a matrix to the solver,
while defining an integer domain.

```julia
using TenSolver

Q = [ 1.0  0.0 -3.0;
      1.5 -4.5 -5.0;
     12.0  0.0 -1.0]

E, psi = TenSolver.minimize(Q; domain = [-1, 1])
```

The returned argument `E` is the calculated estimate for the minimum value,
while `psi` is a probability distribution over all possible solutions to the problem.
You can sample solution vectors from it.
These vectors are the (approximate) optimal solutions to the original optimization problem.

```julia
x = TenSolver.sample(psi)
```

### Constraints

TenSolver can also enforce hard constraints on the variables.
Every sampled solution is guaranteed feasible, with no penalty terms involved:

```julia
budget = SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=))
E, psi = TenSolver.maximize([5.5, 2.1, 3.2]; constraints = [budget])
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

Higher-order polynomial, constrained and non-binary optimization features
are currently unavailable via the JuMP interface.

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
