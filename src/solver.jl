using LinearAlgebra
import Combinatorics: multiset_permutations

import MultivariatePolynomials: AbstractPolynomial, coefficient, monomial, terms, variables, effective_variables, isconstant

maybe(f::Function, mx::Nothing; default=nothing) = default
maybe(f::Function, mx; default=nothing) = f(mx)

ifnotnothing(a, b) = maybe(_ -> b, a)

"""
    AbstractTenSolverBackend

Abstract solver backend marker for TenSolver implementations.

Backends must provide backend-specific `minimize` methods for the normalized
optimization inputs they support. Matrix backends implement
`minimize(::MyBackend, Q::AbstractMatrix, l, c; kwargs...)`; polynomial
backends implement `minimize(::MyBackend, p::AbstractPolynomial; kwargs...)`.
Extensions that support symbolic selection must also define
`normalize_backend(::Val{:my_backend}) = MyBackend(...)`.

# See also

[`DMRGBackend`](@ref), [`normalize_backend`](@ref)
"""
abstract type AbstractTenSolverBackend end

"""
    normalize_backend(backend)

Normalize a user-facing backend selector into a backend object.

Backends can support `backend = :my_backend` by defining
`normalize_backend(::Val{:my_backend}) = MyBackend(...)`.
"""
function normalize_backend end

function backend_error(backend)
  if backend === :peps
    return ArgumentError("backend :peps is not available. Install/load the PEPS extension or use backend = :dmrg.")
  end

  return ArgumentError("No backend-specific `minimize` method is available for backend $(repr(backend)). Use backend = :dmrg or provide a backend-specific `minimize` method.")
end

normalize_backend(backend::AbstractTenSolverBackend) = backend
normalize_backend(backend::Symbol) = normalize_backend(Val(backend))
function normalize_backend(::Val{backend}) where {backend}
  throw(backend_error(backend))
end
normalize_backend(backend) = throw(backend_error(backend))


#
# Backends
#
include("backends/dmrg.jl")
const default_backend = DMRGBackend()

"""
    minimize(Q::Matrix[, l::Vector[, c::Number ; device, cutoff, kwargs...)

Solve the Quadratic Unconstrained Binary Optimization (QUBO) problem

    min  b'Qb + l'b + c
    s.t. b_i in {0, 1}

Return the optimal value `E` and a probability distribution `ψ` over optimal solutions.
You can use [`sample`](@ref) to get an actual bitstring from `ψ`.

There are multiple backends available, selected through the keyword `backend`.
By default, it uses DMRG to calculate the optimal solution.


Keyword arguments:

- `iterations :: Int` - Maximum iterations the solver should run. Defaults to `10`.
- `cutoff :: Float64` - Any absolute value below this threshold is considered zero. Defaults to `1e-8`.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `time_limit :: Float64` - If specified, determines the maximum running time in seconds.
  It only determines whether a new iteration should start or not, thus the solver may run for longer if the threshold happens during an iteration.
- `device = cpu` - Accelerator device used during computation.
  See the section below for how to run on GPUs.
- `preprocess :: Bool` - Defaults to `false`. If `true`, permute QUBO variables before constructing the MPS Hamiltonian
  so coupled variables are closer in the one-dimensional tensor order. Samples are returned in the
  caller's original variable order. This is an experimental feature and may be subject to changes.
- `on_iteration :: Function` - Called after each recorded iteration as
  `f(psi::MPS; iteration, objective, bond_dim, elapsed_time)`.
  `objective` is the expected objective function ⟨ψ|H|ψ⟩ at this iteration.
  Use to collect statistics or serialize intermediate states.
  `psi` is the MPS for that iteration.
  Default: `nothing` (no callback).
- `callback_every :: Int` - Invoke the callback every N iterations. Must be >= 1. Default: `1`.
- `backend` - Solver backend. Defaults to the current DMRG implementation.
  Use `backend = :dmrg` or `backend = DMRGBackend()` to select it explicitly.
  Other backends are reserved for optional extensions.

  Other keywords might be available depending on the chosen backend.
  See the documentation for each backend for comprehensive lists.

The returned `Solution` carries per-iteration stats in the fields `energies`, `bond_dims`, and `elapsed_times`.

Provably infeasible constrained models are reported as a status:
`minimize` logs a warning and returns `+Inf` (the minimum over an empty feasible set)
together with an infeasible [`Solution`](@ref), which cannot be sampled.
Check it with [`is_feasible`](@ref).

Running on GPU:

The optional keyword `device` controls whether the solver should run on CPU or GPU.
For using a GPU, you can import the respective package, e.g. CUDA.jl,
and pass its accelerator as argument.

```julia
import CUDA
minimize(Q; device = CUDA.cu)

import Metal
minimize(Q; device = Metal.mtl)
```


See also [`maximize`](@ref).
"""
function minimize end

function minimize(
  Q :: AbstractMatrix{T},
  l :: Union{AbstractVector{T}, Nothing} = nothing,
  c :: T = zero(T)
  ;
  backend=default_backend,
  kwargs...,
) where T
  return minimize(normalize_backend(backend), Q, l, c; kwargs...)
end

function minimize(
  p::AbstractPolynomial{T}
  ;
  backend=default_backend,
  kwargs...,
) where T
  return minimize(normalize_backend(backend), p; kwargs...)
end

function minimize(backend::AbstractTenSolverBackend, args...; kwargs...)
  throw(backend_error(backend))
end

"""
    minimize(Q::Matrix, c::Number; kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem with no linear term.

    min  b'Qb + c
    s.t. b_i in {0, 1}
"""
minimize(Q :: AbstractMatrix{T}, c :: T; kwargs...) where T = minimize(Q, nothing, c; kwargs...)

"""
    maximize(Q::Matrix[, l::Vector[, c::Number; kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem
for maximization.

    max  b'Qb + l'b + c
    s.t. b_i in {0, 1}

All keywords accepted by [`minimize`](@ref) can also be used for maximization problems.
Provably infeasible constrained models return `-Inf` (the supremum over an
empty feasible set) together with an infeasible [`Solution`](@ref).

See also [`minimize`](@ref).
"""
function maximize(qs... ; kwargs...)
  # Flip the sign of all non-nothing elements
  # max p(x) = - min -p(x)
  mqs = map(q -> maybe(-, q), qs)
  E, psi = minimize(mqs...; kwargs...)

  return -E, psi
end
