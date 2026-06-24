using LinearAlgebra
import Combinatorics: multiset_permutations

import ITensors: inner
import ITensorMPS: MPS, MPO, dmrg, OpSum, @OpName_str, @SiteType_str, @StateName_str, dim

import ITensors, ITensorMPS

import MultivariatePolynomials: AbstractPolynomial, coefficient, monomial, terms, variables, effective_variables, isconstant

# Strict upper triangular part of an array
function upper_indices(a)
  return (Tuple(ci)
            for ci in CartesianIndices(size(a))
            if issorted(Tuple(ci); lt = (<=)))
end

function constant(p::AbstractPolynomial{T}) where T
  ts  = terms(p)
  idx = findfirst(isconstant, ts)

  return isnothing(idx) ? zero(T) : coefficient(ts[idx])
end

maybe(f::Function, mx::Nothing; default=nothing) = default
maybe(f::Function, mx; default=nothing) = f(mx)

expectation(H, x) = inner(x', H, x)
variance(H::MPO, x::MPS) = real(inner(H, x, H, x) - expectation(H, x)^2)

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

# Diagonal matrix whose eigenvalues are the ordered feasible values for an integer variable.
# For qubits, this is a projection on |1>. Or equivalently, (I - σ_z) / 2.
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"D",::SiteType"Qudit", d::Int) = diagm(0:(d-1))

ITensors.state(::StateName"full", ::SiteType"Qudit", s::ITensorMPS.Index) = (d = dim(s); fill(1/sqrt(d), d))

"""
    tensorize(p)

Turn a polynomial function action on bitstrings
into an equivalent MPO Hamiltonian acting on Qudit sites.
The conversion consists of exchanging each integer variable `x_i`
for a matrix `P_i` whose eigenvalues represent its feasible set `K_i`.

    ∑ Q_ij x_i x_j + ∑ l_i x_i --> H = Σ Q_ij D_i D_j + ∑ l_i D_i
"""
function tensorize end

function tensorize(Q::AbstractArray{T}, rest::Vararg{AbstractArray{T}}; cutoff = zero(T)) where T
  Qs = [Q, rest...]
  if !allequal(Iterators.flatmap(size, Qs))
    throw(DimensionMismatch("All arrays should act on the same number of variables.\nEncountered dimensions $(collect(map(size, Qs)))."))
  end

  N = size(Q, 1)
  sites = ITensors.siteinds("Qudit", N; dim = 2)
  os = OpSum{T}()

  for t in Qs
    for idx in upper_indices(t)
      # Due to commutativity, we can group all terms that have the same indices.
      # This speeds up the calculations.
      # e.g., Q_ij x_i x_j + Q_ji x_j x_i = (Q_ij + Q_ji) x_i x_j
      coeff = sum(k -> t[k...], multiset_permutations(idx, ndims(t)))

      if abs(coeff) > cutoff
        op   = Iterators.flatmap(v -> ("D", v), idx)
        os .+= (coeff, op...)
      end
    end
  end

  return isempty(os) ? MPO(T,sites) : MPO(T, os, sites)
end

function tensorize(p::AbstractPolynomial{T}; cutoff = zero(T)) where T
  N = length(effective_variables(p))
  sites = ITensors.siteinds("Qudit", N; dim = 2)
  os = OpSum{T}()

  # Map: var name => index
  indices = Dict(v => i for (i, v) in enumerate(effective_variables(p)))

  for t in terms(p)
    coeff = coefficient(t)

    if abs(coeff) > cutoff && ! isconstant(t)
      vars = effective_variables(t)
      op   = Iterators.flatmap(v -> ("D", indices[v]), vars)
      os .+= (coeff, op...)
    end
  end

  return isempty(os) ? MPO(T,sites) : MPO(T, os, sites)
end


"""
    minimize(Q::Matrix[, l::Vector[, c::Number ; device, cutoff, kwargs...)

Solve the Quadratic Unconstrained Binary Optimization (QUBO) problem

    min  b'Qb + l'b + c
    s.t. b_i in {0, 1}

Return the optimal value `E` and a probability distribution `ψ` over optimal solutions.
You can use [`sample`](@ref) to get an actual bitstring from `ψ`.

This function uses DMRG with tensor networks to calculate the optimal solution,
by finding the ground state (least eigenspace) of the Hamiltonian

    H = Σ Q_ij D_iD_j + Σ l_i D_i

where D_i acts locally on the i-th qubit as [0 0; 0 1], i.e, the projection on |1>.

Keyword arguments:

- `iterations :: Int` - Maximum iterations the solver should run. Defaults to `10`.
- `cutoff :: Float64` - Any absolute value below this threshold is considered zero. Defaults to `1e-8`.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `maxdim` - The maximum allowed bond dimension.
  Integer or array of integer specifying the bond dimension per iteration.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `mindim` - The minimum allowed bond dimension, if possible.  Defaults to `1`.
  Integer or array of integer specifying the bond dimension per iteration.
- `time_limit :: Float64` - If specified, determines the maximum running time in seconds.
  It only determines whether a new iteration should start or not, thus the solver may run for longer if the threshold happens during an iteration.
- `device = cpu` - Accelerator device used during computation.
  See the section below for how to run on GPUs.
- `vtol :: Float64` - If specified, determines the variance tolerance before the algorithm stops.
  The variance test determines whether DMRG converged to an eigenstate (not necessarily the ground state),
  but is expensive to calculate.
- `preprocess :: Bool` - Defaults to `false`. If `true`, permute QUBO variables before constructing the MPS Hamiltonian
  so coupled variables are closer in the one-dimensional tensor order. Samples are returned in the
  caller's original variable order. This is an experimental feature and may be subject to changes.
- `noise` - A float or array of floats (per iteration) specifying the noise term added to the system to help with convergence.
  It is recommended to use a large noise (~ 1e-5) on the initial iterations and let it go to zero on later iterations.
- `eigsolve_krylovdim :: Int = 3` - Maximum Krylov space dimension used in the local eigensolver.
- `eigsolve_tol :: Float64 = 1e-14` - Eigensolver tolerance.
- `eigsolve_maxiter :: Int = 1` - Maximum iterations for eigensolver.
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

The returned `Solution` carries per-iteration stats in `.energies`, `.bond_dims`, and `.elapsed_times`.

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

include("backends/dmrg.jl")

const default_backend = DMRGBackend()

function minimize(Q :: AbstractMatrix{T} , l :: Union{AbstractVector{T}, Nothing} = nothing , c :: T = zero(T); backend=default_backend, kwargs...) where T
  return minimize(normalize_backend(backend), Q, l, c; kwargs...)
end

function minimize(p::AbstractPolynomial{T}; backend=default_backend, kwargs...) where T
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

See also [`maximize`](@ref).
"""
minimize(Q :: AbstractMatrix{T}, c :: T; kwargs...) where T = minimize(Q, nothing, c; kwargs...)

"""
    maximize(Q::Matrix[, l::Vector[, c::Number; kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem
for maximization.

    max  b'Qb + l'b + c
    s.t. b_i in {0, 1}

See also [`minimize`](@ref).
"""
function maximize(qs... ; kwargs...)
  # Flip the sign of all non-nothing elements
  # max p(x) = - min -p(x)
  mqs = map(q -> maybe(-, q), qs)
  E, psi = minimize(mqs...; kwargs...)

  return -E, psi
end
