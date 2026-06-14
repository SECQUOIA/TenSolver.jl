using LinearAlgebra
import Combinatorics: multiset_permutations

import ITensors: inner
import ITensorMPS: MPS, MPO, dmrg, OpSum, @OpName_str, @SiteType_str, @StateName_str, dim

import ITensors, ITensorMPS

import MultivariatePolynomials: AbstractPolynomial, coefficient, monomial, terms, variables, effective_variables, isconstant

function issquare(a :: AbstractArray)
  return allequal(size(a))
end

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

The default implementation is [`DMRGBackend`](@ref). Other backends must provide
backend-specific `_minimize` methods for the normalized optimization inputs they
support. Matrix backends implement
`_minimize(::MyBackend, Q::AbstractMatrix, l, c; kwargs...)`; polynomial
backends implement `_minimize(::MyBackend, p::AbstractPolynomial; kwargs...)`.
Extensions that support symbolic selection can also define
`_normalize_backend(::Val{:my_backend}) = MyBackend(...)`.
"""
abstract type AbstractTenSolverBackend end

"""
    DMRGBackend()

Select TenSolver's default ITensorMPS DMRG backend.
"""
struct DMRGBackend <: AbstractTenSolverBackend end

abstract type AbstractStructuredTopology end

# Structured square-grid topology for optional PEPS solves.
#
# Variables are assumed to be ordered according to SpinGlassNetworks'
# `super_square_lattice((m, n, spins_per_site))` convention.
struct SquareGrid <: AbstractStructuredTopology
  m :: Int
  n :: Int
  spins_per_site :: Int

  function SquareGrid(m::Integer, n::Integer, spins_per_site::Integer=1)
    m > 0 || throw(ArgumentError("SquareGrid requires m > 0. Got $m."))
    n > 0 || throw(ArgumentError("SquareGrid requires n > 0. Got $n."))
    spins_per_site > 0 || throw(ArgumentError("SquareGrid requires spins_per_site > 0. Got $spins_per_site."))
    return new(Int(m), Int(n), Int(spins_per_site))
  end
end

# Structured king-grid topology for optional PEPS solves. It uses the same
# variable ordering as `SquareGrid`, but the PEPS compatibility graph also
# allows diagonal interactions between neighboring grid cells.
struct KingGrid <: AbstractStructuredTopology
  m :: Int
  n :: Int
  spins_per_site :: Int

  function KingGrid(m::Integer, n::Integer, spins_per_site::Integer=1)
    m > 0 || throw(ArgumentError("KingGrid requires m > 0. Got $m."))
    n > 0 || throw(ArgumentError("KingGrid requires n > 0. Got $n."))
    spins_per_site > 0 || throw(ArgumentError("KingGrid requires spins_per_site > 0. Got $spins_per_site."))
    return new(Int(m), Int(n), Int(spins_per_site))
  end
end

_topology_size(topology::AbstractStructuredTopology) = topology.m * topology.n * topology.spins_per_site
_topology_tuple(topology::AbstractStructuredTopology) = (topology.m, topology.n, topology.spins_per_site)
_topology_name(::SquareGrid) = "square"
_topology_name(::KingGrid) = "king"

# Internal scaffold for the optional SpinGlassPEPS structured backend.
#
# The backend is implemented by the `TenSolverSpinGlassPEPSExt` package
# extension, which loads only when `SpinGlassNetworks`, `SpinGlassEngine`, and
# `SpinGlassTensors` are available. Without those packages this backend errors
# clearly and the default DMRG backend remains unchanged.
#
# This constructor is intentionally not exported while the current registered
# SpinGlass component dependency stack does not resolve in the same environment
# as TenSolver's ITensors/QUBOTools stack and CI cannot exercise the extension.
struct PEPSBackend{T <: AbstractStructuredTopology, S} <: AbstractTenSolverBackend
  topology :: T
  beta :: Float64
  bond_dim :: Int
  max_states :: Int
  cutoff_prob :: Float64
  onGPU :: Bool
  contraction :: Symbol
  num_sweeps :: Int
  graduate_truncation :: Bool
  transformations :: S
  local_dimension :: Union{Nothing, Int}
  no_cache :: Bool
end

function PEPSBackend(topology::AbstractStructuredTopology;
                     beta::Real = 2.0,
                     bond_dim::Integer = 16,
                     max_states::Integer = 2^8,
                     cutoff_prob::Real = 1e-4,
                     onGPU::Bool = false,
                     contraction::Symbol = :auto,
                     num_sweeps::Integer = 1,
                     graduate_truncation::Bool = true,
                     transformations = :all,
                     local_dimension::Union{Nothing, Integer} = nothing,
                     no_cache::Bool = false)
  beta > 0 && isfinite(beta) || throw(ArgumentError("PEPSBackend requires finite beta > 0. Got $beta."))
  bond_dim >= 1 || throw(ArgumentError("PEPSBackend requires bond_dim >= 1. Got $bond_dim."))
  max_states >= 1 || throw(ArgumentError("PEPSBackend requires max_states >= 1. Got $max_states."))
  cutoff_prob >= 0 || throw(ArgumentError("PEPSBackend requires cutoff_prob >= 0. Got $cutoff_prob."))
  num_sweeps >= 1 || throw(ArgumentError("PEPSBackend requires num_sweeps >= 1. Got $num_sweeps."))
  contraction in (:auto, :svd, :svd_truncate, :zipper) ||
    throw(ArgumentError("Unsupported PEPS contraction $(repr(contraction)). Use :auto, :svd, :svd_truncate, or :zipper."))
  if !isnothing(local_dimension)
    local_dimension >= 1 || throw(ArgumentError("PEPSBackend requires local_dimension >= 1 when provided. Got $local_dimension."))
  end

  return PEPSBackend{typeof(topology), typeof(transformations)}(
    topology,
    Float64(beta),
    Int(bond_dim),
    Int(max_states),
    Float64(cutoff_prob),
    onGPU,
    contraction,
    Int(num_sweeps),
    graduate_truncation,
    transformations,
    isnothing(local_dimension) ? nothing : Int(local_dimension),
    no_cache,
  )
end

const default_backend = DMRGBackend()

function _backend_error(backend)
  if backend === :peps
    return ArgumentError("backend :peps is not available. Install/load the PEPS extension or use backend = :dmrg.")
  end

  return ArgumentError("No `_minimize` method is available for backend $(repr(backend)). Use backend = :dmrg or provide a backend-specific `_minimize` method.")
end

_backend_error(::PEPSBackend) = ArgumentError("PEPSBackend is not available. Install/load SpinGlassNetworks, SpinGlassEngine, and SpinGlassTensors to activate the PEPS extension, or use backend = :dmrg.")

_normalize_backend(backend::DMRGBackend) = backend
_normalize_backend(backend::PEPSBackend) = backend
_normalize_backend(backend::AbstractTenSolverBackend) = backend
_normalize_backend(backend::Symbol) = _normalize_backend(Val(backend))
_normalize_backend(::Val{:dmrg}) = default_backend
function _normalize_backend(::Val{backend}) where {backend}
  throw(_backend_error(backend))
end
_normalize_backend(backend) = throw(_backend_error(backend))

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
- `mindim` - The minimum allowed bond dimension, if possible.
  Integer or array of integer specifying the bond dimension per iteration.
- `time_limit :: Float64` - If specified, determines the maximum running time in seconds.
  It only determines whether a new iteration should start or not, thus the solver may run for longer if the threshold happens during an iteration.
- `device = cpu` - Accelerator device used during computation.
  See the section below for how to run on GPUs.
- `vtol :: Float64` - If specified, determines the variance tolerance before the algorithm stops.
  The variance test determines whether DMRG converged to an eigenstate (not necessarily the ground state),
  but is expensive to calculate.
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
  Other backends are reserved for optional extensions and error clearly when
  unavailable.

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

function minimize(Q :: AbstractMatrix{T} , l :: Union{AbstractVector{T}, Nothing} = nothing , c :: T = zero(T); backend=default_backend, cutoff=1e-8, kwargs...) where T
  return _minimize(_normalize_backend(backend), Q, l, c; cutoff, kwargs...)
end

function minimize(p::AbstractPolynomial{T}; backend=default_backend, cutoff=1e-8, kwargs...) where T
  return _minimize(_normalize_backend(backend), p; cutoff, kwargs...)
end

function _minimize(backend::AbstractTenSolverBackend, args...; kwargs...)
  throw(_backend_error(backend))
end

function _minimize(backend::PEPSBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); kwargs...) where T
  return _solve_ising(backend, qubo_to_ising(Q, l, c); kwargs...)
end

function _minimize(backend::PEPSBackend, p::AbstractPolynomial; kwargs...)
  throw(ArgumentError("PEPSBackend does not support polynomial inputs directly. Convert to a structured QUBO or call solve_ising with a supported topology."))
end

"""
    solve_ising(model; backend = DMRGBackend(), kwargs...)
    solve_ising(J, h[, offset]; backend = DMRGBackend(), kwargs...)

Solve an Ising model with spins `s_i in {-1, +1}`.

The returned solution still samples TenSolver Boolean vectors using
`x_i = (s_i + 1) / 2`. The default DMRG path converts the Ising model back to a
QUBO and calls [`minimize`](@ref). Optional structured backends can implement
this boundary directly.
"""
function solve_ising end

function solve_ising(model::IsingModel; backend=default_backend, kwargs...)
  return _solve_ising(_normalize_backend(backend), model; kwargs...)
end

function solve_ising(J::AbstractMatrix, h::AbstractVector, offset::Real=0; backend=default_backend, kwargs...)
  return solve_ising(IsingModel(J, h, offset); backend, kwargs...)
end

function _solve_ising(backend::AbstractTenSolverBackend, model::IsingModel; kwargs...)
  throw(_backend_error(backend))
end

function _solve_ising(::DMRGBackend, model::IsingModel; kwargs...)
  qubo = ising_to_qubo(model)
  return _minimize(default_backend, qubo.Q, qubo.l, qubo.c; kwargs...)
end

function _minimize(::DMRGBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); cutoff=1e-8, kwargs...) where T
  H      = tensorize(Q, isnothing(l) ? diag(Q) : diag(Q) + l; cutoff)
  obj(x) = dot(x, Q, x) + c + maybe(l -> dot(l,x), l; default=zero(T))

  return _minimize_mpo(H, c, obj; cutoff, kwargs...)
end

function _minimize(::DMRGBackend, p::AbstractPolynomial{T}; cutoff=1e-8, kwargs...) where T
  H   = tensorize(p)
  cte = constant(p)
  vs = variables(p)
  return _minimize_mpo(H, cte, a -> p(vs => a); cutoff, kwargs...)
end

function _single_variable_minimize(::Type{T}, sites, obj, initial_time; device, verbosity, on_iteration, callback_every) where {T}
  x0 = [0]
  x1 = [1]
  e0 = obj(x0)
  e1 = obj(x1)

  energy, state = if e0 == e1
    (T(e0), ["full"])
  elseif e0 < e1
    (T(e0), string.(x0))
  else
    (T(e1), string.(x1))
  end

  psi = ITensorMPS.MPS(sites, state) |> device
  elapsed_time = time() - initial_time
  bond_dim = ITensorMPS.maxlinkdim(psi)

  iterlog_header(verbosity)
  iterlog_iteration(verbosity, 1, energy, bond_dim, 0.0, elapsed_time)

  dist = Solution{T}(psi, T[energy], Int[bond_dim], Float64[elapsed_time])
  if !isnothing(on_iteration) && 1 % callback_every == 0
    on_iteration(psi; iteration=1, objective=energy, bond_dim, elapsed_time)
  end

  iterlog_footer(verbosity, energy, elapsed_time)

  return energy, dist
end

function _minimize_mpo( H :: MPO
                      , c :: T
                      , obj
                      ; device     :: Function = cpu
                      , cutoff     = 1e-8  #  a cutoff of 1E-5 gives sensible accuracy; a cutoff of 1E-8 is high accuracy; and a cutoff of 1E-12 is near exact accuracy. (https://itensor.org/docs.cgi?page=tutorials/dmrg_params)
                      , verbosity  = 1
                      # Stopping criteria
                      , iterations :: Union{Nothing, Int} = nothing
                      , time_limit = +Inf
                      , vtol       = cutoff
                      , check_variance_every_iteration = 10
                      # DMRG keywords
                      , inidim     = 40
                      , maxdim     = [10, 10, 10, 20, 50, 100, 100, 200, 300, 300, 400, 400, 800, 900, 1000]
                      , mindim     = 1
                      , noise      = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 0.0] # 1E-5 is a lot of noise and 1E-12 is a minimal amount of noise that can still be considered non-zero.
                      , eigsolve_krylovdim :: Int     = 3
                      , eigsolve_maxiter   :: Int     = 2
                      , eigsolve_tol       :: Float64 = 1e-14
                      # Iteration callback
                      , on_iteration :: Union{Nothing, Function} = nothing
                      , callback_every   :: Int = 1
                      ) where {T}
  callback_every >= 1 || throw(ArgumentError("`callback_every` must be >= 1, got $callback_every"))
  initial_time      = time()
  energies_log      = T[]
  bond_dims_log     = Int[]
  elapsed_times_log = Float64[]

  # Quantization
  sites = ITensorMPS.siteinds(first, H; plev=0)
  if length(sites) == 1
    return _single_variable_minimize(T, sites, obj, initial_time; device, verbosity, on_iteration, callback_every)
  end

  H     = device(H)
  psi   = ITensorMPS.random_mps(T, sites; linkdims=inidim) |> device

  @debug("MPO construction finished",
    time=(time() - initial_time),
    max_bond = ITensorMPS.maxlinkdim(H),
    num_coefficients = sum(prod(m) for m in H),
  )

  iterlog_header(verbosity)
  var = Inf
  local energy, psi

  for i in Iterators.countfrom(1)
    energy, psi = dmrg(H, device(psi)
                      ; nsweeps     = 1
                      , ishermitian = true
                      , outputlevel = 0
                      , cutoff
                      , maxdim = maxdim[min(i, length(maxdim))]
                      , mindim = mindim[min(i, length(mindim))]
                      , noise  = noise[min(i, length(noise))]
                      , eigsolve_krylovdim
                      , eigsolve_tol
                      , eigsolve_maxiter
                      , eigsolve_verbosity = 0
                 )

    # Get metadata #
    if i % check_variance_every_iteration == 0
      vtime = time()
      var = variance(H, psi)
      @debug "Calculate variance" variance=var time=(time() - vtime())
    end

    elapsed_time = time() - initial_time

    bond_dim = ITensorMPS.maxlinkdim(psi)

    iterlog_iteration(
      verbosity,
      i,
      energy + c,
      bond_dim,
      i % check_variance_every_iteration == 0 ? var : nothing,
      elapsed_time,
    )

    # Per-iteration stats (always collected)
    push!(energies_log,      energy + c)
    push!(bond_dims_log,     bond_dim)
    push!(elapsed_times_log, elapsed_time)

    # Optional callback
    if !isnothing(on_iteration) && i % callback_every == 0
      on_iteration(psi; iteration=i, objective=energy+c, bond_dim, elapsed_time)
    end

    # Stopping criteria #
    if !isnothing(iterations) && i >= iterations
      @debug "Stopping: maximum iterations reached" iteration=i
      break
    elseif elapsed_time > time_limit
      @debug "Stopping: maximum time reached" iteration=i limit=time_limit time=elapsed_time
      break
    elseif var < vtol
      @debug "Stopping: variance below tolerance" variance=var tolerance=vtol
      break
    end
  end

  # The calculated energy has approximation errors compared to the true solution.
  # It makes more sense to sample a solution and calculate the true objective function applied to it.
  dist = Solution{T}(psi, energies_log, bond_dims_log, elapsed_times_log)
  x    = sample(dist)
  optimal = obj(x)
  elapsed_time = time() - initial_time

  iterlog_footer(verbosity, optimal, elapsed_time)

  return optimal, dist
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
