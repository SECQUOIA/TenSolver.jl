#----------------------------------------------------------#
# Optional SpinGlassPEPS structured backend                 #
#----------------------------------------------------------#
#
# The solve itself is implemented by the `TenSolverSpinGlassPEPSExt` package
# extension, which loads only when `SpinGlassNetworks`, `SpinGlassEngine`, and
# `SpinGlassTensors` are available. Without those packages this backend errors
# clearly and the default DMRG backend remains unchanged.

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

topology_size(topology::AbstractStructuredTopology) = topology.m * topology.n * topology.spins_per_site
topology_tuple(topology::AbstractStructuredTopology) = (topology.m, topology.n, topology.spins_per_site)
topology_name(::SquareGrid) = "square"
topology_name(::KingGrid) = "king"

"""
    PEPSBackend(topology)

Select the optional SpinGlassPEPS structured backend for `minimize` and
`TenSolver.solve_ising`, targeting Ising models whose couplings fit the given
structured `topology` (`SquareGrid` or `KingGrid`).

Like `DMRGBackend`, the backend object only selects the algorithm; solver
parameters are keyword arguments of the solve call. Keywords shared with the
DMRG backend keep the same names (`maxdim`, `iterations`, `device`); the
PEPS-specific keywords are documented on [`TenSolver.peps_options`](@ref).

The solve is provided by the `TenSolverSpinGlassPEPSExt` package extension and
requires `SpinGlassNetworks`, `SpinGlassEngine`, and `SpinGlassTensors` to be
installed and loaded; without them, solves with this backend error clearly and
the default DMRG backend remains unchanged.

This backend is experimental scaffolding: it is intentionally not exported
while the registered SpinGlass component dependency stack does not resolve in
the same environment as TenSolver's ITensors/QUBOTools stack, so CI cannot
exercise the extension.
"""
struct PEPSBackend{T <: AbstractStructuredTopology} <: AbstractTenSolverBackend
  topology :: T
end

"""
    peps_options(; kwargs...)

Normalize and validate the keyword arguments of a PEPS solve into an options
named tuple for the SpinGlassPEPS extension.

Keywords shared with the DMRG backend:

- `maxdim :: Int = 16` - Maximum boundary-MPS bond dimension used during PEPS contraction.
- `iterations :: Int = 1` - Number of boundary-MPS variational sweeps.
- `device :: Function = cpu` - Accelerator device; anything other than `cpu` runs the SpinGlass contractor on GPU.

PEPS-specific keywords:

- `beta :: Float64 = 2.0` - Inverse temperature used for the contraction.
- `max_states :: Int = 256` - Maximum number of low-energy states retained.
- `cutoff_prob :: Float64 = 1e-4` - Probability cutoff for discarding branches.
- `contraction :: Symbol = :auto` - Contraction strategy (`:auto`, `:svd`, `:svd_truncate`, or `:zipper`).
- `graduate_truncation :: Bool = true` - Gradual truncation of the boundary MPS.
- `transformations = :all` - Lattice transformations to try (`:all`, `:identity`, or a collection).
- `local_dimension :: Union{Nothing, Int} = nothing` - Optional fixed Potts cluster dimension.
- `no_cache :: Bool = false` - Disable the SpinGlassEngine memoization cache.
"""
function peps_options(;
  maxdim::Integer = 16,
  iterations::Integer = 1,
  device::Function = cpu,
  beta::Real = 2.0,
  max_states::Integer = 2^8,
  cutoff_prob::Real = 1e-4,
  contraction::Symbol = :auto,
  graduate_truncation::Bool = true,
  transformations = :all,
  local_dimension::Union{Nothing, Integer} = nothing,
  no_cache::Bool = false,
)
  beta > 0 && isfinite(beta) || throw(ArgumentError("PEPS solves require finite beta > 0. Got $beta."))
  maxdim >= 1 || throw(ArgumentError("PEPS solves require maxdim >= 1. Got $maxdim."))
  max_states >= 1 || throw(ArgumentError("PEPS solves require max_states >= 1. Got $max_states."))
  cutoff_prob >= 0 || throw(ArgumentError("PEPS solves require cutoff_prob >= 0. Got $cutoff_prob."))
  iterations >= 1 || throw(ArgumentError("PEPS solves require iterations >= 1. Got $iterations."))
  contraction in (:auto, :svd, :svd_truncate, :zipper) ||
    throw(ArgumentError("Unsupported PEPS contraction $(repr(contraction)). Use :auto, :svd, :svd_truncate, or :zipper."))
  if !isnothing(local_dimension)
    local_dimension >= 1 || throw(ArgumentError("PEPS solves require local_dimension >= 1 when provided. Got $local_dimension."))
  end

  return (;
    maxdim = Int(maxdim),
    iterations = Int(iterations),
    onGPU = !(device === cpu),
    beta = Float64(beta),
    max_states = Int(max_states),
    cutoff_prob = Float64(cutoff_prob),
    contraction,
    graduate_truncation,
    transformations,
    local_dimension = isnothing(local_dimension) ? nothing : Int(local_dimension),
    no_cache,
  )
end

backend_error(::PEPSBackend) = ArgumentError("PEPSBackend is not available. Install/load SpinGlassNetworks, SpinGlassEngine, and SpinGlassTensors to activate the PEPS extension, or use backend = :dmrg.")

# Internal result scaffold for the optional SpinGlassPEPS extension.
#
# This type is not exported while the optional SpinGlassPEPS component stack is
# not covered by CI. The states are decoded to TenSolver Boolean vectors, and
# `energies` are objective values in the original TenSolver convention.
# Backend-specific diagnostics live in `metadata` and the raw extension result
# is stored in `raw`.
struct PEPSSolution{T <: Real} <: AbstractSolution
    states        :: Vector{Vector{Int}}
    energies      :: Vector{T}
    probabilities :: Vector{T}
    metadata      :: Dict{String, Any}
    raw           :: Any

  function PEPSSolution{T}(states, energies, probabilities, metadata, raw) where {T <: Real}
    # The extension merges duplicate decoded states (summing their
    # probabilities) before construction; enforce that invariant here so each
    # retained state carries exactly one probability.
    allunique(states) || throw(ArgumentError("PEPS solution states must be unique. Merge duplicate states before constructing a PEPSSolution."))
    return new{T}(states, energies, probabilities, metadata, raw)
  end
end

function sample(psi::PEPSSolution)
  isempty(psi.states) && throw(ArgumentError("Cannot sample an empty PEPS solution."))
  if isempty(psi.probabilities)
    return copy(first(psi.states))
  end
  length(psi.probabilities) == length(psi.states) ||
    throw(ArgumentError("PEPS solution probabilities must match the number of retained states."))
  any(probability -> probability < 0, psi.probabilities) &&
    throw(ArgumentError("PEPS solution probabilities must be nonnegative."))

  total = sum(psi.probabilities)
  total > 0 || throw(ArgumentError("PEPS solution probabilities must have positive total weight."))

  threshold = rand() * total
  cumulative = zero(total)
  for (state, probability) in zip(psi.states, psi.probabilities)
    cumulative += probability
    if threshold <= cumulative
      return copy(state)
    end
  end

  return copy(last(psi.states))
end

function prob(psi::PEPSSolution{T}, bs) where {T}
  target = collect(Int, bs)
  index = findfirst(==(target), psi.states)
  return isnothing(index) ? zero(T) : get(psi.probabilities, index, zero(T))
end

function minimize(
  backend::PEPSBackend,
  Q::AbstractMatrix{T},
  l::Union{AbstractVector{T}, Nothing}=nothing,
  c::T=zero(T)
  ;
  cutoff=1e-8,
  preprocess::Bool=false,
  kwargs...,
) where T
  preprocess && throw(ArgumentError("PEPSBackend does not support preprocess=true because the topology fixes the variable order."))
  return solve_ising(backend, IsingModel(qubo_to_ising(Q, l, c)); cutoff, kwargs...)
end

function minimize(backend::PEPSBackend, p::AbstractPolynomial; kwargs...)
  throw(ArgumentError("PEPSBackend does not support polynomial inputs directly. Convert to a structured QUBO or call solve_ising with a supported topology."))
end
