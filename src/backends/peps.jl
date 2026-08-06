#----------------------------------------------------------#
# Optional SpinGlassPEPS structured backend                #
#----------------------------------------------------------#

abstract type AbstractStructuredTopology end

"""
    SquareGrid(m, n, spins_per_site = 1)

Structured square-grid topology for optional PEPS solves.

Variables are assumed to be ordered according to SpinGlassNetworks'
`super_square_lattice((m, n, spins_per_site))` convention.
"""
struct SquareGrid <: AbstractStructuredTopology
  m              :: Int
  n              :: Int
  spins_per_site :: Int

  function SquareGrid(m::Integer, n::Integer, spins_per_site::Integer = 1)
    if !(m > 0)
      throw(ArgumentError("SquareGrid requires m > 0. Got $m."))
    end
    if !(n > 0)
      throw(ArgumentError("SquareGrid requires n > 0. Got $n."))
    end
    if !(spins_per_site > 0)
      throw(ArgumentError("SquareGrid requires spins_per_site > 0. Got $spins_per_site.",),)
    end
    return new(Int(m), Int(n), Int(spins_per_site))
  end
end

"""
    KingGrid(m, n, spins_per_site = 1)

Structured king-grid topology for optional PEPS solves. It uses the same
variable ordering as [`SquareGrid`](@ref), but the PEPS compatibility graph
also allows diagonal interactions between neighboring grid cells.
"""
struct KingGrid <: AbstractStructuredTopology
  m              :: Int
  n              :: Int
  spins_per_site :: Int

  function KingGrid(m::Integer, n::Integer, spins_per_site::Integer = 1)
    if !(m > 0)
      throw(ArgumentError("KingGrid requires m > 0. Got $m."))
    end
    if !(n > 0)
      throw(ArgumentError("KingGrid requires n > 0. Got $n."))
    end
    if !(spins_per_site > 0)
      throw(ArgumentError("KingGrid requires spins_per_site > 0. Got $spins_per_site.",),)
    end
    return new(Int(m), Int(n), Int(spins_per_site))
  end
end

function topology_size(topology::AbstractStructuredTopology)
  return topology.m * topology.n * topology.spins_per_site
end
function topology_tuple(topology::AbstractStructuredTopology)
  return (topology.m, topology.n, topology.spins_per_site)
end
topology_name(::SquareGrid) = "square"
topology_name(::KingGrid) = "king"

"""
    PEPSBackend(topology)

Select the optional SpinGlassPEPS structured backend for Ising problems whose
couplings fit the given `topology` (`SquareGrid` or `KingGrid`).

The solve is provided by the `TenSolverSpinGlassPEPSExt` package extension and
requires `SpinGlassNetworks`, `SpinGlassEngine`, and `SpinGlassTensors` to be
installed and loaded; without them, solves with this backend error clearly.
"""
struct PEPSBackend{T<:AbstractStructuredTopology} <: AbstractTenSolverBackend
  topology::T
end

function backend_error(::PEPSBackend)
  return ArgumentError("PEPSBackend is not available. Install/load SpinGlassNetworks, " *
                       "SpinGlassEngine, and SpinGlassTensors to activate the PEPS extension.",)
end

# Internal result scaffold for the optional SpinGlassPEPS extension.
struct PEPSSolution{T<:Real} <: Solution
  states        :: Vector{Vector{Int}}
  energies      :: Vector{T}
  probabilities :: Vector{T}
  metadata      :: Dict{String,Any}

  function PEPSSolution{T}(
    states::Vector{Vector{Int}},
    energies::Vector{T},
    probabilities::Vector{T},
    metadata::Dict{String,Any},
  ) where {T<:Real}
    if isempty(states)
      throw(ArgumentError("PEPS solution states must not be empty."),)
    end
    if !(length(energies) == length(states))
      throw(ArgumentError("PEPS solution energies must match the number of retained states.",),)
    end
    if !(isempty(probabilities) || length(probabilities) == length(states))
      throw(ArgumentError("PEPS solution probabilities must match the number of retained states.",),)
    end
    if any(<(0), probabilities)
      throw(ArgumentError("PEPS solution probabilities must be nonnegative."),)
    end
    if (!isempty(probabilities) && !(sum(probabilities) > 0))
      throw(ArgumentError("PEPS solution probabilities must have positive total weight.",),)
    end
    if !(allunique(states))
      throw(ArgumentError("PEPS solution states must be unique. Merge duplicate states before " *
                          "constructing a PEPSSolution.",),)
    end

    state_length = length(first(states))
    if !(all(state -> length(state) == state_length, states))
      throw(ArgumentError("PEPS solution states must all have the same length."),)
    end

    return new{T}(states, energies, probabilities, metadata)
  end
end

is_feasible(::PEPSSolution) = true

function sample(psi::PEPSSolution)
  if isempty(psi.probabilities)
    return copy(first(psi.states))
  end

  threshold = rand() * sum(psi.probabilities)
  cumulative = zero(threshold)
  for (state, probability) in zip(psi.states, psi.probabilities)
    cumulative += probability
    threshold <= cumulative && return copy(state)
  end

  return copy(last(psi.states))
end

function prob(psi::PEPSSolution{T}, xs) where {T}
  state = collect(Int, xs)
  index = findfirst(==(state), psi.states)
  if isnothing(index)
    return zero(T)
  end
  if isempty(psi.probabilities)
    return index == firstindex(psi.states) ? one(T) : zero(T)
  end
  return psi.probabilities[index]
end
