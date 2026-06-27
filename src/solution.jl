import ITensors, ITensorMPS
import ITensorMPS: MPS, siteinds

"""
    Solution

The result of running [`minimize`](@ref) or [`maximize`](@ref): an MPS wave function
over the optimal solution space, together with per-iteration convergence stats.

Use [`sample`](@ref) to draw bitstrings from it.

## Fields

- `tensor`: the underlying MPS.
- `energies`: expected objective value of the problem recorded at each iteration of the solver.
- `bond_dims`: maximum MPS bond dimension at each iteration.
- `elapsed_times`: wall-clock time in seconds from the start of the solve at each iteration.
- `permutation`: original variable index represented by each tensor site.

The three stats vectors are parallel — `energies[i]`, `bond_dims[i]`, and `elapsed_times[i]`
all correspond to iteration `i`.
"""
struct Solution{T <: Real}
    tensor        :: MPS
    energies      :: Vector{T}
    bond_dims     :: Vector{Int}
    elapsed_times :: Vector{Float64}
    permutation   :: Vector{Int}
end

identity_permutation(tensor::MPS) = collect(1:length(siteinds(tensor)))

Solution{T}(
  tensor::MPS,
  energies::Vector{T},
  bond_dims::Vector{Int},
  elapsed_times::Vector{Float64},
) where {T <: Real} = Solution{T}(tensor, energies, bond_dims, elapsed_times, identity_permutation(tensor))

Solution(
  tensor::MPS,
  energies::Vector{T},
  bond_dims::Vector{Int},
  elapsed_times::Vector{Float64},
) where {T <: Real} = Solution{T}(tensor, energies, bond_dims, elapsed_times)

function original_order(bs, permutation)
  x = similar(bs)
  x[permutation] = bs
  return x
end

# Internal result scaffold for the optional SpinGlassPEPS extension.
#
# This type is not exported while the optional SpinGlassPEPS component stack is
# not covered by CI. The states are decoded to TenSolver Boolean vectors, and
# `energies` are objective values in the original TenSolver convention.
# Backend-specific diagnostics live in `metadata` and the raw extension result
# is stored in `raw`.
struct PEPSSolution{T <: Real}
    states        :: Vector{Vector{Int}}
    energies      :: Vector{T}
    probabilities :: Vector{T}
    metadata      :: Dict{String, Any}
    raw           :: Any
end

# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample(psi)

Sample a bitstring from a (quantum) probability distribution.
"""
function sample(psi::Solution)
  bs = ITensorMPS.sample!(psi.tensor) .- 1
  return original_order(bs, psi.permutation)
end

sample(psi::Solution, n :: Integer) = [sample(psi) for _ in 1:n]

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

sample(psi::PEPSSolution, n :: Integer) = [sample(psi) for _ in 1:n]

"""
    in(xs, psi::Solution [; cutoff)

Whether the vector `xs` has a positive probability of being sampleable from `psi`.
When setting `cutoff`, it will be used as the minimum probability considered positive.
"""
function Base.in(bs::AbstractVector, psi::Solution; cutoff = 1e-8)
  return prob(psi, bs) > cutoff
end

# Whether `xs` is one of the decoded Boolean states retained by the PEPS backend.
function Base.in(bs::AbstractVector, psi::PEPSSolution; cutoff = 1e-8)
  return prob(psi, bs) > cutoff
end

function prob(psi::Solution, bs)
  return abs2(coeff(psi, bs))
end

function prob(psi::PEPSSolution{T}, bs) where {T}
  target = collect(Int, bs)
  p = zero(T)
  for (state, probability) in zip(psi.states, psi.probabilities)
    if state == target
      p += probability
    end
  end
  return p
end

function coeff(psi::Solution, bs)
  tn    = psi.tensor
  sites = siteinds(tn)
  bs    = bs[psi.permutation]
  psi0  = MPS(sites, string.(Int.(bs)))

  return inner(psi0,  tn)
end
