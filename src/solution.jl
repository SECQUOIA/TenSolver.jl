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

The three stats vectors are parallel — `energies[i]`, `bond_dims[i]`, and `elapsed_times[i]`
all correspond to iteration `i`.
"""
struct Solution{T <: Real}
    tensor        :: MPS
    energies      :: Vector{T}
    bond_dims     :: Vector{Int}
    elapsed_times :: Vector{Float64}
end

"""
    PEPSSolution

Result returned by the optional SpinGlassPEPS backend.

The states are decoded to TenSolver Boolean vectors, and `energies` are
objective values in the original TenSolver convention. Backend-specific
diagnostics live in `metadata` and the raw extension result is stored in `raw`.
"""
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
sample(psi::Solution) = ITensorMPS.sample!(psi.tensor) .- 1

sample(psi::Solution, n :: Integer) = [sample(psi) for _ in 1:n]

function sample(psi::PEPSSolution)
  isempty(psi.states) && throw(ArgumentError("Cannot sample an empty PEPS solution."))
  idx = isempty(psi.probabilities) ? firstindex(psi.states) : argmax(psi.probabilities)
  return copy(psi.states[idx])
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

"""
    in(xs, psi::PEPSSolution [; cutoff])

Whether `xs` is one of the decoded Boolean states retained by the PEPS backend.
"""
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
  psi0  = MPS(sites, string.(Int.(bs)))

  return inner(psi0,  tn)
end
