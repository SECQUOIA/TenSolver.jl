import ITensors, ITensorMPS
import ITensorMPS: MPS, siteinds

"""
    Solution

The result of running [`minimize`](@ref) or [`maximize`](@ref): an MPS wave function
over the optimal solution space, together with per-iteration convergence stats.

Use [`sample`](@ref) to draw bitstrings from it.

## Fields

- `tensor`: the underlying MPS, or `nothing` when the model is infeasible.
- `energies`: expected objective value of the problem recorded at each iteration of the solver.
- `bond_dims`: maximum MPS bond dimension at each iteration.
- `elapsed_times`: wall-clock time in seconds from the start of the solve at each iteration.
- `permutation`: original variable index represented by each tensor site.

The three stats vectors are parallel — `energies[i]`, `bond_dims[i]`, and `elapsed_times[i]`
all correspond to iteration `i`.

Provably infeasible models produce a `Solution` with no MPS and empty stats
vectors; check with [`is_feasible`](@ref) before sampling.
"""
struct Solution{T <: Real}
    tensor        :: Union{MPS, Nothing}
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

infeasible_solution(::Type{T}) where {T <: Real} =
  Solution{T}(nothing, T[], Int[], Float64[], Int[])

"""
    is_feasible(psi::Solution)

Whether `psi` came from solving a provably infeasible model, i.e. one whose
constraints admit no binary vector. Infeasible solutions carry no MPS and
cannot be sampled; [`minimize`](@ref) reports them with an objective of `+Inf`
(`-Inf` for [`maximize`](@ref)).

Whether `psi` came from solving a satisfiable model, i.e. one whose
constraints admit at least one binary vector.
Feasible solutions carry an MPS and can be sampled;

Check this before calling [`sample`](@ref).
"""
is_feasible(psi::Solution) = !isnothing(psi.tensor)

function original_order(bs, permutation)
  x = similar(bs)
  x[permutation] = bs
  return x
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

Throw a `DomainError` when `psi` is infeasible (see [`is_feasible`](@ref)),
since there is no solution to query.
"""
function sample(psi::Solution)
  if is_feasible(psi)
    bs = ITensorMPS.sample!(psi.tensor) .- 1
    return original_order(bs, psi.permutation)
  else
    throw(DomainError("the model is infeasible; there is no solution to sample"))
  end
end

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
Always `false` for infeasible solutions.
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

function prob(psi::Solution{T}, bs) where {T}
  return is_feasible(psi) ? abs2(coeff(psi, bs)) : zero(T)
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
