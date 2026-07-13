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
vectors; check with [`is_infeasible`](@ref) before sampling.
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
    is_infeasible(psi::Solution)

Whether `psi` came from solving a provably infeasible model, i.e. one whose
constraints admit no binary vector. Infeasible solutions carry no MPS and
cannot be sampled; [`minimize`](@ref) reports them with an objective of `+Inf`
(`-Inf` for [`maximize`](@ref)).
"""
is_infeasible(psi::Solution) = isnothing(psi.tensor)

function original_order(bs, permutation)
  x = similar(bs)
  x[permutation] = bs
  return x
end

# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample(psi)

Sample a bitstring from a (quantum) probability distribution.

Throws an `ArgumentError` when `psi` is infeasible (see [`is_infeasible`](@ref)):
the solve itself reports infeasibility as a status, but there is no solution to
query.
"""
function sample(psi::Solution)
  is_infeasible(psi) && throw(ArgumentError("the model is infeasible; there is no solution to sample"))
  bs = ITensorMPS.sample!(psi.tensor) .- 1
  return original_order(bs, psi.permutation)
end

sample(psi::Solution, n :: Integer) = [sample(psi) for _ in 1:n]

"""
    in(xs, psi::Solution [; cutoff)

Whether the vector `xs` has a positive probability of being sampleable from `psi`.
When setting `cutoff`, it will be used as the minimum probability considered positive.
Always `false` for infeasible solutions.
"""
function Base.in(bs, psi::Solution; cutoff = 1e-8)
  is_infeasible(psi) && return false
  return prob(psi, bs) > cutoff
end

function prob(psi::Solution, bs)
  return abs2(coeff(psi, bs))
end

function coeff(psi::Solution, bs)
  tn    = psi.tensor
  sites = siteinds(tn)
  bs    = bs[psi.permutation]
  psi0  = MPS(sites, string.(Int.(bs)))

  return inner(psi0,  tn)
end
