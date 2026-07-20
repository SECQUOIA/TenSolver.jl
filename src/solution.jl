import ITensors, ITensorMPS
import ITensorMPS: MPS, siteinds

"""
    Solution

The result of running [`minimize`](@ref) or [`maximize`](@ref): an MPS wave function
over the optimal solution space, together with per-iteration convergence stats.

Use [`sample`](@ref) to draw vectors from it.

## Fields

- `tensor`: the underlying MPS, or `nothing` when the model is infeasible.
- `energies`: expected objective value of the problem recorded at each iteration of the solver.
- `bond_dims`: maximum MPS bond dimension at each iteration.
- `elapsed_times`: wall-clock time in seconds from the start of the solve at each iteration.
- `permutation`: original variable index represented by each tensor site.
- `domain`: physical variable values, ordered to match the local tensor basis.

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
    domain        :: Vector{T}

    function Solution{T}(
      tensor::Union{MPS,Nothing},
      energies::Vector{T},
      bond_dims::Vector{Int},
      elapsed_times::Vector{Float64},
      permutation::Vector{Int},
      domain,
    ) where {T <: Real}
      return new{T}(tensor, energies, bond_dims, elapsed_times, permutation, collect(T, domain))
    end
end

function infeasible_solution(::Type{T}, domain) where {T <: Real}
  return Solution{T}(nothing, T[], Int[], Float64[], Int[], domain)
end

"""
    is_feasible(psi::Solution)

Whether `psi` came from solving a satisfiable model, i.e. one whose
constraints admit at least one solution.
Feasible solutions carry an MPS and can be sampled;

Check this before calling [`sample`](@ref).
"""
is_feasible(psi::Solution) = !isnothing(psi.tensor)

original_order(bs, permutation) = bs[invperm(permutation)]

"""
    sample(psi)

Sample a vector from a (quantum) probability distribution.

Throw a `DomainError` when `psi` is infeasible (see [`is_feasible`](@ref)),
since there is no solution to query.
"""
function sample(psi::Solution)
  if is_feasible(psi)
    bs = psi.domain[ITensorMPS.sample!(psi.tensor)]
    return original_order(bs, psi.permutation)
  else
    throw(DomainError("the model is infeasible; there is no solution to sample"))
  end
end

sample(psi::Solution, n :: Integer) = [sample(psi) for _ in 1:n]

"""
    in(xs, psi::Solution [; cutoff)

Whether the vector `xs` has a positive probability of being sampleable from `psi`.
When setting `cutoff`, it will be used as the minimum probability considered positive.
Always `false` for infeasible solutions.
"""
function Base.in(bs, psi::Solution; cutoff = 1e-8)
  return prob(psi, bs) > cutoff
end

function prob(psi::Solution{T}, bs) where {T}
  return is_feasible(psi) ? abs2(coeff(psi, bs)) : zero(T)
end

function coeff(psi::Solution, bs)
  tn    = psi.tensor
  sites = siteinds(tn)
  bs    = bs[psi.permutation]
  positions = map(bs) do value
    position = findfirst(==(value), psi.domain)
    isnothing(position) && throw(DomainError(value, "value is outside the solution domain $(psi.domain)"))
    return position - 1
  end
  # Qudit state names are zero-based basis positions, not physical domain values.
  psi0  = MPS(sites, string.(positions))

  return inner(psi0,  tn)
end
