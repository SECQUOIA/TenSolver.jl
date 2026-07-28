import ITensors, ITensorMPS
import ITensorMPS: MPS, siteinds

"""
    SolverStatistics{T}

## Fields

- `energies`: per-iteration calculated energy/objective value;
- `bond_dims`: per-iteration solution maximum bond dimension;
- `elapsed_times`: per-iteration total until the iteration completed;
- `max_bonds`: structure containing bond dimensions for multiple tensors used throughout the iteration
  - `initial_state`: bond for the initial MPS guess;
  - `objective`: bond for the MPO representing the objective function `H`;
  - `projections`: bonds for the MPOs representing each constraint;
  - `hamiltonian`: bond for the MPO representing the actual Hamiltonian `P'HP` representing both objective and constraints;
"""
struct SolverStatistics{T <: Real}
  energies      :: Vector{T}
  bond_dims     :: Vector{Int64}
  elapsed_times :: Vector{Float64}
  max_bonds     :: @NamedTuple begin
    projections   :: Vector{Int64}
    objective     :: Int64
    initial_state :: Int64
    hamiltonian   :: Int64
  end

  function SolverStatistics{T}(; projections, objective, initial_state, hamiltonian) where {T}
    new{T}(T[], Int64[], Float64[], (; projections, objective, initial_state, hamiltonian))
  end
end

function record_stats!(stats::SolverStatistics, _i::Integer; energy, bond_dim, elapsed_time)
    push!(stats.energies,      energy)
    push!(stats.bond_dims,     bond_dim)
    push!(stats.elapsed_times, elapsed_time)

    return stats
end


"""
    Solution{T}

The result of running [`minimize`](@ref) or [`maximize`](@ref): an MPS wave function
over the optimal solution space, together with per-iteration convergence stats.

Use [`sample`](@ref) to draw vectors from it.

## Fields

- `tensor`: the underlying MPS, or `nothing` when the model is infeasible.
- `domain`: possible variable values.
- `permutation`: original variable index represented by each tensor site.
- `stats`: per-iteration convergence stats. See [`SolverStatistics`](@ref).

Provably infeasible models produce a `Solution` with no MPS and empty stats
vectors; check with [`is_feasible`](@ref) before sampling.
"""
struct Solution{T <: Real}
  tensor      :: Union{MPS, Nothing}
  domain      :: Vector{T}
  permutation :: Vector{Int}
  stats       :: SolverStatistics{T}

  function Solution{T}(
    tensor::Union{MPS,Nothing},
    domain,
    permutation::Vector{Int},
    stats::SolverStatistics,
  ) where {T <: Real}
    return new{T}(tensor, domain, permutation, stats)
  end
end

function infeasible_solution(::Type{T}, domain, stats) where {T <: Real}
  return Solution{T}(nothing, domain, Int[], stats)
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
    if isnothing(position)
      throw(DomainError(value, "value is outside the solution domain $(psi.domain)"))
    end
    return position - 1
  end
  # Qudit state names are zero-based basis positions, not physical domain values.
  psi0  = MPS(sites, string.(positions))

  return inner(psi0,  tn)
end
