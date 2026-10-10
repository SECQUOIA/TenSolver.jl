"""
    Solution

Supertype for backend solution distributions returned by [`minimize`](@ref)
and [`maximize`](@ref).

Each backend provides its own concrete subtype. Subtypes must implement:

- `sample(psi)`: draw one vector from the distribution.
- `prob(psi, xs)`: probability of drawing the vector `xs`.
- `is_feasible(psi)`: whether the solver found a feasible solution.

Subtypes inherit `sample(psi, n)` and `in(xs, psi; cutoff)` from those methods.
"""
abstract type Solution end

sample(psi::Solution, n::Integer) = [sample(psi) for _ in 1:n]

"""
    in(xs, psi::Solution [; cutoff])

Whether the vector `xs` has a positive probability of being sampleable from
`psi`. When setting `cutoff`, it will be used as the minimum probability
considered positive. Always `false` for infeasible solutions.
"""
function Base.in(bs::AbstractVector, psi::Solution; cutoff = 1e-8)
  return is_feasible(psi) && prob(psi, bs) > cutoff
end

"""
    SolverStatistics{T}

## Fields

- `energies`: per-iteration calculated energy/objective value;
- `bond_dims`: per-iteration solution maximum bond dimension;
- `elapsed_times`: per-iteration total until the iteration completed;
- `variances`: per-iteration Hamiltonian variance, or `nothing` on iterations
  where the variance was not checked;
- `max_bonds`: structure containing bond dimensions for multiple tensors used
  throughout the iteration
  - `initial_state`: bond for the initial MPS guess;
  - `objective`: bond for the MPO representing the objective function `H`;
  - `projections`: bonds for the MPOs representing each constraint;
  - `hamiltonian`: bond for the MPO representing the actual Hamiltonian
    `P'HP` representing both objective and constraints;
"""
struct SolverStatistics{T <: Real}
  energies      :: Vector{T}
  bond_dims     :: Vector{Int64}
  elapsed_times :: Vector{Float64}
  variances     :: Vector{Union{Nothing,T}}
  max_bonds     :: @NamedTuple begin
    projections   :: Vector{Int64}
    objective     :: Int64
    initial_state :: Int64
    hamiltonian   :: Int64
  end

  function SolverStatistics{T}(; projections, objective, initial_state, hamiltonian) where {T}
    new{T}(
      T[],
      Int64[],
      Float64[],
      Union{Nothing,T}[],
      (; projections, objective, initial_state, hamiltonian),
    )
  end
end

function record_stats!(stats::SolverStatistics; energy, bond_dim, elapsed_time, variance)
  push!(stats.energies,      energy)
  push!(stats.bond_dims,     bond_dim)
  push!(stats.elapsed_times, elapsed_time)
  push!(stats.variances,     variance)

  return stats
end
