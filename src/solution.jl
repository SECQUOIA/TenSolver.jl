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

# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample(psi)

Sample a bitstring from a (quantum) probability distribution.
"""
sample(psi::Solution) = ITensorMPS.sample!(psi.tensor) .- 1

sample(psi::Solution, n :: Integer) = [sample(psi) for _ in 1:n]

"""
    GTNSolution

Result wrapper for exact solution-space data returned by the optional
GenericTensorNetworks backend.

Unlike [`Solution`](@ref), this is not an MPS. It stores exact configurations
when the selected GTN property produces them, together with the raw backend
result and metadata.
"""
struct GTNSolution{T <: Real, C, R, F}
    objective :: T
    configs   :: C
    result    :: R
    property  :: Symbol
    metadata  :: Dict{String, Any}
    sampler   :: F
end

function GTNSolution(objective::T, configs, result, property::Symbol, metadata::Dict{String, Any}=Dict{String, Any}(); sampler=nothing) where {T <: Real}
    return GTNSolution{T, typeof(configs), typeof(result), typeof(sampler)}(objective, configs, result, property, metadata, sampler)
end

function sample(psi::GTNSolution)
    if psi.sampler !== nothing
        return psi.sampler()
    elseif psi.configs isa AbstractVector && !isempty(psi.configs)
        return rand(psi.configs)
    else
        throw(ArgumentError("GTNSolution with property `$(psi.property)` does not contain sampleable configurations."))
    end
end

sample(psi::GTNSolution, n::Integer) = [sample(psi) for _ in 1:n]

function Base.in(bs, psi::GTNSolution)
    return psi.configs isa AbstractVector && collect(Int, bs) in psi.configs
end

function _with_objective(psi::GTNSolution, objective)
    return GTNSolution(objective, psi.configs, psi.result, psi.property, psi.metadata; sampler=psi.sampler)
end

"""
    in(xs, psi::Solution [; cutoff)

Whether the vector `xs` has a positive probability of being sampleable from `psi`.
When setting `cutoff`, it will be used as the minimum probability considered positive.
"""
function Base.in(bs, psi::Solution; cutoff = 1e-8)
  return prob(psi, bs) > cutoff
end

function prob(psi::Solution, bs)
  return abs2(coeff(psi, bs))
end

function coeff(psi::Solution, bs)
  tn    = psi.tensor
  sites = siteinds(tn)
  psi0  = MPS(sites, string.(Int.(bs)))

  return inner(psi0,  tn)
end
