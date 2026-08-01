module TenSolver

import ITensors, ITensorMPS
using QUBODrivers: QUBODrivers, QUBOTools, MOI

using LinearAlgebra

const __VERSION__ = pkgversion(@__MODULE__)

include("preprocess.jl")

include("ising.jl")
export bool_to_spin, spin_to_bool, qubo_to_ising, ising_to_qubo

include("constraints.jl")
export AbstractConstraint
export SumConstraint, SumModConstraint, NotEqualsConstraint, AssignmentConstraint, RelationConstraint
export is_feasible

include("projection_mpo.jl")

include("solution.jl")
export sample

include("solver.jl")
export minimize, maximize
export AbstractTenSolverBackend, DMRGBackend

# Convergence logging
include("log.jl")

cpu = identity


## ~:~ Welcome to the QUBOVerse ~:~ ##
# The functions below allow us to solve QUBO JuMP models
# with the solvers in this package.

QUBODrivers.@setup Optimizer begin
  name    = "TenSolver"
  version = __VERSION__
  attributes = begin
    # JuMP-specific
    NumberOfReads["num_reads"] :: Integer = 1_000
    # Solver keywords
    Cutoff["cutoff"]                         :: Float64                         = 1e-8
    Device["device"]                         :: Function                        = cpu
    Vtol["vtol"]                             :: Float64                         = 0.0
    Iterations["iterations"]                 :: Int                             = 10
    TimeLimit["time_limit"]                  :: Float64                         = +Inf
    MaxDim["maxdim"]                         :: Union{Int, Vector{Int}}         = [10, 20, 50, 100, 100, 200]
    MinDim["mindim"]                         :: Union{Int, Vector{Int}}         = 1
    Noise["noise"]                           :: Union{Float64, Vector{Float64}} = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 0.0]
    EigsolveKrylovDim["eigsolve_krylovdim"]  :: Int                             = 3
    EigsolveMaxiter["eigsolve_maxiter"]      :: Int                             = 1
    EigsolveTol["eigsolve_tol"]              :: Float64                         = 1e-14
    Preprocess["preprocess"]                 :: Bool                            = false
    Verbosity["verbosity"]                   :: Int                             = 1
    # Backend selection
    "backend" :: Union{Symbol, String} = :dmrg
    # PEPS backend keywords
    "peps_topology"             :: Any                  = nothing
    "peps_layout"               :: Union{Symbol, String} = :square
    "peps_beta"                 :: Float64              = 2.0
    "peps_bond_dim"             :: Int                  = 16
    "peps_max_states"           :: Int                  = 256
    "peps_cutoff_prob"          :: Float64              = 1e-4
    "peps_device"               :: Function             = cpu
    "peps_strategy"             :: Union{Symbol, String} = :auto
    "peps_num_sweeps"           :: Int                  = 1
    "peps_graduate_truncation"  :: Bool                 = true
    "peps_transformations"      :: Any                  = :all
    "peps_local_dimension"      :: Any                  = nothing
    "peps_no_cache"             :: Bool                 = false
  end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true
QUBODrivers.enforces_time_limit(::Type{<:Optimizer}) = true

optimizer_symbol(value::Symbol, ::AbstractString) = Symbol(lowercase(String(value)))
optimizer_symbol(value::AbstractString, ::AbstractString) = Symbol(lowercase(strip(value)))

function optimizer_symbol(value, attribute::AbstractString)
  throw(ArgumentError("Optimizer attribute `$attribute` must be a Symbol or String. " *
                      "Got $(repr(value)).",),)
end

function peps_topology_tuple(topology)
  if isnothing(topology)
    throw(ArgumentError("PEPS backend requires `peps_topology`, for example `(m, n)` " *
                        "for a square or king grid.",),)
  elseif topology isa Tuple
    return topology
  elseif topology isa AbstractVector
    return Tuple(topology)
  end

  throw(ArgumentError("`peps_topology` must be a tuple/vector such as `(m, n)` or " *
                      "`(m, n, spins_per_site)`. Got $(repr(topology)).",),)
end

function peps_topology(layout, topology)
  topology isa AbstractStructuredTopology && return topology

  dimensions = peps_topology_tuple(topology)
  if !(length(dimensions) in (2, 3))
    throw(ArgumentError("`peps_topology` must have 2 or 3 entries. " *
                        "Got $(repr(topology)).",),)
  end

  layout = optimizer_symbol(layout, "peps_layout")
  layout === :square && return SquareGrid(dimensions...)
  layout === :king && return KingGrid(dimensions...)
  throw(ArgumentError("Unsupported `peps_layout` $(repr(layout)). " *
                      "Use :square or :king.",),)
end

function peps_local_dimension(local_dimension)
  isnothing(local_dimension) && return nothing
  local_dimension isa Integer && return Int(local_dimension)
  throw(ArgumentError("`peps_local_dimension` must be an integer or `nothing`. " *
                      "Got $(repr(local_dimension)).",),)
end

function optimizer_backend(get_attribute)
  value = get_attribute("backend")
  backend = optimizer_symbol(value, "backend")
  backend === :dmrg && return :dmrg
  backend === :peps && return PEPSBackend(
    peps_topology(get_attribute("peps_layout"), get_attribute("peps_topology")),
  )
  throw(ArgumentError(
    "Unsupported optimizer backend $(repr(value)). Use :dmrg or :peps.",
  ))
end

function peps_optimizer_parameters(get_attribute)
  return (
    maxdim              = get_attribute("peps_bond_dim"),
    iterations          = get_attribute("peps_num_sweeps"),
    device              = get_attribute("peps_device"),
    beta                = get_attribute("peps_beta"),
    max_states          = get_attribute("peps_max_states"),
    cutoff_prob         = get_attribute("peps_cutoff_prob"),
    contraction         = optimizer_symbol(get_attribute("peps_strategy"), "peps_strategy"),
    graduate_truncation = get_attribute("peps_graduate_truncation"),
    transformations     = get_attribute("peps_transformations"),
    local_dimension     = peps_local_dimension(get_attribute("peps_local_dimension")),
    no_cache            = get_attribute("peps_no_cache"),
  )
end

function boolean_peps_solution(solution::PEPSSolution{T}) where {T}
  states = spin_to_bool.(solution.states)
  return PEPSSolution{T}(
    states,
    copy(solution.energies),
    copy(solution.probabilities),
    copy(solution.metadata),
  )
end

function minimize_peps_qubo(
  backend::PEPSBackend,
  Q::AbstractMatrix,
  l::AbstractVector,
  c::Real;
  kwargs...,
)
  form = qubo_to_ising(Q, l, c)
  _, h, J, scale, offset, _, _ = form
  isone(scale) || error("Internal QUBO-to-Ising conversion returned a non-unit scale.")
  energy, solution = minimize(backend, J, h, offset; domain = [-1, 1], kwargs...)
  return energy, boolean_peps_solution(solution)
end

function qubo_samples(
  ::Type{T},
  solution::Solution,
  l,
  Q,
  scale,
  offset,
  num_reads,
) where {T}
  reads = is_feasible(solution) ? num_reads : 0
  samples = Vector{QUBOTools.Sample{T,Int}}(undef, reads)
  for i in eachindex(samples)
    state = Int.(sample(solution))
    energy = QUBOTools.value(state, l, Q, scale, offset)
    samples[i] = QUBOTools.Sample{T,Int}(state, energy)
  end

  return samples
end

function peps_read_counts(solution::PEPSSolution, num_reads::Integer)
  if num_reads < 0
    throw(ArgumentError("num_reads must be nonnegative. Got $num_reads."),)
  end

  counts = zeros(Int, length(solution.states))
  num_reads == 0 && return counts

  probabilities = solution.probabilities
  if isempty(probabilities)
    counts[firstindex(counts)] = num_reads
    return counts
  end

  weights = Float64.(probabilities) ./ Float64(sum(probabilities)) .* num_reads
  counts .= floor.(Int, weights)
  remaining = num_reads - sum(counts)
  fractions = weights .- counts
  order = sortperm(collect(eachindex(fractions)); by = i -> (-fractions[i], i))
  for i in Iterators.take(order, remaining)
    counts[i] += 1
  end

  return counts
end

function qubo_samples(
  ::Type{T},
  solution::PEPSSolution,
  l,
  Q,
  scale,
  offset,
  num_reads,
) where {T}
  counts = peps_read_counts(solution, num_reads)
  samples = QUBOTools.Sample{T,Int}[]
  sizehint!(samples, count(>(0), counts))

  for (state, reads) in zip(solution.states, counts)
    reads == 0 && continue
    energy = QUBOTools.value(state, l, Q, scale, offset)
    push!(samples, QUBOTools.Sample{T,Int}(copy(state), energy, reads))
  end

  return samples
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
  # ~ Manage Attributes ~ #
  get(attr) = MOI.get(sampler, MOI.RawOptimizerAttribute(attr))

  moi_time_limit = MOI.get(sampler, MOI.TimeLimitSec())
  time_limit = isnothing(moi_time_limit) ? get("time_limit") : moi_time_limit
  verbosity = MOI.get(sampler, MOI.Silent()) ? 0 : get("verbosity")

  num_reads = MOI.get(sampler, NumberOfReads())
  final_num_reads = MOI.get(sampler, QUBODrivers.FinalNumberOfReads())

  if num_reads < 0
    error("Number of reads must be a non-negative integer")
  end

  # ~ Solve ~ #
  n, l, Q, a, b = QUBOTools.qubo(sampler, :sparse; sense = :min)
  # min_x a*(x'Qx + l'x + b)
  #  s.t. x in {0, 1}^n
  backend = optimizer_backend(get)
  peps_parameters = backend isa PEPSBackend ? peps_optimizer_parameters(get) : nothing
  results = if backend isa PEPSBackend
    @timed minimize_peps_qubo(
      backend,
      Q,
      l,
      b;
      cutoff = get("cutoff"),
      preprocess = get("preprocess"),
      verbosity,
      peps_parameters...,
    )
  else
    @timed minimize(Q, l, b;
      backend,
      time_limit,
      verbosity,
      cutoff      = get("cutoff"),
      vtol        = get("vtol"),
      iterations  = get("iterations"),
      maxdim      = get("maxdim"),
      mindim      = get("mindim"),
      noise       = get("noise"),
      device      = get("device"),
      preprocess  = get("preprocess"),
      eigsolve_krylovdim =  get("eigsolve_krylovdim"),
      eigsolve_tol       =  get("eigsolve_tol"),
      eigsolve_maxiter   =  get("eigsolve_maxiter"),
    )
  end
  _, psi = results.value

  # ~ Samples and Output ~ #
  samples = qubo_samples(T, psi, l, Q, a, b, final_num_reads)

  # ~ Metadata ~ #
  metadata = if psi isa PEPSSolution
    tensolver_metadata(
      psi;
      effective_time = results.time,
      num_reads,
      final_num_reads,
      peps_parameters,
    )
  else
    tensolver_metadata(
      psi;
      effective_time  = results.time,
      num_reads,
      final_num_reads,
      time_limit,
      iterations      = get("iterations"),
      cutoff          = get("cutoff"),
      vtol            = get("vtol"),
      maxdim          = get("maxdim"),
    )
  end

  return QUBOTools.SampleSet{T}(samples, metadata; sense = :min, domain = :bool)
end

function tensolver_metadata(
  solution::PEPSSolution;
  effective_time::Real,
  num_reads::Integer,
  final_num_reads::Integer,
  peps_parameters::NamedTuple,
)
  algorithm_name = get(solution.metadata, "backend", "SpinGlassPEPS")
  metadata = QUBODrivers._sampler_metadata(
    origin                = "TenSolver.jl",
    algorithm_name        = algorithm_name,
    backend_name          = "TenSolver",
    backend_version       = __VERSION__,
    execution_mode        = "tensor_network_peps",
    optimizer_iterations  = peps_parameters.iterations,
    optimizer_evaluations = length(solution.states),
    number_of_reads       = num_reads,
    final_number_of_reads = final_num_reads,
    status                = "locally_solved",
    termination_status    = MOI.LOCALLY_SOLVED,
  )

  parameters = Dict{String,Any}(
    "beta"                => peps_parameters.beta,
    "bond_dim"            => peps_parameters.maxdim,
    "max_states"          => peps_parameters.max_states,
    "cutoff_prob"         => peps_parameters.cutoff_prob,
    "device"              => string(peps_parameters.device),
    "strategy"            => string(peps_parameters.contraction),
    "num_sweeps"          => peps_parameters.iterations,
    "graduate_truncation" => peps_parameters.graduate_truncation,
    "transformations"     => peps_parameters.transformations,
    "local_dimension"     => peps_parameters.local_dimension,
    "no_cache"            => peps_parameters.no_cache,
  )
  peps = copy(solution.metadata)
  peps["candidate_states"] = length(solution.states)
  peps["effective_time"] = effective_time
  peps["parameters"] = parameters

  metadata["time"] = Dict{String,Any}("effective" => effective_time)
  metadata["tensolver"] = Dict{String,Any}("peps" => peps)
  return metadata
end

function tensolver_metadata(
  solution::DMRGSolution;
  effective_time::Real,
  num_reads::Integer,
  final_num_reads::Integer,
  time_limit::Real,
  iterations::Integer,
  cutoff::Real,
  vtol::Real,
  maxdim,
)
  optimizer_iterations = length(solution.stats.energies)
  termination_status, status = tensolver_status(
    solution;
    iterations,
    time_limit,
  )
  metadata = QUBODrivers._sampler_metadata(
    origin                = "TenSolver.jl",
    algorithm_name        = "DMRG",
    backend_name          = "TenSolver",
    backend_version       = __VERSION__,
    execution_mode        = "tensor_network_dmrg",
    optimizer_iterations  = optimizer_iterations,
    optimizer_evaluations = nothing,
    number_of_reads       = num_reads,
    final_number_of_reads = final_num_reads,
    status                = status,
    termination_status    = termination_status,
  )
  metadata["time"] = Dict{String,Any}("effective" => effective_time)
  metadata["tensolver"] = Dict{String,Any}(
    "dmrg" => Dict{String,Any}(
      "sweep_elapsed" => copy(solution.stats.elapsed_times),
      "sweep_times"   => sweep_times(solution.stats.elapsed_times),
    ),
    "parameters" => Dict{String,Any}(
      "cutoff"     => cutoff,
      "vtol"       => vtol,
      "maxdim"     => maxdim isa AbstractVector ? copy(maxdim) : maxdim,
      "iterations" => iterations,
      "time_limit" => time_limit,
    ),
  )

  return metadata
end

function tensolver_status(
  solution::DMRGSolution;
  iterations::Integer,
  time_limit::Real,
)
  elapsed_time = isempty(solution.stats.elapsed_times) ? 0.0 : last(solution.stats.elapsed_times)
  if !is_feasible(solution)
    return MOI.INFEASIBLE, "infeasible"
  elseif length(solution.stats.energies) >= iterations
    return MOI.ITERATION_LIMIT, "iteration_limit"
  elseif isfinite(time_limit) && elapsed_time > time_limit
    return MOI.TIME_LIMIT, "time_limit"
  else
    return MOI.LOCALLY_SOLVED, "locally_solved"
  end
end

function sweep_times(elapsed_times::Vector{Float64})
  isempty(elapsed_times) && return Float64[]

  return diff(vcat(0.0, elapsed_times))
end

end # module TenSolver
