module TenSolver

import ITensors, ITensorMPS
using QUBODrivers: QUBODrivers, QUBOTools, MOI

using LinearAlgebra

const __VERSION__ = pkgversion(@__MODULE__)

include("solution.jl")
export sample

include("ising.jl")
export bool_to_spin, spin_to_bool, qubo_to_ising, ising_to_qubo

include("solver.jl")
export minimize, maximize, solve_ising
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
    "backend"              :: Union{Symbol, String}           = :dmrg
    # PEPS backend keywords
    "peps_topology"        :: Any                             = nothing
    "peps_layout"          :: Union{Symbol, String}           = :square
    "peps_beta"            :: Float64                         = 1.0
    "peps_bond_dim"        :: Int                             = 8
    "peps_max_states"      :: Int                             = 256
    "peps_cutoff_prob"     :: Float64                         = 0.0
    "peps_onGPU"           :: Bool                            = false
    "peps_strategy"        :: Union{Symbol, String}           = :svd
    "peps_num_sweeps"      :: Int                             = 1
    "peps_transformations" :: Any                             = :all
    "peps_truncation"      :: Any                             = nothing
  end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true
QUBODrivers.enforces_time_limit(::Type{<:Optimizer}) = true

function _optimizer_symbol(value::Symbol, attr::AbstractString)
  return Symbol(lowercase(String(value)))
end

function _optimizer_symbol(value::AbstractString, attr::AbstractString)
  return Symbol(lowercase(strip(value)))
end

function _optimizer_symbol(value, attr::AbstractString)
  throw(ArgumentError("Optimizer attribute `$attr` must be a Symbol or String. Got $(repr(value))."))
end

function _optimizer_backend(get)
  backend = _optimizer_symbol(get("backend"), "backend")
  backend === :dmrg && return :dmrg
  backend === :peps && return _optimizer_peps_backend(get)
  throw(ArgumentError("Unsupported optimizer backend $(repr(get("backend"))). Use :dmrg or :peps."))
end

function _peps_topology_tuple(topology)
  topology === nothing &&
    throw(ArgumentError("PEPS backend requires `peps_topology`, for example `(m, n)` for a square or king grid."))

  if topology isa Tuple
    return topology
  elseif topology isa AbstractVector
    return Tuple(topology)
  else
    throw(ArgumentError("`peps_topology` must be a tuple/vector such as `(m, n)` or `(m, n, spins_per_site)`. Got $(repr(topology))."))
  end
end

function _peps_topology(layout, topology)
  topology isa AbstractStructuredTopology && return topology

  dims = _peps_topology_tuple(topology)
  if !(length(dims) in (2, 3))
    throw(ArgumentError("`peps_topology` must have 2 or 3 entries. Got $(repr(topology))."))
  end

  layout = _optimizer_symbol(layout, "peps_layout")
  layout === :square && return SquareGrid(dims...)
  layout === :king && return KingGrid(dims...)
  throw(ArgumentError("Unsupported `peps_layout` $(repr(layout)). Use :square or :king."))
end

function _peps_truncation(truncation)
  truncation === nothing && return nothing
  truncation isa Integer && return Int(truncation)
  throw(ArgumentError("Only integer `peps_truncation` values are currently supported as local dimension limits. Got $(repr(truncation))."))
end

function _optimizer_peps_backend(get)
  return PEPSBackend(
    _peps_topology(get("peps_layout"), get("peps_topology"));
    beta = get("peps_beta"),
    bond_dim = get("peps_bond_dim"),
    max_states = get("peps_max_states"),
    cutoff_prob = get("peps_cutoff_prob"),
    onGPU = get("peps_onGPU"),
    contraction = _optimizer_symbol(get("peps_strategy"), "peps_strategy"),
    num_sweeps = get("peps_num_sweeps"),
    transformations = get("peps_transformations"),
    local_dimension = _peps_truncation(get("peps_truncation")),
  )
end

function _qubo_samples(::Type{T}, psi::Solution, l, Q, a, b, num_reads::Integer) where {T}
  samples = Vector{QUBOTools.Sample{T,Int}}(undef, num_reads)
  for i in 1:num_reads
    x = sample(psi)
    E = QUBOTools.value(x, l, Q, a, b)

    samples[i] = QUBOTools.Sample{T,Int}(x, E)
  end

  return samples
end

function _peps_read_counts(psi::PEPSSolution, num_reads::Integer)
  num_reads >= 0 || throw(ArgumentError("num_reads must be nonnegative. Got $num_reads."))

  states = psi.states
  isempty(states) && throw(ArgumentError("Cannot build QUBOTools samples from an empty PEPS solution."))

  counts = zeros(Int, length(states))
  num_reads == 0 && return counts

  probabilities = psi.probabilities
  if isempty(probabilities)
    counts[begin] = Int(num_reads)
    return counts
  end

  length(probabilities) == length(states) ||
    throw(ArgumentError("PEPS probabilities length must match states length. Got $(length(probabilities)) probabilities for $(length(states)) states."))
  any(p -> p < 0, probabilities) &&
    throw(ArgumentError("PEPS probabilities must be nonnegative. Got $(repr(probabilities))."))

  total = sum(probabilities)
  total > 0 || throw(ArgumentError("PEPS probabilities must have positive total weight. Got $(repr(probabilities))."))

  weights = (Float64.(probabilities) ./ Float64(total)) .* Int(num_reads)
  counts .= floor.(Int, weights)
  remaining = Int(num_reads) - sum(counts)

  if remaining > 0
    order = sortperm(collect(eachindex(weights)); by = i -> (weights[i] - counts[i], -i), rev = true)
    for i in Iterators.take(order, remaining)
      counts[i] += 1
    end
  end

  return counts
end

function _qubo_samples(::Type{T}, psi::PEPSSolution, l, Q, a, b, num_reads::Integer) where {T}
  counts = _peps_read_counts(psi, num_reads)
  samples = QUBOTools.Sample{T,Int}[]
  sizehint!(samples, count(>(0), counts))

  for (x, reads) in zip(psi.states, counts)
    reads == 0 && continue
    E = QUBOTools.value(x, l, Q, a, b)
    push!(samples, QUBOTools.Sample{T,Int}(copy(x), E, reads))
  end

  return samples
end

function _add_backend_metadata!(metadata::Dict{String,Any}, psi::PEPSSolution)
  peps = copy(psi.metadata)
  peps["candidate_states"] = length(psi.states)
  peps["effective_time"] = metadata["time"]["effective"]
  tensolver = get!(metadata, "tensolver", Dict{String,Any}())
  tensolver["peps"] = peps
  return metadata
end

_add_backend_metadata!(metadata::Dict{String,Any}, psi) = metadata

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
  backend = _optimizer_backend(get)
  results = if backend isa PEPSBackend
    @timed minimize(Q, l, b;
      backend,
      cutoff = get("cutoff"),
      verbosity,
    )
  else
    @timed minimize(Q, l, b;
      backend,
      cutoff      = get("cutoff"),
      vtol        = get("vtol"),
      iterations  = get("iterations"),
      time_limit,
      maxdim      = get("maxdim"),
      mindim      = get("mindim"),
      noise       = get("noise"),
      device      = get("device"),
      verbosity,
      eigsolve_krylovdim =  get("eigsolve_krylovdim"),
      eigsolve_tol       =  get("eigsolve_tol"),
      eigsolve_maxiter   =  get("eigsolve_maxiter"),
    )
  end
  _, psi = results.value

  # ~ Samples and Output ~ #
  samples = _qubo_samples(T, psi, l, Q, a, b, final_num_reads)

  # ~ Metadata ~ #
  metadata = _tensolver_metadata(
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
  _add_backend_metadata!(metadata, psi)

  return QUBOTools.SampleSet{T}(samples, metadata; sense = :min, domain = :bool)
end

function _tensolver_metadata(
  solution::Solution;
  effective_time::Real,
  num_reads::Integer,
  final_num_reads::Integer,
  time_limit::Real,
  iterations::Integer,
  cutoff::Real,
  vtol::Real,
  maxdim,
)
  optimizer_iterations = length(solution.energies)
  termination_status, status = _tensolver_status(
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
      "sweep_elapsed" => copy(solution.elapsed_times),
      "sweep_times"   => _sweep_times(solution.elapsed_times),
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

function _tensolver_metadata(
  solution::PEPSSolution;
  effective_time::Real,
  num_reads::Integer,
  final_num_reads::Integer,
  time_limit::Real,
  iterations::Integer,
  cutoff::Real,
  vtol::Real,
  maxdim,
)
  algorithm_name = get(solution.metadata, "backend", "SpinGlassPEPS")
  metadata = QUBODrivers._sampler_metadata(
    origin                = "TenSolver.jl",
    algorithm_name,
    backend_name          = "TenSolver",
    backend_version       = __VERSION__,
    execution_mode        = "tensor_network_peps",
    optimizer_iterations  = 1,
    optimizer_evaluations = length(solution.states),
    number_of_reads       = num_reads,
    final_number_of_reads = final_num_reads,
    status                = "locally_solved",
    termination_status    = MOI.LOCALLY_SOLVED,
  )
  metadata["time"] = Dict{String,Any}("effective" => effective_time)
  metadata["tensolver"] = Dict{String,Any}(
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

function _tensolver_status(solution::Solution; iterations::Integer, time_limit::Real)
  elapsed_time = isempty(solution.elapsed_times) ? 0.0 : last(solution.elapsed_times)
  if length(solution.energies) >= iterations
    return MOI.ITERATION_LIMIT, "iteration_limit"
  elseif isfinite(time_limit) && elapsed_time > time_limit
    return MOI.TIME_LIMIT, "time_limit"
  else
    return MOI.LOCALLY_SOLVED, "locally_solved"
  end
end

function _sweep_times(elapsed_times::Vector{Float64})
  isempty(elapsed_times) && return Float64[]

  return diff(vcat(0.0, elapsed_times))
end

end # module TenSolver
