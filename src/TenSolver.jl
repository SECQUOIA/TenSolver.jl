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
export SumConstraint, NotEqualsConstraint, ExactlyOneConstraint, RelationConstraint
export is_feasible

include("projection_mpo.jl")

include("solution.jl")
export sample
export Solution, PEPSSolution

include("ising.jl")
export IsingModel, bool_to_spin, spin_to_bool, qubo_to_ising, ising_to_qubo, ising_energy

include("solver.jl")
export minimize, maximize, solve_ising
export AbstractTenSolverBackend, DMRGBackend, PEPSBackend, SquareGrid, KingGrid

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
  end
end

QUBODrivers.honors_final_reads(::Type{<:Optimizer}) = true
QUBODrivers.enforces_time_limit(::Type{<:Optimizer}) = true

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
  results = @timed minimize(Q, l, b;
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
  _, psi = results.value

  # ~ Samples and Output ~ #
  # Infeasible models have no samples; they are reported through the
  # termination status alone, like other MOI solvers.
  reads = is_feasible(psi) ? final_num_reads : 0
  samples = Vector{QUBOTools.Sample{T,Int}}(undef, reads)
  for i in 1:reads
    x = sample(psi)
    E = QUBOTools.value(x, l, Q, a, b)

    samples[i] = QUBOTools.Sample{T,Int}(x, E)
  end

  # ~ Metadata ~ #
  metadata = tensolver_metadata(
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

  return QUBOTools.SampleSet{T}(samples, metadata; sense = :min, domain = :bool)
end

function tensolver_metadata(
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
      "sweep_elapsed" => copy(solution.elapsed_times),
      "sweep_times"   => sweep_times(solution.elapsed_times),
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

function tensolver_status(solution::Solution; iterations::Integer, time_limit::Real)
  elapsed_time = isempty(solution.elapsed_times) ? 0.0 : last(solution.elapsed_times)
  if !is_feasible(solution)
    return MOI.INFEASIBLE, "infeasible"
  elseif length(solution.energies) >= iterations
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
