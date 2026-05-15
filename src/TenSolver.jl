module TenSolver

import ITensors, ITensorMPS
using QUBODrivers: QUBODrivers, QUBOTools, MOI

using LinearAlgebra

include("solution.jl")
export sample
export Solution

include("ising.jl")
export IsingModel, bool_to_spin, spin_to_bool, qubo_to_ising, ising_to_qubo, ising_energy

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
  version = v"0.1.0"
  attributes = begin
    # JuMP-specific
    NumberOfReads["num_reads"]::Integer = 1_000
    # Solver keywords
    "cutoff"               :: Float64                         = 1e-8
    "device"               :: Function                        = cpu
    "vtol"                 :: Float64                         = 0.0
    "iterations"           :: Int                             = 10
    "time_limit"           :: Float64                         = +Inf
    "maxdim"               :: Union{Int, Vector{Int}}         = [10, 20, 50, 100, 100, 200]
    "mindim"               :: Union{Int, Vector{Int}}         = 1
    "noise"                :: Union{Float64, Vector{Float64}} = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 0.0]
    "eigsolve_krylovdim"   :: Int                             = 3
    "eigsolve_maxiter"     :: Int                             = 1
    "eigsolve_tol"         :: Float64                         = 1e-14
    "preprocess"           :: Bool                            = false
    "verbosity"            :: Int                             = 1
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
  topology isa SquareGrid && return topology
  topology isa KingGrid && return topology

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

function _qubo_samples(::Type{T}, psi::PEPSSolution, l, Q, a, b, num_reads::Integer) where {T}
  states = psi.states
  isempty(states) && throw(ArgumentError("Cannot build QUBOTools samples from an empty PEPS solution."))

  samples = Vector{QUBOTools.Sample{T,Int}}(undef, num_reads)
  for i in 1:num_reads
    x = states[mod1(i, length(states))]
    E = QUBOTools.value(x, l, Q, a, b)

    samples[i] = QUBOTools.Sample{T,Int}(x, E)
  end

  return samples
end

function _add_backend_metadata!(metadata::Dict{String,Any}, psi::PEPSSolution)
  peps = copy(psi.metadata)
  peps["candidate_states"] = length(psi.states)
  peps["effective_time"] = metadata["time"]["effective"]
  metadata["backend"] = get(peps, "backend", "SpinGlassPEPS")
  metadata["peps"] = peps
  return metadata
end

_add_backend_metadata!(metadata::Dict{String,Any}, psi) = metadata

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
  # ~ Manage Attributes ~ #
  get(attr) = MOI.get(sampler, MOI.RawOptimizerAttribute(attr))

  if MOI.get(sampler, MOI.TimeLimitSec()) !== nothing
    MOI.set(sampler, MOI.RawOptimizerAttribute("time_limit"), MOI.get(sampler, MOI.TimeLimitSec()))
  end

  if MOI.get(sampler, MOI.Silent())
    MOI.set(sampler, MOI.RawOptimizerAttribute("verbosity"), 0)
  end

  num_reads = MOI.get(sampler, NumberOfReads())

  # ~ Solve ~ #
  n, l, Q, a, b = QUBOTools.qubo(sampler, :sparse; sense = :min)
  # min_x a*(x'Qx + l'x + b)
  #  s.t. x in {0, 1}^n
  backend = _optimizer_backend(get)
  results = if backend isa PEPSBackend
    @timed minimize(Q, l, b;
      backend,
      cutoff = get("cutoff"),
      verbosity = get("verbosity"),
    )
  else
    @timed minimize(Q, l, b;
      backend,
      cutoff      = get("cutoff"),
      vtol        = get("vtol"),
      iterations  = get("iterations"),
      time_limit  = get("time_limit"),
      maxdim      = get("maxdim"),
      mindim      = get("mindim"),
      noise       = get("noise"),
      device      = get("device"),
      verbosity   = get("verbosity"),
      eigsolve_krylovdim =  get("eigsolve_krylovdim"),
      eigsolve_tol       =  get("eigsolve_tol"),
      eigsolve_maxiter   =  get("eigsolve_maxiter"),
    )
  end
  energy, psi = results.value
  obj = a * energy

  # ~ Samples and Output ~ #
  samples = _qubo_samples(T, psi, l, Q, a, b, num_reads)

  # ~ Metadata ~ #
  metadata = Dict{String,Any}(
      "origin" => "TenSolver",
      "time"   => Dict{String,Any}(
          "effective" => results.time,
      ),
  )
  _add_backend_metadata!(metadata, psi)

  return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :bool)
end

end # module TenSolver
