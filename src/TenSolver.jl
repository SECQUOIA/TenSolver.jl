module TenSolver

import ITensors, ITensorMPS
using QUBODrivers: QUBODrivers, QUBOTools, MOI

using LinearAlgebra

include("solution.jl")
export sample

include("solver.jl")
export minimize, maximize

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
  end
end

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
  results = @timed minimize(Q, l, b;
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
  energy, psi = results.value
  obj = a * energy

  # ~ Samples and Output ~ #
  samples = Vector{QUBOTools.Sample{T,Int}}(undef, num_reads)
  for i in 1:num_reads
    x = sample(psi)
    E = QUBOTools.value(x, l, Q, a, b)

    samples[i] = QUBOTools.Sample{T,Int}(x, E)
  end

  # ~ Metadata ~ #
  metadata = Dict{String,Any}(
      "origin" => "TenSolver",
      "time"   => Dict{String,Any}(
          "effective" => results.time,
      ),
  )

  return QUBOTools.SampleSet{T,Int}(samples, metadata; sense = :min, domain = :bool)
end

end # module TenSolver
