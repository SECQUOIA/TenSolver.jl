import JuMP: MOI
import TOML

function qubodrivers_test_model(; iterations = 1, time_limit = nothing)
  model = MOI.instantiate(TenSolver.Optimizer; with_bridge_type = Float64)
  x, _ = MOI.add_constrained_variables(model, fill(MOI.ZeroOne(), 2))
  f = MOI.ScalarQuadraticFunction{Float64}(
    MOI.ScalarQuadraticTerm{Float64}[
      MOI.ScalarQuadraticTerm{Float64}(2.0, x[1], x[1]),
      MOI.ScalarQuadraticTerm{Float64}(-4.0, x[1], x[2]),
      MOI.ScalarQuadraticTerm{Float64}(2.0, x[2], x[2]),
    ],
    MOI.ScalarAffineTerm{Float64}[],
    0.0,
  )

  MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
  MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
  MOI.set(model, MOI.RawOptimizerAttribute("iterations"), iterations)
  MOI.set(model, MOI.RawOptimizerAttribute("verbosity"), 0)
  if !isnothing(time_limit)
    MOI.set(model, MOI.TimeLimitSec(), time_limit)
  end

  return model
end

function single_variable_test_model()
  model = MOI.instantiate(TenSolver.Optimizer; with_bridge_type = Float64)
  x, _ = MOI.add_constrained_variables(model, [MOI.ZeroOne()])
  f = MOI.ScalarQuadraticFunction{Float64}(
    MOI.ScalarQuadraticTerm{Float64}[
      MOI.ScalarQuadraticTerm{Float64}(-1.0, x[1], x[1]),
    ],
    MOI.ScalarAffineTerm{Float64}[],
    0.0,
  )

  MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
  MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
  MOI.set(model, MOI.RawOptimizerAttribute("iterations"), 10)
  MOI.set(model, MOI.RawOptimizerAttribute("verbosity"), 0)

  return model
end

function model_solution(model)
  raw = MOI.get(model, MOI.RawSolver())

  return TenSolver.QUBOTools.solution(raw)
end

function model_metadata(model)
  return TenSolver.QUBOTools.metadata(model_solution(model))
end

@testset "QUBODrivers.jl" begin
  import QUBODrivers

  @test !QUBODrivers.supports_seed(TenSolver.Optimizer)
  @test QUBODrivers.honors_final_reads(TenSolver.Optimizer)
  @test QUBODrivers.enforces_time_limit(TenSolver.Optimizer)

  model = qubodrivers_test_model()
  MOI.set(model, QUBODrivers.FinalNumberOfReads(), 3)
  MOI.optimize!(model)

  sampleset = model_solution(model)
  metadata  = model_metadata(model)

  @test isempty(QUBODrivers.validate_metadata(sampleset))
  @test length(sampleset) <= 3
  @test sum(TenSolver.QUBOTools.reads(sample) for sample in sampleset) == 3
  @test metadata["origin"] == "TenSolver.jl"
  @test metadata["algorithm"]["name"] == "DMRG"
  @test metadata["backend"]["name"] == "TenSolver"
  @test metadata["backend"]["version"] == TenSolver.__VERSION__
  @test metadata["optimizer"]["evaluations"] === nothing
  @test metadata["reads"]["number_of_reads"] == 1_000
  @test metadata["reads"]["final_number_of_reads"] == 3
  @test metadata["status"] == "iteration_limit"
  @test metadata["termination_status"] == MOI.ITERATION_LIMIT
  @test metadata["time"]["effective"] > 0.0
  @test metadata["tensolver"]["dmrg"]["sweep_elapsed"] isa Vector{Float64}
  @test metadata["tensolver"]["dmrg"]["sweep_times"] isa Vector{Float64}
  @test metadata["tensolver"]["parameters"]["maxdim"] == [10, 20, 50, 100, 100, 200]

  time_limit_model = qubodrivers_test_model(; iterations = 10, time_limit = 1e-12)
  MOI.optimize!(time_limit_model)
  time_limit_metadata = model_metadata(time_limit_model)
  @test isempty(QUBODrivers.validate_metadata(time_limit_metadata))
  @test time_limit_metadata["status"] == "time_limit"
  @test time_limit_metadata["termination_status"] == MOI.TIME_LIMIT

  local_solve_model = single_variable_test_model()
  MOI.optimize!(local_solve_model)
  local_solve_metadata = model_metadata(local_solve_model)
  @test isempty(QUBODrivers.validate_metadata(local_solve_metadata))
  @test local_solve_metadata["status"] == "locally_solved"
  @test local_solve_metadata["termination_status"] == MOI.LOCALLY_SOLVED

  # QUBODrivers accepts raw attribute values, so TenSolver validates that read
  # counts are non-negative before generating the final SampleSet.
  negative_reads_model = qubodrivers_test_model()
  MOI.set(negative_reads_model, TenSolver.NumberOfReads(), -1)
  negative_reads_error = try
    MOI.optimize!(negative_reads_model)
  catch err
    err
  end
  @test negative_reads_error isa ErrorException
  @test occursin(
    "Number of reads must be a non-negative integer",
    sprint(showerror, negative_reads_error),
  )

  QUBODrivers.test(TenSolver.Optimizer; benchmark_conformance = true) do model
    MOI.set(model, MOI.RawOptimizerAttribute("iterations"), 1)
    MOI.set(model, MOI.RawOptimizerAttribute("verbosity"), 0)
  end
end

@testset "QUBO ecosystem compat" begin
  compat = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))["compat"]

  @test compat["QUBODrivers"] == "0.6.1"
  @test compat["QUBOTools"] == "0.13, 0.14, 0.15, 0.16"
end

@testset "Aqua.jl" begin
  import Aqua

  Aqua.test_all(
    TenSolver;
    ambiguities = (exclude=[MOI.supports],) ,
    piracies=(treat_as_own=[TenSolver.ITensors.state],)
  )
end
