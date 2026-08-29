import JuMP
import QUBODrivers
import QUBOTools

struct FakePEPSTopology <: TenSolver.AbstractStructuredTopology
  variables::Int
end

fake_peps_device(x) = x

function TenSolver.minimize(
  backend::TenSolver.PEPSBackend{FakePEPSTopology},
  J::AbstractMatrix{T},
  h::AbstractVector{T},
  offset::T;
  domain,
  cutoff,
  preprocess,
  verbosity,
  maxdim,
  iterations,
  device,
  beta,
  max_states,
  cutoff_prob,
  contraction,
  graduate_truncation,
  transformations,
  local_dimension,
  no_cache,
) where {T}
  @test backend.topology.variables == length(h)
  @test domain == [-1, 1]
  @test J == zeros(2, 2)
  @test h ≈ [-0.5, -1.0]
  @test offset ≈ -1.5
  @test cutoff == 1e-8
  @test !preprocess
  @test verbosity == 0
  @test maxdim == 3
  @test iterations == 2
  @test device === fake_peps_device
  @test beta == 1.75
  @test max_states == 4
  @test cutoff_prob == 0.0
  @test contraction == :svd
  @test !graduate_truncation
  @test transformations == :identity
  @test local_dimension == 2
  @test no_cache

  states = [[1, 1], [1, -1]]
  energies = [-3.0, -1.0]
  probabilities = [0.8, 0.2]
  metadata = Dict{String,Any}(
    "backend" => "FakePEPS",
    "topology" => "fake",
    "selected_transformation" => "identity",
    "largest_discarded_probability" => 0.0,
  )

  return first(energies),
         TenSolver.PEPSSolution{Float64}(states, energies, probabilities, metadata)
end

@testset "JuMP backend attributes" begin
  @testset "Default and explicit DMRG backends" begin
    for backend in (nothing, :dmrg, " DMRG ")
      model = JuMP.Model(TenSolver.Optimizer)
      JuMP.set_silent(model)
      if !isnothing(backend)
        JuMP.set_attribute(model, "backend", backend)
      end
      @JuMP.variable(model, x, Bin)
      @JuMP.objective(model, Min, -x)

      JuMP.optimize!(model)

      @test JuMP.objective_value(model) ≈ -1.0
      @test JuMP.value(x) ≈ 1.0
    end
  end

  @testset "Invalid backend selection" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "backend", :unknown)
    @JuMP.variable(model, x, Bin)
    @JuMP.objective(model, Min, -x)

    error = try
      JuMP.optimize!(model)
      nothing
    catch error
      error
    end

    @test error isa ArgumentError
    @test occursin("Use :dmrg or :peps", sprint(showerror, error))
  end

  @testset "PEPS requires valid topology metadata" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "backend", :peps)
    @JuMP.variable(model, x, Bin)
    @JuMP.objective(model, Min, -x)

    error = try
      JuMP.optimize!(model)
      nothing
    catch error
      error
    end

    @test error isa ArgumentError
    @test occursin("peps_topology", sprint(showerror, error))
    @test occursin("(m, n)", sprint(showerror, error))

    @test TenSolver.peps_topology(:square, [2, 3]) == TenSolver.SquareGrid(2, 3)
    @test TenSolver.peps_topology(" KING ", (2, 3, 2)) == TenSolver.KingGrid(2, 3, 2)
    @test_throws ArgumentError TenSolver.peps_topology(:square, (2,))
    @test_throws ArgumentError TenSolver.peps_topology(:square, 4)
    @test_throws ArgumentError TenSolver.peps_topology(:pegasus, (2, 3))
    @test_throws ArgumentError TenSolver.peps_local_dimension(1.5)
  end

  @testset "Unavailable PEPS extension errors clearly" begin
    has_spinglasspeps_components = all(
      package -> !isnothing(Base.find_package(package)),
      ("SpinGlassNetworks", "SpinGlassEngine", "SpinGlassTensors"),
    )

    if has_spinglasspeps_components
      @test_skip "SpinGlassPEPS components are available; this error path does not apply."
    else
      model = JuMP.Model(TenSolver.Optimizer)
      JuMP.set_silent(model)
      JuMP.set_attribute(model, "backend", :peps)
      JuMP.set_attribute(model, "peps_topology", (1, 1))
      @JuMP.variable(model, x, Bin)
      @JuMP.objective(model, Min, -x)

      error = try
        JuMP.optimize!(model)
        nothing
      catch error
        error
      end

      @test error isa ArgumentError
      @test occursin("PEPSBackend is not available", sprint(showerror, error))
      @test occursin("SpinGlassNetworks", sprint(showerror, error))
    end
  end

  @testset "Optional PEPS optimizer solve" begin
    has_spinglasspeps_components = all(
      package -> !isnothing(Base.find_package(package)),
      ("SpinGlassNetworks", "SpinGlassEngine", "SpinGlassTensors"),
    )

    if !has_spinglasspeps_components
      @test_skip "SpinGlassPEPS component packages are not available in this environment."
    else
      import SpinGlassEngine
      import SpinGlassNetworks
      import SpinGlassTensors

      Q = [
        -1.0 0.5 0.0 0.0
         0.0 -0.5 0.0 0.0
         0.0 0.0 -0.25 0.25
         0.0 0.0 0.0 -0.75
      ]
      l = [0.0, 0.25, -0.25, 0.0]
      c = 0.125
      objective(state) = dot(state, Q, state) + dot(l, state) + c
      exact_energy, _ = brute_force(objective, Float64, 4)

      model = JuMP.Model(TenSolver.Optimizer)
      JuMP.set_silent(model)
      JuMP.set_attribute(model, "backend", :peps)
      JuMP.set_attribute(model, "peps_layout", :square)
      JuMP.set_attribute(model, "peps_topology", (2, 2))
      JuMP.set_attribute(model, "peps_bond_dim", 4)
      JuMP.set_attribute(model, "peps_max_states", 4)
      JuMP.set_attribute(model, "peps_cutoff_prob", 0.0)
      JuMP.set_attribute(model, "peps_strategy", :svd)
      JuMP.set_attribute(model, "peps_transformations", :identity)
      @JuMP.variable(model, x[1:4], Bin)
      @JuMP.objective(
        model,
        Min,
        sum(Q[i, j] * x[i] * x[j] for i in 1:4, j in 1:4) +
        sum(l[i] * x[i] for i in 1:4) + c,
      )

      JuMP.optimize!(model)

      state = round.(Int, JuMP.value.(x))
      @test JuMP.objective_value(model) ≈ exact_energy atol = 1e-6
      @test objective(state) ≈ JuMP.objective_value(model) atol = 1e-6
    end
  end

  @testset "Fake PEPS optimizer solve" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "backend", :peps)
    JuMP.set_attribute(model, "peps_topology", FakePEPSTopology(2))
    JuMP.set_attribute(model, "peps_beta", 1.75)
    JuMP.set_attribute(model, "peps_bond_dim", 3)
    JuMP.set_attribute(model, "peps_max_states", 4)
    JuMP.set_attribute(model, "peps_cutoff_prob", 0.0)
    JuMP.set_attribute(model, "peps_device", fake_peps_device)
    JuMP.set_attribute(model, "peps_strategy", :svd)
    JuMP.set_attribute(model, "peps_num_sweeps", 2)
    JuMP.set_attribute(model, "peps_graduate_truncation", false)
    JuMP.set_attribute(model, "peps_transformations", :identity)
    JuMP.set_attribute(model, "peps_local_dimension", 2)
    JuMP.set_attribute(model, "peps_no_cache", true)
    JuMP.set_attribute(model, QUBODrivers.FinalNumberOfReads(), 5)
    @JuMP.variable(model, x[1:2], Bin)
    @JuMP.objective(model, Min, -x[1] - 2x[2])

    JuMP.optimize!(model)

    solution = QUBOTools.solution(JuMP.unsafe_backend(model))
    metadata = QUBOTools.metadata(solution)
    peps_metadata = metadata["tensolver"]["peps"]

    @test JuMP.objective_value(model) ≈ -3.0
    @test round.(Int, JuMP.value.(x)) == [1, 1]
    @test QUBOTools.reads(solution) == 5
    @test QUBOTools.state(solution, 1) == [1, 1]
    @test QUBOTools.reads(solution, 1) == 4
    @test QUBOTools.state(solution, 2) == [1, 0]
    @test QUBOTools.reads(solution, 2) == 1
    @test isempty(QUBODrivers.validate_metadata(solution))
    @test metadata["algorithm"]["name"] == "FakePEPS"
    @test metadata["backend"]["name"] == "TenSolver"
    @test metadata["backend"]["version"] == TenSolver.__VERSION__
    @test metadata["optimizer"]["iterations"] == 2
    @test metadata["optimizer"]["evaluations"] == 2
    @test metadata["reads"]["final_number_of_reads"] == 5
    @test peps_metadata["topology"] == "fake"
    @test peps_metadata["candidate_states"] == 2
    @test peps_metadata["parameters"]["bond_dim"] == 3
    @test peps_metadata["parameters"]["strategy"] == "svd"
    @test peps_metadata["parameters"]["local_dimension"] == 2
  end

  @testset "PEPS SampleSet adaptation" begin
    spin_solution = TenSolver.PEPSSolution{Float64}(
      [[1, -1], [-1, 1]],
      [-2.0, -1.0],
      [0.75, 0.25],
      Dict{String,Any}("backend" => "SpinGlassPEPS"),
    )
    solution = TenSolver.boolean_peps_solution(spin_solution)
    Q = [0.0 -1.0; 0.0 0.0]
    l = [0.0, -0.5]
    samples = TenSolver.qubo_samples(Float64, solution, l, Q, 1.0, 0.0, 3)

    @test solution.states == [[1, 0], [0, 1]]
    @test getfield.(samples, :state) == [[1, 0], [0, 1]]
    @test getfield.(samples, :value) == [0.0, -0.5]
    @test getfield.(samples, :reads) == [2, 1]
    @test TenSolver.peps_read_counts(solution, 0) == [0, 0]
    @test TenSolver.peps_read_counts(solution, 2) == [2, 0]
    @test_throws ArgumentError TenSolver.peps_read_counts(solution, -1)

    no_probabilities = TenSolver.PEPSSolution{Float64}(
      [[1, 0], [0, 1]],
      [-2.0, -1.0],
      Float64[],
      Dict{String,Any}(),
    )
    @test TenSolver.peps_read_counts(no_probabilities, 3) == [3, 0]
  end
end

@testset "JuMP interface and preprocess attribute" begin
  Q = [0.0 0.0 -2.0;
       0.0 0.0  0.0;
      -2.0 0.0  0.0]
  l = [0.5, 1.0, 0.5]
  c = 1.25

  m = JuMP.Model(TenSolver.Optimizer)
  JuMP.set_silent(m)
  JuMP.set_attribute(m, "preprocess", true)
  JuMP.set_attribute(m, "iterations", 3)
  @JuMP.variable(m, x[1:3], Bin)
  @JuMP.objective(m, Min, dot(x, Q, x) + dot(l, x) + c)

  JuMP.optimize!(m)

  @test JuMP.objective_value(m) ≈ -1.75
  @test JuMP.value.(x) == [1, 0, 1]
end
