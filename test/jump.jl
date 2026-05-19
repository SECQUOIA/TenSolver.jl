import JuMP

@testset "JuMP interface" begin
  dim = 5
  Q = 2*randn(dim, dim)
  l = 2*randn(dim)
  c = randn()
  obj(x) = dot(x, Q, x) + dot(l, x) + c

  m = JuMP.Model(TenSolver.Optimizer)
  @JuMP.variable(m, x[1:dim], Bin)
  @JuMP.objective(m, Min, dot(x, Q, x) + dot(l, x) + c)

  JuMP.optimize!(m)

  e = JuMP.objective_value(m)

  # ~:~ Exact solution ~:~ #
  e0, x0 = brute_force(obj, Float64, dim)
  # Same minimum value
  @test e ≈ e0
  # Same solution
  @test JuMP.value.(x) == x0
end

@testset "JuMP backend attributes" begin
  @testset "Default backend" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    @JuMP.variable(model, x, Bin)
    @JuMP.objective(model, Min, -x)

    JuMP.optimize!(model)

    @test JuMP.objective_value(model) ≈ -1.0
    @test JuMP.value(x) ≈ 1.0
  end

  @testset "Explicit DMRG backend" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "backend", :dmrg)
    JuMP.set_attribute(model, "num_reads", 4)
    @JuMP.variable(model, x, Bin)
    @JuMP.objective(model, Min, -x)

    JuMP.optimize!(model)

    @test JuMP.objective_value(model) ≈ -1.0
    @test JuMP.value(x) ≈ 1.0
  end

  @testset "String backend normalization" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "backend", "dmrg")
    @JuMP.variable(model, x, Bin)
    @JuMP.objective(model, Min, -x)

    JuMP.optimize!(model)

    @test JuMP.objective_value(model) ≈ -1.0
  end

  @testset "PEPS requires topology metadata" begin
    model = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "backend", :peps)
    @JuMP.variable(model, x, Bin)
    @JuMP.objective(model, Min, -x)

    err = try
      JuMP.optimize!(model)
    catch err
      err
    end

    @test err isa ArgumentError
    @test occursin("peps_topology", sprint(showerror, err))
    @test occursin("(m, n)", sprint(showerror, err))
  end

  @testset "Unavailable PEPS extension errors clearly" begin
    has_spinglasspeps_components = all(pkg -> !isnothing(Base.find_package(pkg)), (
      "SpinGlassNetworks",
      "SpinGlassEngine",
      "SpinGlassTensors",
    ))

    if has_spinglasspeps_components
      @test_skip "SpinGlassPEPS component packages are available; unavailable-extension error path does not apply."
    else
      model = JuMP.Model(TenSolver.Optimizer)
      JuMP.set_silent(model)
      JuMP.set_attribute(model, "backend", :peps)
      JuMP.set_attribute(model, "peps_layout", :square)
      JuMP.set_attribute(model, "peps_topology", (1, 1))
      JuMP.set_attribute(model, "peps_beta", 1.5)
      JuMP.set_attribute(model, "peps_bond_dim", 4)
      JuMP.set_attribute(model, "peps_max_states", 2)
      JuMP.set_attribute(model, "peps_strategy", :svd)
      @JuMP.variable(model, x, Bin)
      @JuMP.objective(model, Min, -x)

      err = try
        JuMP.optimize!(model)
      catch err
        err
      end

      @test err isa ArgumentError
      @test occursin("PEPSBackend is not available", sprint(showerror, err))
      @test occursin("SpinGlassNetworks", sprint(showerror, err))
    end
  end

  @testset "Optional PEPS optimizer solve" begin
    has_spinglasspeps_components = all(pkg -> !isnothing(Base.find_package(pkg)), (
      "SpinGlassNetworks",
      "SpinGlassEngine",
      "SpinGlassTensors",
    ))

    if !has_spinglasspeps_components
      @test_skip "SpinGlassPEPS component packages are not available in this environment."
    else
      import SpinGlassNetworks
      import SpinGlassEngine
      import SpinGlassTensors

      Q = [
        -1.0 0.5 0.0 0.0
         0.0 -0.5 0.0 0.0
         0.0 0.0 -0.25 0.25
         0.0 0.0 0.0 -0.75
      ]
      l = [0.0, 0.25, -0.25, 0.0]
      c = 0.125
      objective(x) = dot(x, Q, x) + dot(l, x) + c
      exact_energy, _ = brute_force(objective, Float64, 4)

      model = JuMP.Model(TenSolver.Optimizer)
      JuMP.set_silent(model)
      JuMP.set_attribute(model, "backend", :peps)
      JuMP.set_attribute(model, "peps_layout", :square)
      JuMP.set_attribute(model, "peps_topology", (2, 2))
      JuMP.set_attribute(model, "peps_beta", 2.0)
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
        sum(l[i] * x[i] for i in 1:4) + c
      )

      JuMP.optimize!(model)

      state = round.(Int, JuMP.value.(x))
      @test JuMP.objective_value(model) ≈ exact_energy atol = 1e-6
      @test objective(state) ≈ JuMP.objective_value(model) atol = 1e-6
    end
  end

  @testset "PEPS SampleSet adaptation" begin
    peps = TenSolver.PEPSSolution{Float64}(
      [[1, 0], [0, 1]],
      [-2.0, -1.0],
      [0.75, 0.25],
      Dict{String, Any}(
        "backend" => "SpinGlassPEPS",
        "topology" => "square",
        "selected_transformation" => "identity",
        "largest_discarded_probability" => 0.01,
      ),
      nothing,
    )
    Q = [0.0 -1.0; 0.0 0.0]
    l = [0.0, -0.5]
    samples = TenSolver._qubo_samples(Float64, peps, l, Q, 1.0, 0.0, 3)

    @test getfield.(samples, :state) == [[1, 0], [0, 1], [1, 0]]
    @test getfield.(samples, :value) == [0.0, -0.5, 0.0]

    metadata = Dict{String, Any}(
      "origin" => "TenSolver",
      "time" => Dict{String, Any}("effective" => 1.25),
    )
    TenSolver._add_backend_metadata!(metadata, peps)

    @test metadata["backend"] == "SpinGlassPEPS"
    @test metadata["peps"]["topology"] == "square"
    @test metadata["peps"]["candidate_states"] == 2
    @test metadata["peps"]["effective_time"] == 1.25
    @test metadata["peps"]["largest_discarded_probability"] == 0.01
  end
end
