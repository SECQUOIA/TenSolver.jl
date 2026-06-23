@testset "PEPS backend core" begin
  @test !(:Solution in names(TenSolver))
  @test !(:PEPSBackend in names(TenSolver))
  @test !(:SquareGrid in names(TenSolver))
  @test !(:KingGrid in names(TenSolver))
  @test !(:PEPSSolution in names(TenSolver))

  @test TenSolver.SquareGrid(2, 3).m == 2
  @test TenSolver.SquareGrid(2, 3).n == 3
  @test TenSolver.SquareGrid(2, 3).spins_per_site == 1
  @test TenSolver.KingGrid(2, 3, 2).spins_per_site == 2
  @test_throws ArgumentError TenSolver.SquareGrid(0, 1)
  @test_throws ArgumentError TenSolver.KingGrid(1, 0)

  backend = TenSolver.PEPSBackend(
    TenSolver.SquareGrid(1, 1);
    beta = 1.5,
    bond_dim = 4,
    max_states = 2,
    cutoff_prob = 0.0,
    onGPU = false,
    contraction = :svd,
    transformations = :identity,
  )

  @test backend.topology == TenSolver.SquareGrid(1, 1)
  @test backend.beta == 1.5
  @test backend.bond_dim == 4
  @test backend.max_states == 2
  @test backend.cutoff_prob == 0.0
  @test backend.contraction == :svd
  @test_throws ArgumentError TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1); beta = 0)
  @test_throws ArgumentError TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1); bond_dim = 0)
  @test_throws ArgumentError TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1); max_states = 0)
  @test_throws ArgumentError TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1); contraction = :unknown)

  Q = reshape([-1.0], 1, 1)
  peps_error = try
    minimize(Q; backend, verbosity = 0)
  catch err
    err
  end
  @test peps_error isa ArgumentError
  @test occursin("PEPSBackend is not available", sprint(showerror, peps_error))
  @test occursin("SpinGlassNetworks", sprint(showerror, peps_error))
  @test_throws ArgumentError minimize(Q; backend, preprocess = true, verbosity = 0)

  model = TenSolver.IsingModel(zeros(1, 1), [-1.0])
  energy, solution = solve_ising(model; backend = :dmrg, verbosity = 0)
  @test energy ≈ -1.0
  @test sample(solution) == [1]

  peps_solution = TenSolver.PEPSSolution{Float64}(
    [[1, 0], [0, 1]],
    [-2.0, -1.0],
    [0.0, 1.0],
    Dict{String, Any}("backend" => "SpinGlassPEPS"),
    nothing,
  )
  @test sample(peps_solution) == [0, 1]
  @test sample(peps_solution, 2) == [[0, 1], [0, 1]]
  @test [0, 1] in peps_solution
  @test !([1, 0] in peps_solution)
  @test !([0, 0] in peps_solution)
  @test TenSolver.prob(peps_solution, [0, 1]) ≈ 1.0

  duplicate_state_solution = TenSolver.PEPSSolution{Float64}(
    [[1, 0], [0, 1], [1, 0]],
    [-2.0, -1.0, -2.0],
    [0.2, 0.3, 0.4],
    Dict{String, Any}("backend" => "SpinGlassPEPS"),
    nothing,
  )
  @test TenSolver.prob(duplicate_state_solution, [1, 0]) ≈ 0.6
  @test TenSolver.prob(duplicate_state_solution, [0, 1]) ≈ 0.3

  @test_throws ArgumentError sample(TenSolver.PEPSSolution{Float64}(
    [[1, 0], [0, 1]],
    [-2.0, -1.0],
    [1.0],
    Dict{String, Any}(),
    nothing,
  ))
  @test_throws ArgumentError sample(TenSolver.PEPSSolution{Float64}(
    [[1, 0], [0, 1]],
    [-2.0, -1.0],
    [1.0, -0.5],
    Dict{String, Any}(),
    nothing,
  ))
  @test_throws ArgumentError sample(TenSolver.PEPSSolution{Float64}(
    [[1, 0], [0, 1]],
    [-2.0, -1.0],
    [0.0, 0.0],
    Dict{String, Any}(),
    nothing,
  ))
end

@testset "Optional SpinGlassPEPS extension" begin
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

    backend = TenSolver.PEPSBackend(
      TenSolver.SquareGrid(2, 2);
      beta = 2.0,
      bond_dim = 4,
      max_states = 4,
      cutoff_prob = 0.0,
      onGPU = false,
      contraction = :svd,
      transformations = :identity,
    )

    Q = [
      -1.0 0.5 0.0 0.0
       0.0 -0.5 0.0 0.0
       0.0 0.0 -0.25 0.25
       0.0 0.0 0.0 -0.75
    ]
    l = [0.0, 0.25, -0.25, 0.0]
    c = 0.125
    objective(x) = dot(x, Q, x) + dot(l, x) + c
    exact_energy, _ = brute_force(objective, Int, 4)

    energy, solution = minimize(Q, l, c; backend, verbosity = 0)
    state = sample(solution)

    @test energy ≈ exact_energy atol = 1e-6
    @test objective(state) ≈ energy atol = 1e-6
    @test solution.metadata["backend"] == "SpinGlassPEPS"
    @test solution.metadata["topology"] == "square"
    @test solution.metadata["selected_transformation"] == string(SpinGlassEngine.rotation(0))
    @test first(solution.energies) ≈ energy atol = 1e-6
  end
end
