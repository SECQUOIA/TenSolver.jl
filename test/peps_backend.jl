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

  backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1))

  @test backend.topology == TenSolver.SquareGrid(1, 1)

  # Solve keywords are validated by peps_options, mirroring how the DMRG
  # backend takes its parameters as minimize keywords.
  opts = TenSolver.peps_options(;
    beta = 1.5,
    maxdim = 4,
    max_states = 2,
    cutoff_prob = 0.0,
    contraction = :svd,
    transformations = :identity,
  )
  @test opts.beta == 1.5
  @test opts.maxdim == 4
  @test opts.iterations == 1
  @test opts.max_states == 2
  @test opts.cutoff_prob == 0.0
  @test opts.onGPU == false
  @test opts.contraction == :svd
  @test_throws ArgumentError TenSolver.peps_options(; beta = 0)
  @test_throws ArgumentError TenSolver.peps_options(; maxdim = 0)
  @test_throws ArgumentError TenSolver.peps_options(; max_states = 0)
  @test_throws ArgumentError TenSolver.peps_options(; contraction = :unknown)
  @test_throws ArgumentError TenSolver.peps_options(; iterations = 0)

  Q = reshape([-1.0], 1, 1)
  peps_error = try
    minimize(Q; backend, verbosity = 0)
  catch err
    err
  end
  @test peps_error isa ArgumentError
  @test occursin("PEPSBackend is not available", sprint(showerror, peps_error))
  @test occursin("SpinGlassNetworks", sprint(showerror, peps_error))

  model = TenSolver.IsingModel(zeros(1, 1), [-1.0])
  energy, solution = TenSolver.solve_ising(model; backend = :dmrg, verbosity = 0)
  @test energy ≈ -1.0
  @test sample(solution) == [1]

  @testset "solve_ising preserves pair couplings and offsets" begin
    J = [0.0 0.5; 1.0 0.0]
    h = [-0.25, 0.75]
    offset = 2.0
    ising = TenSolver.IsingModel(J, h, offset)
    spin_states = [[s1, s2] for s1 in (-1, 1) for s2 in (-1, 1)]
    energies = [TenSolver.ising_energy(ising, spin) for spin in spin_states]
    expected_energy, expected_index = findmin(energies)
    expected_spin = spin_states[expected_index]

    for solve_call in (
      () -> TenSolver.solve_ising(ising; backend=:dmrg, verbosity=0),
      () -> TenSolver.solve_ising(J, h, offset; backend=:dmrg, verbosity=0),
    )
      ising_energy_value, ising_solution = solve_call()
      sample_bits = TenSolver.sample(ising_solution)
      sample_spin = TenSolver.bool_to_spin(sample_bits)

      @test ising_energy_value ≈ expected_energy
      @test sample_spin == expected_spin
      @test TenSolver.ising_energy(ising, sample_spin) ≈ expected_energy
    end
  end

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

  # Duplicate decoded states must be merged (probabilities summed) before
  # construction; the constructor enforces the invariant.
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1, 0], [0, 1], [1, 0]],
    [-2.0, -1.0, -2.0],
    [0.2, 0.3, 0.4],
    Dict{String, Any}("backend" => "SpinGlassPEPS"),
    nothing,
  )

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

    backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(2, 2))
    peps_kwargs = (
      beta = 2.0,
      maxdim = 4,
      max_states = 4,
      cutoff_prob = 0.0,
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

    energy, solution = minimize(Q, l, c; backend, verbosity = 0, peps_kwargs...)
    state = sample(solution)

    @test energy ≈ exact_energy atol = 1e-6
    @test objective(state) ≈ energy atol = 1e-6
    @test solution.metadata["backend"] == "SpinGlassPEPS"
    @test solution.metadata["topology"] == "square"
    @test solution.metadata["selected_transformation"] == string(SpinGlassEngine.rotation(0))
    @test first(solution.energies) ≈ energy atol = 1e-6
  end
end
