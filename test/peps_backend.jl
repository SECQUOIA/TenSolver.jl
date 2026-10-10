import DynamicPolynomials

@testset "PEPS backend core" begin
  @test !(:Solution in names(TenSolver))
  @test !(:DMRGSolution in names(TenSolver))
  @test !(:PEPSBackend in names(TenSolver))
  @test !(:SquareGrid in names(TenSolver))
  @test !(:KingGrid in names(TenSolver))
  @test !(:PEPSSolution in names(TenSolver))

  @test isabstracttype(TenSolver.Solution)
  @test TenSolver.DMRGSolution <: TenSolver.Solution
  @test TenSolver.PEPSSolution <: TenSolver.Solution

  @test TenSolver.SquareGrid(2, 3).m == 2
  @test TenSolver.SquareGrid(2, 3).n == 3
  @test TenSolver.SquareGrid(2, 3).spins_per_site == 1
  @test TenSolver.KingGrid(2, 3, 2).spins_per_site == 2
  @test_throws ArgumentError TenSolver.SquareGrid(0, 1)
  @test_throws ArgumentError TenSolver.KingGrid(1, 0)

  backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1))
  @test backend.topology == TenSolver.SquareGrid(1, 1)

  J = [0.0 0.5; 1.0 0.0]
  h = [-0.25, 0.75]
  offset = 2.0
  spins = [-1, 1]
  @test TenSolver.ising_energy(J, h, offset, spins) ==
        offset + dot(h, spins) + dot(spins, J, spins)
  @test_throws DimensionMismatch TenSolver.ising_energy(J, h, offset, [1])

  peps_error = TenSolver.backend_error(backend)
  @test peps_error isa ArgumentError
  @test occursin("PEPSBackend is not available", sprint(showerror, peps_error))
  @test occursin("SpinGlassPEPS", sprint(showerror, peps_error))
  @test !occursin("DMRG", sprint(showerror, peps_error))

  metadata = Dict{String,Any}("backend" => "SpinGlassPEPS")
  peps_solution =
    TenSolver.PEPSSolution{Float64}([[1, -1], [-1, 1]], [-2.0, -1.0], [0.0, 1.0], metadata)
  @test is_feasible(peps_solution)
  @test sample(peps_solution) == [-1, 1]
  @test sample(peps_solution, 2) == [[-1, 1], [-1, 1]]
  @test [-1, 1] in peps_solution
  @test !([1, -1] in peps_solution)
  @test !([1, 1] in peps_solution)
  @test TenSolver.prob(peps_solution, [-1, 1]) ≈ 1.0

  deterministic_solution =
    TenSolver.PEPSSolution{Float64}([[1, -1], [-1, 1]], [-2.0, -1.0], Float64[], metadata)
  @test sample(deterministic_solution) == [1, -1]
  @test TenSolver.prob(deterministic_solution, [1, -1]) == 1.0
  @test TenSolver.prob(deterministic_solution, [-1, 1]) == 0.0

  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    Vector{Int}[],
    Float64[],
    Float64[],
    metadata,
  )
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1, -1], [-1, 1]],
    [-2.0],
    [0.5, 0.5],
    metadata,
  )
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1, -1], [-1, 1]],
    [-2.0, -1.0],
    [1.0],
    metadata,
  )
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1, -1], [-1, 1]],
    [-2.0, -1.0],
    [1.0, -0.5],
    metadata,
  )
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1, -1], [-1, 1]],
    [-2.0, -1.0],
    [0.0, 0.0],
    metadata,
  )
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1, -1], [-1, 1], [1, -1]],
    [-2.0, -1.0, -2.0],
    [0.2, 0.3, 0.4],
    metadata,
  )
  @test_throws ArgumentError TenSolver.PEPSSolution{Float64}(
    [[1], [-1, 1]],
    [-2.0, -1.0],
    [0.5, 0.5],
    metadata,
  )
end

@testset "Optional SpinGlassPEPS extension" begin
  if isnothing(Base.find_package("SpinGlassPEPS"))
    @test isnothing(Base.get_extension(TenSolver, :TenSolverSpinGlassPEPSExt))
    @test_throws r"Install/load SpinGlassPEPS 2.x" minimize(
      zeros(1, 1),
      zeros(1),
      0.0;
      domain = [-1, 1],
      backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(1, 1)),
      verbosity = 0,
    )
    @test_skip "SpinGlassPEPS is not available in this environment."
  else
    import SpinGlassPEPS
    extension = Base.get_extension(TenSolver, :TenSolverSpinGlassPEPSExt)
    @test !isnothing(extension)

    @testset "Retained log probability normalization" begin
      # A repeated state has twice the weight of the other state, despite weights
      # being too small to exponentiate directly in Float64.
      records = [
        (; state = [1, -1], energy = -2.0, log_probability = -1000.0),
        (; state = [-1, 1], energy = -1.0, log_probability = -1000.0),
        (; state = [1, -1], energy = -2.0, log_probability = -1000.0),
        (; state = [1, 1], energy = 0.0, log_probability = -Inf),
      ]
      normalized = extension.normalized_records(records)
      @test getproperty.(normalized, :state) == [[1, -1], [-1, 1], [1, 1]]
      @test getproperty.(normalized, :probability) ≈ [2 / 3, 1 / 3, 0.0]
      @test_throws ArgumentError extension.normalized_records([(;
        state = [1],
        energy = 0.0,
        log_probability = -Inf,
      ),])
    end

    backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(2, 2))
    peps_kwargs = (
      device = TenSolver.cpu,
      beta = 2.0,
      maxdim = 4,
      max_states = 4,
      cutoff_prob = 0.0,
      contraction = :svd,
      transformations = :identity,
    )

    J = [
      0.0 0.5 0.0 0.0
      0.0 0.0 0.0 0.0
      0.0 0.0 0.0 0.25
      0.0 0.0 0.0 0.0
    ]
    h = [-1.0, -0.25, 0.25, -0.75]
    offset = 0.125
    objective(spins) = dot(spins, J, spins) + dot(h, spins) + offset
    exact_energy = -2.375
    optimum = [1, -1, -1, 1]
    @test objective(optimum) == exact_energy
    @test first(brute_force(objective, 4; domain = [-1, 1])) == exact_energy

    energy, solution =
      minimize(J, h, offset; domain = [-1, 1], backend, verbosity = 0, peps_kwargs...)
    state = sample(solution)

    @test energy ≈ exact_energy atol = 1e-6
    @test optimum in solution.states
    @test TenSolver.prob(solution, optimum) > 0
    @test all(isfinite, solution.probabilities)
    @test all(>=(0), solution.probabilities)
    @test sum(solution.probabilities) ≈ 1.0
    @test all(isapprox(objective(xs), value; atol = 1e-6) for
              (xs, value) in zip(solution.states, solution.energies))
    @test state in solution.states
    @test objective(state) ≈ solution.energies[findfirst(==(state), solution.states)] atol =
      1e-6
    @test all(in((-1, 1)), state)
    @test solution.metadata["backend"] == "SpinGlassPEPS"
    @test solution.metadata["topology"] == "square"
    @test solution.metadata["selected_transformation"] == string(SpinGlassPEPS.rotation(0))
    @test haskey(solution.metadata, "raw")
    @test all(<=(0), solution.metadata["spin_glass_probabilities"])
    @test first(solution.metadata["spin_glass_probabilities"]) < 0
    @test first(solution.energies) ≈ energy atol = 1e-6

    DynamicPolynomials.@polyvar s[1:4]
    polynomial = 0.5s[1] * s[2] + 0.25s[3] * s[4] + dot(h, s) + offset
    quadratic, linear, constant = extension.quadratic_form(polynomial)
    @test quadratic == J
    @test linear == h
    @test constant == offset
    polynomial_energy, polynomial_solution = minimize(
      polynomial;
      domain = [-1, 1],
      backend,
      verbosity = 0,
      peps_kwargs...,
      transformations = :all,
    )
    polynomial_state = sample(polynomial_solution)
    @test polynomial_energy ≈ exact_energy atol = 1e-6
    @test optimum in polynomial_solution.states
    @test all(isapprox(objective(xs), value; atol = 1e-6) for
              (xs, value) in zip(polynomial_solution.states, polynomial_solution.energies))
    @test polynomial_state in polynomial_solution.states
    @test sum(polynomial_solution.probabilities) ≈ 1.0
    @test allunique(polynomial_solution.states)
    @test length(polynomial_solution.metadata["raw"]) == 8

    # Symmetrization keeps the same quadratic form; spin squares contribute
    # a constant 0.375, including negative diagonal coefficients.
    diagonal_J = (J + J') / 2 + Diagonal([0.25, -0.5, 0.75, -0.125])
    diagonal_energy, diagonal_solution = minimize(
      diagonal_J,
      h,
      offset;
      domain = [-1, 1],
      backend,
      verbosity = 0,
      peps_kwargs...,
    )
    @test diagonal_energy ≈ -2.0 atol = 1e-6
    @test optimum in diagonal_solution.states
    @test all(isapprox(dot(xs, diagonal_J, xs) + dot(h, xs) + offset, value; atol = 1e-6)
              for (xs, value) in zip(diagonal_solution.states, diagonal_solution.energies))

    cubic = polynomial + s[1] * s[2] * s[3]
    @test_throws ArgumentError minimize(
      cubic;
      domain = [-1, 1],
      backend,
      verbosity = 0,
      peps_kwargs...,
    )
    @test_throws ArgumentError minimize(
      J,
      h,
      offset;
      domain = [0, 1],
      backend,
      verbosity = 0,
      peps_kwargs...,
    )
  end
end
