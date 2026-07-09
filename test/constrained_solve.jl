import DynamicPolynomials
const ConstrainedSolveDP = DynamicPolynomials

function constrained_brute_force(obj, n, constraints)
  best = Inf
  best_x = nothing

  for bits in Iterators.product(fill(0:1, n)...)
    x = collect(bits)
    is_feasible(x, constraints) || continue

    value = obj(x)
    if value < best
      best = value
      best_x = x
    end
  end

  isnothing(best_x) && throw(ArgumentError("no feasible bitstring"))
  return best, best_x
end

function assert_constrained_solution(E, psi, obj, constraints, expected_energy, expected_sample)
  x = TenSolver.sample(psi)

  @test is_feasible(x, constraints)
  @test x == expected_sample
  @test obj(x) ≈ E
  @test E ≈ expected_energy

  for sampled in TenSolver.sample(psi, 5)
    @test is_feasible(sampled, constraints)
  end
end

@testset "Constrained solving" begin
  all_constraint_types = AbstractConstraint[
    SumConstraint([1, 2, 3], [1, 1, 1], 2; relation=:(<=)),
    NotEqualsConstraint([1, 2], [1, 1]),
    ExactlyOneConstraint([2, 3], 1),
    RelationConstraint(1, :(>=), 3),
  ]

  qubo_kwargs = (
    constraints=all_constraint_types,
    iterations=3,
    verbosity=0,
    cutoff=1e-12,
    noise=[0.0],
  )

  @testset "QUBO minimize supports all v1 constraints" begin
    Q = zeros(3, 3)
    l = [-3.0, -2.0, -1.0]
    obj(x) = dot(x, Q, x) + dot(l, x)
    expected_energy, expected_sample = constrained_brute_force(obj, 3, all_constraint_types)

    E, psi = minimize(Q, l; qubo_kwargs...)

    @test expected_sample == [1, 0, 1]
    assert_constrained_solution(E, psi, obj, all_constraint_types, expected_energy, expected_sample)
  end

  @testset "QUBO preprocessing keeps constraints in original variable order" begin
    Q = [
      0.0 0.0 0.1
      0.0 0.0 0.0
      0.1 0.0 0.0
    ]
    l = [-3.0, -2.0, -1.0]
    obj(x) = dot(x, Q, x) + dot(l, x)
    expected_energy, expected_sample = constrained_brute_force(obj, 3, all_constraint_types)

    E, psi = minimize(Q, l; preprocess=true, qubo_kwargs...)

    @test !TenSolver.is_identity_permutation(TenSolver.qmatrix_permutation(Q; cutoff=1e-12))
    assert_constrained_solution(E, psi, obj, all_constraint_types, expected_energy, expected_sample)
  end

  @testset "Maximize forwards constraints" begin
    Q = zeros(2, 2)
    l = [1.0, 2.0]
    constraints = AbstractConstraint[ExactlyOneConstraint([1, 2], 1)]
    obj(x) = dot(x, Q, x) + dot(l, x)

    E, psi = maximize(
      Q,
      l;
      constraints,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )
    x = TenSolver.sample(psi)

    @test is_feasible(x, constraints)
    @test x == [0, 1]
    @test obj(x) ≈ E
    @test E ≈ 2.0
  end

  @testset "Polynomial minimize supports constraints" begin
    ConstrainedSolveDP.@polyvar y[1:3]
    p = -3.0y[1] - 2.0y[2] - 1.0y[3]
    obj(x) = p(y => x)
    expected_energy, expected_sample = constrained_brute_force(obj, 3, all_constraint_types)

    E, psi = minimize(
      p;
      constraints=all_constraint_types,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E, psi, obj, all_constraint_types, expected_energy, expected_sample)
  end

  @testset "Single-variable constrained problems use the scalar fast path" begin
    Q = zeros(1, 1)
    l = [-1.0]
    force_zero = AbstractConstraint[SumConstraint([1], [1], 0; relation=:(==))]
    obj(x) = dot(x, Q, x) + dot(l, x)

    E, psi = minimize(
      Q,
      l;
      constraints=force_zero,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E, psi, obj, force_zero, 0.0, [0])

    ConstrainedSolveDP.@polyvar z
    p = -2.0z + 0.0
    force_one = AbstractConstraint[ExactlyOneConstraint([1], 1)]
    poly_obj(x) = p([z] => x)

    E_poly, psi_poly = minimize(
      p;
      constraints=force_one,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E_poly, psi_poly, poly_obj, force_one, -2.0, [1])
  end

  @testset "Callbacks and stats remain available" begin
    Q = zeros(2, 2)
    l = [-1.0, -2.0]
    constraints = AbstractConstraint[NotEqualsConstraint([1, 2], [1, 1])]
    calls = Int[]
    objectives = Float64[]

    E, psi = minimize(
      Q,
      l;
      constraints,
      iterations=2,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
      on_iteration=(mps; iteration, objective, kw...) -> begin
        push!(calls, iteration)
        push!(objectives, objective)
        @test is_feasible(TenSolver.sample(TenSolver.Solution(deepcopy(mps), Float64[], Int[], Float64[])), constraints)
      end,
    )

    @test calls == [1, 2]
    @test length(objectives) == 2
    @test length(psi.energies) == 2
    @test length(psi.bond_dims) == 2
    @test length(psi.elapsed_times) == 2
    @test is_feasible(TenSolver.sample(psi), constraints)
    @test E ≈ -2.0
  end

  @testset "Zero objective keeps feasible constrained samples" begin
    constraints = AbstractConstraint[ExactlyOneConstraint([1, 2], 1)]
    E, psi = minimize(
      zeros(2, 2);
      constraints,
      iterations=2,
      verbosity=0,
      cutoff=1e-12,
    )

    @test E ≈ 0.0
    for sampled in TenSolver.sample(psi, 5)
      @test is_feasible(sampled, constraints)
    end
  end

  @testset "Infeasible constraints fail clearly" begin
    impossible = AbstractConstraint[SumConstraint([1, 2], [1, 1], 3; relation=:(==))]
    err = try
      minimize(zeros(2, 2); constraints=impossible, verbosity=0)
    catch err
      err
    end

    @test err isa ArgumentError
    @test occursin("empty feasible subspace", sprint(showerror, err))
  end
end
