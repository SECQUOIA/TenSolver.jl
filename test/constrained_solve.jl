import DynamicPolynomials

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
    expected_energy, expected_sample = brute_force(obj, 3, all_constraint_types)

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
    expected_energy, expected_sample = brute_force(obj, 3, all_constraint_types)

    E, psi = minimize(Q, l; preprocess=true, qubo_kwargs...)

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
    DynamicPolynomials.@polyvar y[1:3]
    p = -3.0y[1] - 2.0y[2] - 1.0y[3]
    obj(x) = p(y => x)
    expected_energy, expected_sample = brute_force(obj, 3, all_constraint_types)

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

    DynamicPolynomials.@polyvar z
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

    # Regression: with a positive feasible objective, the projected Hamiltonian
    # P'HP assigns zero energy to the (infeasible) rejected basis state, so the
    # scalar path must still select the feasible one rather than the global
    # minimum. Previously this threw a misleading "constraints may be infeasible"
    # error even though the pinned assignment is feasible.
    Qpos = fill(3.0, 1, 1)
    force_one_qubo = AbstractConstraint[SumConstraint([1], [1], 1; relation=:(==))]
    qubo_pos_obj(x) = dot(x, Qpos, x)

    E_qpos, psi_qpos = minimize(
      Qpos;
      constraints=force_one_qubo,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E_qpos, psi_qpos, qubo_pos_obj, force_one_qubo, 3.0, [1])

    DynamicPolynomials.@polyvar w
    p_pos = 2.0w + 0.0
    force_one_poly = AbstractConstraint[ExactlyOneConstraint([1], 1)]
    poly_pos_obj(x) = p_pos([w] => x)

    E_ppos, psi_ppos = minimize(
      p_pos;
      constraints=force_one_poly,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E_ppos, psi_ppos, poly_pos_obj, force_one_poly, 2.0, [1])
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
        callback_solution = TenSolver.Solution{Float64}(
          deepcopy(mps),
          Float64[],
          Int[],
          Float64[],
          collect(1:length(mps)),
          0:1,
        )
        @test is_feasible(TenSolver.sample(callback_solution), constraints)
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

  @testset "Infeasible constraints report status, not exception" begin
    impossible = AbstractConstraint[SumConstraint([1, 2], [1, 1], 3; relation=:(==))]

    E, psi = @test_logs (:warn, r"empty feasible subspace") minimize(
      zeros(2, 2);
      constraints=impossible,
      verbosity=0,
    )

    @test E == Inf
    @test !is_feasible(psi)
    @test isempty(psi.energies)
    @test isempty(psi.bond_dims)
    @test isempty(psi.elapsed_times)
    # The solve reports; querying a nonexistent solution throws.
    @test_throws DomainError TenSolver.sample(psi)
    @test [0, 0] ∉ psi

    # The supremum over an empty feasible set is -Inf.
    Emax, psimax = @test_logs (:warn, r"empty feasible subspace") maximize(
      zeros(2, 2);
      constraints=impossible,
      verbosity=0,
    )

    @test Emax == -Inf
    @test !is_feasible(psimax)
  end

  @testset "Infeasible single-variable solves report status" begin
    impossible = AbstractConstraint[SumConstraint([1], [1], 2; relation=:(==))]

    E, psi = @test_logs (:warn, r"empty feasible subspace") minimize(
      ones(1, 1);
      constraints=impossible,
      verbosity=0,
    )

    @test E == Inf
    @test !is_feasible(psi)
    @test_throws DomainError TenSolver.sample(psi)
  end

  @testset "Infeasible solutions map to MOI.INFEASIBLE" begin
    impossible = AbstractConstraint[SumConstraint([1, 2], [1, 1], 3; relation=:(==))]
    _, psi = @test_logs (:warn, r"empty feasible subspace") minimize(
      zeros(2, 2);
      constraints=impossible,
      verbosity=0,
    )

    termination_status, status = TenSolver.tensolver_status(psi; iterations=10, time_limit=Inf)

    @test termination_status == TenSolver.MOI.INFEASIBLE
    @test status == "infeasible"
  end

  @testset "Knapsack: maximize value under a capacity SumConstraint" begin
    # 0/1 knapsack as a constrained solve: minimize -value subject to a
    # weight budget. Weights/values chosen so the optimum is unique
    # (items 3 and 4: weight 5 + 2 = 7 == capacity, value 7 + 3 = 10).
    weights  = [3, 4, 5, 2]
    values   = [4.0, 5.0, 7.0, 3.0]
    capacity = 7
    constraints = AbstractConstraint[SumConstraint([1, 2, 3, 4], weights, capacity; relation=:(<=))]
    obj(x) = -dot(values, x)
    expected_energy, expected_sample = brute_force(obj, 4, constraints)

    @test expected_sample == [0, 0, 1, 1]

    E, psi = minimize(
      zeros(4, 4),
      -values;
      constraints,
      iterations=4,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E, psi, obj, constraints, expected_energy, expected_sample)
    @test sum(weights .* TenSolver.sample(psi)) <= capacity
  end

  @testset "SumConstraint over non-adjacent sites (n = 5)" begin
    # Exactly one of the odd-indexed sites {1, 3, 5} is selected. The linear
    # objective makes site 3 uniquely optimal among them and leaves the
    # unconstrained even sites at 0, so the optimum is the unique [0,0,1,0,0].
    l = [-1.0, 0.5, -3.0, 0.5, -2.0]
    constraints = AbstractConstraint[SumConstraint([1, 3, 5], [1, 1, 1], 1; relation=:(==))]
    obj(x) = dot(l, x)
    expected_energy, expected_sample = brute_force(obj, 5, constraints)

    @test expected_sample == [0, 0, 1, 0, 0]

    E, psi = minimize(
      zeros(5, 5),
      l;
      constraints,
      iterations=4,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
    )

    assert_constrained_solution(E, psi, obj, constraints, expected_energy, expected_sample)
  end

  @testset "Degenerate feasible optima are both represented in psi" begin
    # Exactly-one over two sites with a symmetric objective: [1, 0] and [0, 1]
    # are both optimal with E = -1. `mindim = 2` keeps the MPS from truncating
    # to a single bitstring — the same mechanism test/cases/vrp.jl uses to
    # force positive probability for both optimal solutions. The optimum split
    # inherits the random initial state, so seed for reproducibility; the
    # membership cutoff (1e-8) sits orders of magnitude below the observed
    # worst-case split (~1e-4 over 30 seeds).
    Random.seed!(20260715)

    l = [-1.0, -1.0]
    constraints = AbstractConstraint[ExactlyOneConstraint([1, 2], 1)]

    E, psi = minimize(
      zeros(2, 2),
      l;
      constraints,
      iterations=3,
      verbosity=0,
      cutoff=1e-12,
      noise=[0.0],
      mindim=2,
    )

    @test E ≈ -1.0
    @test [1, 0] in psi
    @test [0, 1] in psi
    # Infeasible states carry no amplitude, so the two optima carry all of it.
    @test [0, 0] ∉ psi
    @test [1, 1] ∉ psi
    @test TenSolver.prob(psi, [1, 0]) + TenSolver.prob(psi, [0, 1]) ≈ 1.0 atol=1e-8

    # The constrained scalar fast path is exactly deterministic: with both
    # assignments feasible and degenerate it returns the uniform superposition.
    relaxed = AbstractConstraint[SumConstraint([1], [1], 1; relation=:(<=))]
    E1, psi1 = minimize(
      zeros(1, 1);
      constraints=relaxed,
      iterations=2,
      verbosity=0,
      cutoff=1e-12,
    )

    @test E1 ≈ 0.0
    @test [0] in psi1
    @test [1] in psi1
    @test TenSolver.prob(psi1, [0]) ≈ 0.5
    @test TenSolver.prob(psi1, [1]) ≈ 0.5
  end
end
