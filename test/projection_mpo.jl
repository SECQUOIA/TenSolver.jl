import ITensors, ITensorMPS

function projection_diagonal(H, sites)
  diagonal = Dict{Tuple,Float64}()

  for assignment in Iterators.product(fill(0:1, length(sites))...)
    bits = Tuple(assignment)
    psi = ITensorMPS.MPS(sites, string.(bits))
    diagonal[bits] = real(ITensors.inner(adjoint(psi), H, psi))
  end

  return diagonal
end

function assert_projection_matches_feasibility(constraint, sites)
  H = TenSolver.projection_mpo(constraint, sites)
  diagonal = projection_diagonal(H, sites)

  for bits in keys(diagonal)
    expected = is_feasible(collect(bits), constraint) ? 1.0 : 0.0
    @test diagonal[bits] ≈ expected
  end

  return H
end

function dfa_diagonal(H, sites)
  diagonal = Dict{Tuple,Float64}()

  for assignment in Iterators.product(fill(0:1, length(sites))...)
    bits = Tuple(assignment)
    psi = ITensorMPS.MPS(sites, string.(bits))
    diagonal[bits] = real(ITensors.inner(adjoint(psi), H, psi))
  end

  return diagonal
end

function exactly_one_transition_table()
  return Dict{Tuple{Int,Int},Int}(
    (0, 0) => 0,
    (0, 1) => 1,
    (1, 0) => 1,
    (1, 1) => 2,
    (2, 0) => 2,
    (2, 1) => 2,
  )
end

@testset "DFA construction" begin
  sites = ITensors.siteinds("Qudit", 3; dim=2)

  @testset "dfa_to_mpo exact diagonal" begin
    dfa = TenSolver.DFA(
      [0, 1, 2],
      [0, 1],
      0,
      Set([1]),
      [exactly_one_transition_table() for _ in eachindex(sites)],
    )

    H = TenSolver.dfa_to_mpo(Float64, dfa, sites)
    diagonal = dfa_diagonal(H, sites)

    for (bits, value) in diagonal
      expected = count(==(1), bits) == 1 ? 1.0 : 0.0
      @test value ≈ expected
    end

    @test count(==(1.0), values(diagonal)) == 3
    @test ITensorMPS.maxlinkdim(H) == 3
  end
end

@testset "Projection MPOs" begin
  sites = ITensors.siteinds("Qudit", 3; dim=2)

  @testset "Projection API inference" begin
    constraint = ExactlyOneConstraint([1, 2])
    constraints = AbstractConstraint[
      constraint,
      RelationConstraint(1, Symbol("<="), 2),
    ]

    typed_mpo = @inferred TenSolver.projection_mpo(Float64, constraint, sites)
    default_mpo = @inferred TenSolver.projection_mpo(constraint, sites)
    typed_mpos = @inferred TenSolver.projection_mpos(Float64, constraints, sites)
    default_mpos = @inferred TenSolver.projection_mpos(constraints, sites)

    @test typed_mpo isa ITensorMPS.MPO
    @test default_mpo isa ITensorMPS.MPO
    @test typed_mpos isa Vector{ITensorMPS.MPO}
    @test default_mpos isa Vector{ITensorMPS.MPO}
  end

  @testset "Constraint -> DFA -> MPO" begin
    constraint = NotEqualsConstraint([1, 3], [1, 0])
    dfa = TenSolver.constraint_to_dfa(Float64, constraint, sites)
    H = TenSolver.dfa_to_mpo(Float64, dfa, sites)
    diagonal = projection_diagonal(H, sites)

    for bits in keys(diagonal)
      expected = is_feasible(collect(bits), constraint) ? 1.0 : 0.0
      @test diagonal[bits] ≈ expected
    end
  end

  @testset "Manual diagonal masks" begin
    constraint = NotEqualsConstraint([1, 3], [1, 0])
    assert_projection_matches_feasibility(constraint, sites)
  end

  @testset "SumConstraint automaton masks" begin
    automaton_sites = ITensors.siteinds("Qudit", 4; dim=2)
    constraints = [
      SumConstraint([1, 3], [2, 1], Symbol("<="), 2),
      SumConstraint([2, 4], [2, 3], Symbol(">="), 3),
      SumConstraint([1, 2, 4], [1, 2, 3], Symbol("=="), 3),
      SumConstraint([1, 2, 4], [1, 2, 3], Symbol("!="), 3),
    ]

    for constraint in constraints
      assert_projection_matches_feasibility(constraint, automaton_sites)
    end
  end

  @testset "SumConstraint prunes dead partial sums" begin
    pruning_sites = ITensors.siteinds("Qudit", 6; dim=2)
    le_constraint = SumConstraint(1:6, [1, 2, 4, 8, 16, 32], Symbol("<="), 3)
    eq_constraint = SumConstraint(1:6, [1, 2, 4, 8, 16, 32], Symbol("=="), 3)

    le_projection = assert_projection_matches_feasibility(le_constraint, pruning_sites)
    eq_projection = assert_projection_matches_feasibility(eq_constraint, pruning_sites)

    @test ITensorMPS.maxlinkdim(le_projection) <= 4
    @test ITensorMPS.maxlinkdim(eq_projection) <= 4
  end

  @testset "Knapsack-style capacity mask" begin
    knapsack_sites = ITensors.siteinds("Qudit", 4; dim=2)
    constraint = SumConstraint(1:4, [2, 3, 4, 5], 5; relation=Symbol("<="))
    H = assert_projection_matches_feasibility(constraint, knapsack_sites)
    diagonal = projection_diagonal(H, knapsack_sites)

    @test count(==(1.0), values(diagonal)) == 6
    @test diagonal[(1, 1, 0, 0)] == 1.0
    @test diagonal[(0, 0, 0, 1)] == 1.0
    @test diagonal[(1, 0, 1, 0)] == 0.0
    @test diagonal[(0, 1, 0, 1)] == 0.0
  end

  @testset "Feasible states are preserved" begin
    constraints = AbstractConstraint[
      ExactlyOneConstraint([1, 2, 3]),
      RelationConstraint(1, Symbol("<="), 2),
      SumConstraint([1, 3], [2, 1], Symbol("<="), 2),
    ]
    projections = TenSolver.projection_mpos(constraints, sites)

    @test length(projections) == length(constraints)

    for (constraint, H) in zip(constraints, projections)
      diagonal = projection_diagonal(H, sites)

      for bits in keys(diagonal)
        expected = is_feasible(collect(bits), constraint) ? 1.0 : 0.0
        @test diagonal[bits] ≈ expected
      end
    end
  end

  @testset "Infeasible constraints build zero masks" begin
    constraint = SumConstraint([1, 2], [1, 1], Symbol("<="), -1)
    H = TenSolver.projection_mpo(constraint, sites)
    diagonal = projection_diagonal(H, sites)

    @test all(iszero, values(diagonal))
  end
end
