import ITensors, ITensorMPS

function projection_diagonal(H, sites)
  diagonal = Dict{Vector{Int},Float64}()

  for assignment in Iterators.product(fill(0:1, length(sites))...)
    bits = collect(assignment)
    psi = ITensorMPS.MPS(sites, string.(bits))
    diagonal[bits] = real(ITensors.inner(adjoint(psi), H, psi))
  end

  return diagonal
end

function assert_projection_matches_feasibility(constraint, sites)
  H = TenSolver.projection_mpo(constraint, sites)
  diagonal = projection_diagonal(H, sites)

  for bits in keys(diagonal)
    expected = is_feasible(bits, constraint) ? 1.0 : 0.0
    @test diagonal[bits] ≈ expected
  end

  return H
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

  @testset "Sparse ITensor construction" begin
    s1, s2 = sites[1], sites[2]
    s1p, s2p = ITensors.prime(s1), ITensors.prime(s2)
    tensor = TenSolver.itensor_from_nonzeros(
      Float64,
      (s1, s1p, s2, s2p),
      [((1, 1, 2, 2), 3.0), ((2, 2, 1, 1), 5.0)],
    )
    materialized = Array(tensor, s1, s1p, s2, s2p)

    @test materialized[1, 1, 2, 2] == 3.0
    @test materialized[2, 2, 1, 1] == 5.0
    @test count(!iszero, materialized) == 2
  end

  @testset "Entry value on a pass-through leading site" begin
    # Regression: the entry `value` must survive even when the first register
    # site is a pass-through site. Here only site 2 is constrained, so sites 1
    # and 3 pass through; the coefficient must be anchored on site 2.
    entry = TenSolver.SparseTensorEntry(Dict(2 => 2), 7.0)
    H = TenSolver.tensor_to_mpo(Float64, [entry], sites)
    diagonal = projection_diagonal(H, sites)

    for (bits, value) in diagonal
      expected = bits[2] == 1 ? 7.0 : 0.0
      @test value ≈ expected
    end
  end

  @testset "Validation errors" begin
    s1 = sites[1]
    s1p = ITensors.prime(s1)
    qutrit_sites = ITensors.siteinds("Qudit", 1; dim=3)
    too_large_int = big(typemax(Int)) + 1

    @test_throws DimensionMismatch TenSolver.itensor_from_nonzeros(
      Float64,
      (s1, s1p),
      [((1, 1, 1), 1.0)],
    )
    @test_throws ArgumentError TenSolver.tensor_to_mpo(Float64, [], ITensors.Index{Int64}[])
    @test_throws ArgumentError TenSolver.tensor_to_mpo(Float64, [], qutrit_sites)
    @test_throws BoundsError TenSolver.projection_mpo(ExactlyOneConstraint([4]), sites)
    @test_throws ArgumentError TenSolver.projection_mpo(
      SumConstraint([1], [1], Symbol("<="), 1),
      qutrit_sites,
    )
    @test_throws ArgumentError TenSolver.projection_mpo(
      SumConstraint([1], [0.5], Symbol("<="), 1),
      sites,
    )
    @test_throws ArgumentError TenSolver.projection_mpo(
      SumConstraint([1], [1], Symbol("<="), 1.5),
      sites,
    )
    @test_throws ArgumentError TenSolver.projection_mpo(
      SumConstraint([1], [too_large_int], Symbol("<="), 1),
      sites,
    )
    @test_throws ArgumentError TenSolver.projection_mpo(
      SumConstraint([1, 2], [typemax(Int), 1], Symbol(">="), typemax(Int)),
      sites,
    )
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
    @test diagonal[[1, 1, 0, 0]] == 1.0
    @test diagonal[[0, 0, 0, 1]] == 1.0
    @test diagonal[[1, 0, 1, 0]] == 0.0
    @test diagonal[[0, 1, 0, 1]] == 0.0
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
        expected = is_feasible(bits, constraint) ? 1.0 : 0.0
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
