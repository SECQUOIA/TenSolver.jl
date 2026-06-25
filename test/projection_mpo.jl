import ITensors, ITensorMPS

function _projection_diagonal(H, sites)
  diagonal = Dict{Vector{Int},Float64}()

  for assignment in Iterators.product(fill(0:1, length(sites))...)
    bits = collect(assignment)
    psi = ITensorMPS.MPS(sites, string.(bits))
    diagonal[bits] = real(ITensors.inner(adjoint(psi), H, psi))
  end

  return diagonal
end

@testset "Projection MPOs" begin
  sites = ITensors.siteinds("Qudit", 3; dim=2)

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

  @testset "Manual diagonal masks" begin
    constraint = not_equals_constraint([1, 3], [1, 0])
    H = TenSolver.projection_mpo(constraint, sites)
    diagonal = _projection_diagonal(H, sites)

    for bits in keys(diagonal)
      expected = is_feasible(bits, constraint) ? 1.0 : 0.0
      @test diagonal[bits] ≈ expected
    end
  end

  @testset "Feasible states are preserved" begin
    constraints = AbstractConstraint[
      exactly_one_constraint([1, 2, 3]),
      relation_constraint(1, Symbol("<="), 2),
      sum_constraint([1, 3], [2, 1], Symbol("<="), 2),
    ]
    projections = TenSolver.projection_mpos(constraints, sites)

    @test length(projections) == length(constraints)

    for (constraint, H) in zip(constraints, projections)
      diagonal = _projection_diagonal(H, sites)

      for bits in keys(diagonal)
        expected = is_feasible(bits, constraint) ? 1.0 : 0.0
        @test diagonal[bits] ≈ expected
      end
    end
  end

  @testset "Infeasible constraints build zero masks" begin
    constraint = sum_constraint([1, 2], [1, 1], Symbol("<"), 0)
    H = TenSolver.projection_mpo(constraint, sites)
    diagonal = _projection_diagonal(H, sites)

    @test all(iszero, values(diagonal))
  end
end
