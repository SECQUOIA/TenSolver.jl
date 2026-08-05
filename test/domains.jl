import DynamicPolynomials as DP
import TenSolver as TS
using LinearAlgebra

@testset "Domain Simplification" begin
  domains = map(d -> TS.canonicalize_domain(d, 3), [
    [0, 1],
    [-1, 1],
    [-3, 4, 5.7],
    [[0, 1], [-1, 1], [-3, 4, 5.7]],
   ])

  @testset "Quadratic Objectives" begin
    Q, l, c = TS.domain_residue([1 0; 0 1.0], [0.0, 0.0], 0.0, TS.canonicalize_domain([-1, 1], 2))
    @test iszero(Q)
    @test iszero(l)
    @test c ≈ 2

    for domain in domains
      Q, l, c = randn(3, 3), randn(3), randn()
      Qr, lr, cr = TS.domain_residue(Q, l, c, domain)
      obj(x) = dot(x, Q, x) + dot(l, x) + c
      obj_r(x) = dot(x, Qr, x) + dot(lr, x) + cr

      if all(length.(domain) .== 2)
        @test iszero(Diagonal(Qr))
      end

      for x in domain[1], y in domain[2], z in domain[3]
        @test obj([x, y, z]) ≈ obj_r([x, y, z])
      end
    end
  end

  @testset "Polynomial Objectives" begin
    DP.@polyvar x[1:3]

    for (deg, domain) in Iterators.product([2, 3, 5], domains)
      p = randpoly(x, deg)
      q = TS.domain_residue(p, domain)

      for v in DP.effective_variables(q)
        @test DP.maxdegree(q, v) <= maximum(length, domain)
      end

      for a in domain[1], b in domain[2], d in domain[3]
        @test p(a, b, d) ≈ q(a, b, d)
      end
    end
  end
end

@testset "Non-binary domains" begin
  @testset "Unconstrained quadratic" begin
    Q = [
      2.0  -1.0   0.0
     -1.0   3.0  -1.0
      0.0  -1.0   2.0
    ]

    E, psi = minimize(Q; domain = 0:2, iterations = 5, cutoff = 1e-12, verbosity = 0)

    @test E ≈ 0.0
    @test [0, 0, 0] in psi
  end

  @testset "Unconstrained linear" begin
    l = [-1.0,  2.0, -3.0]

    E, psi = minimize(l; domain = 0:2, iterations = 5, cutoff = 1e-12, verbosity = 0)

    @test E ≈ -8
    @test [2, 0, 2] in psi
  end

  @testset "Unconstrained quadratic + linear" begin
    Q = [
      1.5   0.5  -0.5
      0.5   2.0   0.25
     -0.5   0.25  1.0
    ]
    l = [-2.0, -1.0, -3.0]

    E, psi = minimize(Q, l; domain = 0:2, iterations = 5, cutoff = 1e-12, verbosity = 0)

    @test E ≈ -4.5
    @test [1, 0, 2] in psi
  end

  @testset "Single site case" begin
    Q = reshape([-2.0], 1, 1)
    l = [3.0]
    c = 5.0

    E, psi = minimize(Q, l, c; domain = 0:2, verbosity = 0)

    @test E ≈ 3.0
    @test [2] in psi
  end

  @testset "Small polynomial case" begin
    DP.@polyvar y[1:3]
    p = y[1]^2 + y[1] * y[2] + 2y[2]^2 - y[2] * y[3] - 3.0y[3]

    E, psi = minimize(p; domain = 0:2, iterations = 10, mindim = 5, cutoff = 1e-8, verbosity = 0)

    @test E ≈ -6.0
    @test [0, 0, 2] in psi
    @test sample(psi) in ([0, 0, 2], [0, 1, 2])
  end

  @testset "Polynomial exponents are preserved" begin
    DP.@polyvar y[1:2]
    p = 2.0y[1]^2 - 3.0y[1] + 2.0y[2]^2 - 3.0y[2]

    E, psi = maximize(p; domain = 0:2, iterations = 5, cutoff = 1e-12, verbosity = 0)

    @test E ≈ 4.0
    @test [2, 2] in psi
    @test [0, 0] ∉ psi
  end

  @testset "Constrained quadratic + linear" begin
    Q = [
      1.0   0.5   0.0
      0.5   1.5  -0.5
      0.0  -0.5   1.0
    ]
    l = [-3.0, -2.0, -1.0]
    constraints = AbstractConstraint[
      SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=))
    ]

    E, psi = minimize(
      Q,
      l;
      constraints,
      domain = 0:2,
      iterations = 5,
      cutoff = 1e-8,
      mindim = 10,
      verbosity = 0,
    )

    expected_sample = [1, 0, 0]
    @test E ≈ -2.0
    @test expected_sample in psi
    @test is_feasible(sample(psi), constraints)
  end

  @testset "Zero domain" begin
    Q = [-1.0 0.0; 1.0 2.0]
    E, psi = minimize(Q; domain = [0], verbosity = 0)
    x = TenSolver.sample(psi)

    @test E == 0
    @test [0, 0] in psi
    @test x == [0, 0]
  end

  @testset "Domain API and validation" begin
    Q = [-1.0 0.0; 0.0 2.0]

    E_default, psi_default = minimize(Q; iterations = 4, verbosity = 0)
    E_bool, psi_bool = minimize(Q; domain = [0, 1], iterations = 4, verbosity = 0)

    @test E_default ≈ E_bool
    @test sample(psi_default) == [1, 0]
    @test [1, 0] in psi_bool

    E_ordered, psi_ordered = minimize([1.0]; domain = [1, -1], verbosity = 0)
    @test E_ordered ≈ -1.0
    @test sample(psi_ordered) == [-1.0]
    @test [-1] in psi_ordered
    @test [1] ∉ psi_ordered
    @test_throws DomainError [2] in psi_ordered

    E_sparse, psi_sparse = minimize(
      [1.0, -4.0, 2.0];
      domain = [-2, 0, 3],
      iterations = 4,
      verbosity = 0,
    )

    @test E_sparse ≈ -18.0
    @test sample(psi_sparse) == [-2.0, 3.0, -2.0]
    @test [-2.0, 3.0, -2.0] in psi_sparse
    @test [0.0, 0.0, 0.0] ∉ psi_sparse

    @test_throws ArgumentError minimize(Q; domain = Int[], verbosity = 0)
    @test_throws MethodError minimize(Q; domain = ["0", "1"], verbosity = 0)
  end
end

@testset "Fractional domains" begin
  @testset "Unconstrained objective" begin
    domain = [0.0, 0.5, 1.0]
    l = [-2.0, 3.0]

    E, psi = minimize(l; domain, verbosity=0)
    x = sample(psi)

    @test E ≈ -2.0
    @test x == [1.0, 0.0]
    @test [1.0, 0.0] in psi
  end

  @testset "SumConstraint fails fast" begin
    capacity = SumConstraint([1], [1], 1; relation=:(<=))
    @test_throws ArgumentError minimize([0.0]; domain=[0.0, 0.5, 1.0], constraints=[capacity], verbosity=0)
  end
end

@testset "Per-variable domains" begin
  domain = [0:1, 0:2, [-1, 1, 3.5]]
  l = [-1.0, 3.0, -5.0]

  E, psi = minimize(l; domain, verbosity=0)
  x = sample(psi)

  @test E ≈ -18.5
  @test x == [1.0, 0.0, 3.5]
  @test [1.0, 0.0, 3.5] in psi
end
