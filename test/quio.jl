import DynamicPolynomials

@testset "Non-binary domains" begin
  @testset "Unconstrained quadratic" begin
    Q = [
      2.0  -1.0   0.0
     -1.0   3.0  -1.0
      0.0  -1.0   2.0
    ]
    obj(x) = dot(x, Q, x)

    E, psi = minimize(Q; domain_dim = 3, iterations = 5, cutoff = 1e-12, verbosity = 0)

    @test E ≈ 0.0
    @test [0, 0, 0] in psi
  end

  @testset "Unconstrained linear" begin
    l = [-1.0,  2.0, -3.0]
    obj(x) = dot(l, x)

    E, psi = minimize(l; domain_dim = 3, iterations = 5, cutoff = 1e-12, verbosity = 0)
    E0, x0 = brute_force(obj, 3; domain = 0:2)

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
    obj(x) = dot(x, Q, x) + dot(l, x)

    E, psi = minimize(Q, l; domain_dim = 3, iterations = 5, cutoff = 1e-12, verbosity = 0)
    E0, x0 = brute_force(obj, 3; domain = 0:2)

    @test E ≈ E0
    @test x0 in psi
  end

  @testset "Single site case" begin
    Q = reshape([-2.0], 1, 1)
    l = [3.0]
    c = 5.0
    obj(x) = dot(x, Q, x) + dot(l, x)

    E, psi = minimize(Q, l, c; domain_dim = 3, verbosity = 0)
    E0, x0 = brute_force(obj, 1; domain = 0:2)

    @test E ≈ 3.0
    @test [2] in psi
  end

  @testset "Small polynomial case" begin
    DynamicPolynomials.@polyvar y[1:3]
    p = y[1]^2 + y[1] * y[2] + 2y[2]^2 - y[2] * y[3] - 3.0y[3]
    obj(x) = p(y => x)

    E, psi = minimize(p; domain_dim = 3, iterations = 10, mindim = 5, cutoff = 1e-8, verbosity = 0)
    E0, x0 = brute_force(obj, 3; domain = 0:2)

    @test E ≈ E0
    @test x0 in psi
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
    obj(x) = dot(x, Q, x) + dot(l, x)

    E, psi = minimize(
      Q,
      l;
      constraints,
      domain_dim = 3,
      iterations = 5,
      cutoff = 1e-8,
      mindim = 10,
      verbosity = 0,
    )

    @test E ≈ -2.0
  end
end

