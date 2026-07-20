import DynamicPolynomials

@testset "Spin domains" begin
  spin_domain = [-1, 1]

  @testset "Quadratic objective" begin
    J = [
       0.5  -1.0   0.25
       0.0   0.0  -0.75
       0.5   0.0   1.0
    ]
    h = [0.25, -1.5, 0.75]
    offset = 2.0
    obj(s) = dot(s, J, s) + dot(h, s) + offset

    E, psi = minimize(
      J,
      h,
      offset;
      domain = spin_domain,
      iterations = 8,
      mindim = 8,
      cutoff = 1e-12,
      verbosity = 0,
    )
    E0, s0 = brute_force(obj, 3; domain = spin_domain)

    @test E ≈ E0
    @test s0 in psi
    @test all(in(spin_domain), sample(psi))
    @test [0, 0, 0] ∉ psi
  end

  @testset "Single-site and maximize" begin
    J = reshape([2.0], 1, 1)
    h = [-3.0]
    obj(s) = dot(s, J, s) + dot(h, s)

    E, psi = minimize(J, h; domain = (-1, 1), verbosity = 0)
    Emax, psimax = maximize(J, h; domain = spin_domain, verbosity = 0)

    @test E ≈ obj([1])
    @test sample(psi) == [1]
    @test Emax ≈ obj([-1])
    @test sample(psimax) == [-1]
  end

  @testset "Polynomial objective" begin
    DynamicPolynomials.@polyvar s[1:3]
    p = 1.5s[1] * s[2] - 2.0s[2] * s[3] + 0.5s[1]^2 + s[3]
    obj(x) = real(p(s => x))

    E, psi = minimize(
      p;
      domain = spin_domain,
      iterations = 8,
      mindim = 8,
      cutoff = 1e-12,
      verbosity = 0,
    )
    E0, s0 = brute_force(obj, 3; domain = spin_domain)

    @test E ≈ E0
    @test s0 in psi
  end

  @testset "Constraints use physical Spin values" begin
    J = [
      0.0  0.0  0.5
      0.0  0.0 -0.5
      0.5 -0.5  0.0
    ]
    h = [-3.0, -2.0, -1.0]
    constraints = AbstractConstraint[
      SumConstraint([1, 2, 3], [1, 2, 3], 0; relation = :(==)),
      NotEqualsConstraint([1, 2], [1, 1]),
    ]
    obj(s) = dot(s, J, s) + dot(h, s)

    E, psi = minimize(
      J,
      h;
      domain = spin_domain,
      constraints,
      preprocess = true,
      iterations = 8,
      mindim = 8,
      cutoff = 1e-12,
      verbosity = 0,
    )
    E0, s0 = brute_force(obj, 3, constraints; domain = spin_domain)
    sampled = sample(psi)

    @test E ≈ E0
    @test s0 in psi
    @test psi.permutation != collect(1:3)
    @test is_feasible(sampled, constraints)
    @test all(in(spin_domain), sampled)
  end

  @testset "Infeasible Spin constraints preserve the domain" begin
    impossible = AbstractConstraint[
      SumConstraint([1], [1], 2; relation = :(==)),
    ]

    E, psi = minimize([0.0]; domain = spin_domain, constraints = impossible, verbosity = 0)

    @test E == Inf
    @test !is_feasible(psi)
    @test psi.domain == spin_domain
    @test [1] ∉ psi
    @test_throws DomainError sample(psi)
  end

  @testset "Domain validation and compatibility" begin
    Q = [-1.0 0.0; 0.0 2.0]

    E_default, psi_default = minimize(Q; iterations = 4, verbosity = 0)
    E_bool, psi_bool = minimize(Q; domain = [0, 1], iterations = 4, verbosity = 0)
    E_dim, psi_dim = minimize(Q; domain_dim = 2, iterations = 4, verbosity = 0)

    @test E_default ≈ E_bool ≈ E_dim
    @test sample(psi_default) == [1, 0]
    @test [1, 0] in psi_bool
    @test [1, 0] in psi_dim

    @test_throws ArgumentError minimize(Q; domain = [0, 2], verbosity = 0)
    @test_throws ArgumentError minimize(Q; domain = [1, -1], verbosity = 0)
    @test_throws ArgumentError minimize(Q; domain = [-1, 1], domain_dim = 2, verbosity = 0)
  end
end
