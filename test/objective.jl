import DynamicPolynomials

@testset "Canonical pseudo-Boolean objectives" begin
  @testset "QUBO diagonal, linear, and asymmetric terms" begin
    Q = [-1.0 2.0; 3.0 4.0]
    l = [0.5, -2.0]
    c = 1.25
    model = TenSolver.pseudoboolean(Q, l, c)
    obj(x) = dot(x, Q, x) + dot(l, x) + c

    for x in ([0, 0], [1, 0], [0, 1], [1, 1])
      @test TenSolver.evaluate(model, x) ≈ obj(x)
    end

    @test model.terms[[1]] ≈ -0.5
    @test model.terms[[2]] ≈ 2.0
    @test model.terms[[1, 2]] ≈ 5.0
  end

  @testset "Polynomial powers collapse on binary variables" begin
    DynamicPolynomials.@polyvar x[1:3]
    p = 3.0 * x[1]^4 * x[2]^2 - 2.0 * x[1] + 7.0
    model = TenSolver.pseudoboolean(p)

    @test model.constant ≈ 7.0
    @test model.terms[[1, 2]] ≈ 3.0
    @test model.terms[[1]] ≈ -2.0

    for bits in ([0, 0], [1, 0], [1, 1])
      @test TenSolver.evaluate(model, bits) ≈ p(model.variables => bits)
    end
  end
end
