import DynamicPolynomials as DP
import TypedPolynomials   as TP
import MultivariatePolynomials: maxdegree, effective_variables

form(a, x) = sum(a[t] * prod(x[i] for i in Tuple(t)) for t in CartesianIndices(a))

function randpoly(x, maxdegree)
  dim = length(x)
  mkarray(i) = randn(Iterators.repeated(dim, i)...)

  return sum(form(mkarray(i), x) for i in 1:maxdegree) + randn()
end

function test_correctness(dim, obj, args...)
    # TenSolver solution
    e, psi = TenSolver.minimize(args...; verbosity = 0)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test obj(x) ≈ e

    for _ in 1:10
      y = rand(Bool, dim)
      @test obj(y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #
    e0, x0 = brute_force(obj, dim)
    # Same minimum value
    @test e ≈ e0
    # Solution is sampleable
    @test x0 in psi
end

@testset "DynamicPolynomials.jl" begin
  dim = 5
  DP.@polyvar x[1:dim]

  @testset "Quadratic" begin
    p = randpoly(x, 2)
    @test maxdegree(p) == 2
    test_correctness(dim, a -> p(x => a), p)
  end

  @testset "Cubic" begin
    p = randpoly(x, 3)
    @test maxdegree(p) == 3
    test_correctness(dim, a -> p(x => a), p)
  end

  @testset "Domain Simplification" begin
    domains = [[0, 1], [-1, 1], [-3, 4, 5.7]]
    for (deg, domain) in Iterators.product([2, 3, 5], domains)
      p = randpoly(x, deg)
      q = TenSolver.simplify_polynomial(p, domain)

      for v in effective_variables(q)
        @test maxdegree(q, v) <= length(domain)
      end
    end
  end
end

@testset "TypedPolynomials.jl" begin
  dim = 5
  TP.@polyvar x[1:5]

  @testset "Quadratic" begin
    p = randpoly(x, 2)
    @test maxdegree(p) == 2
    test_correctness(dim, a -> p(x => a), p)
  end

  @testset "Cubic" begin
    p = randpoly(x, 3)
    @test maxdegree(p) == 3
    test_correctness(dim, a -> p(x => a), p)
  end
end
