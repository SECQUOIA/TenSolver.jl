import DynamicPolynomials, TypedPolynomials
import MultivariatePolynomials: maxdegree
const DP = DynamicPolynomials
const TP = TypedPolynomials

form(a, x) = sum(a[t] * prod(x[i] for i in Tuple(t)) for t in CartesianIndices(a))

function randpoly(x, maxdegree)
  dim = length(x)
  mkarray(i) = randn(Iterators.repeated(dim, i)...)

  return sum(form(mkarray(i), x) for i in 1:maxdegree) + randn()
end

function test_correctness(dim, obj, args...)
    # TenSolver solution
    e, psi = TenSolver.minimize(args...)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test obj(x) ≈ e

    for _ in 1:10
      y = rand(Bool, dim)
      @test obj(y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #
    e0, x0 = brute_force(Float64, obj, dim)
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
