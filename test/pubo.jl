import DynamicPolynomials as DP
import TypedPolynomials   as TP
import MultivariatePolynomials: maxdegree

function test_correctness(obj, expected_energy, expected_sample, args...)
    # TenSolver solution
    e, psi = TenSolver.minimize(args...; verbosity = 0)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test obj(x) ≈ e
    @test e ≈ expected_energy
    @test x == expected_sample
    @test expected_sample in psi
end

@testset "DynamicPolynomials.jl" begin
  DP.@polyvar x[1:3]
  # One fixed mixed-degree objective exercises linear, quadratic, and cubic
  # lowering. The cubic term changes the unique optimum from [1, 1, 1].
  p = 1.5 - 3x[1] - 2x[2] - x[3] + 0.5x[1] * x[2] +
      5x[1] * x[2] * x[3]

  @test maxdegree(p) == 3
  test_correctness(a -> p(x => a), -3.0, [1, 1, 0], p)
end

@testset "TypedPolynomials.jl" begin
  TP.@polyvar x[1:3]
  p = 1.5 - 3x[1] - 2x[2] - x[3] + 0.5x[1] * x[2] +
      5x[1] * x[2] * x[3]

  @test maxdegree(p) == 3
  test_correctness(a -> p(x => a), -3.0, [1, 1, 0], p)
end
