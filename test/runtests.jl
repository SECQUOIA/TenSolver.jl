using Test, Random, LinearAlgebra

import ITensors
using  TenSolver

# Makes ITensor slower but catches more errors. Good for development.
ITensors.enable_debug_checks()

filepath(x) = joinpath(dirname(@__FILE__), x)


@testset "Correctness" begin
  dim = 5
  Q = randn(dim, dim)

  # TenSolver solution
  e, psi = TenSolver.solve_qubo(Q)
  x = sample_solution(psi)

  # Does the ground energy match solution?
  @test dot(x, Q, x) ≈ e

  for i in 1:10
    y = rand(Bool, dim)
    @test dot(y, Q, y) >= e
  end

  # Exact solution
  e0, x0 = TenSolver.brute_force_qubo(Q)
  # Same minimum value
  @test e ≈ e0
  # Same solution
  @test x == x0

end
