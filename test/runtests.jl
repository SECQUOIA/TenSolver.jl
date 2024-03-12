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
  e, psi = TenSolver.solve_qubo(Q; cutoff = 1e-16)
  x = TenSolver.sample_solution(psi)

  # Does the ground energy match solution?
  @test dot(x, Q, x) ≈ e

  for i in 1:10
    y = rand(Bool, dim)
    @test dot(y, Q, y) >= e - 1e-8 # A small gap to amount for floating errors
  end

  # ~:~ Exact solution ~:~ #

  e0, x0 = TenSolver.brute_force_qubo(Q)
  # Same minimum value
  @test e ≈ e0
  # Same solution
  @test x == x0

end
