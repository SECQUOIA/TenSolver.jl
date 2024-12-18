using LinearAlgebra

function vrp_simple()
  A = Float64[
      1 0 0 1 1 1 0 1 1 1 1
      0 1 0 1 0 1 1 0 1 1 1
      0 0 1 0 1 0 1 1 1 1 1
  ]
  b = Float64[1, 1, 1]
  c = Float64[2, 4, 4, 4, 4, 4, 5, 4, 5, 6, 5];

  ϵ = 1
  ρ = sum(abs, c) + ϵ
  Q = diagm(c) + ρ * (A'A - 2 * diagm(A'b))
  β = ρ * b'b

  return Q, β
end

@testset "Vehicle Routing Problem (VRP)" begin
  # See https://secquoia.github.io/QUBOBook/QUBO%20and%20Ising%20Models.html
  Q, beta = vrp_simple()

  E, psi = solve(Q, nothing, beta)
  x = TenSolver.sample(psi)

  # Known Solution
  @test E ≈ 5.0 atol=1e-4

  @test [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] in psi
  @test [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] in psi
  @test x in ( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
             , [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
             )
end
