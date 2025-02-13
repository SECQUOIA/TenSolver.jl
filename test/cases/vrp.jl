using LinearAlgebra

# See https://secquoia.github.io/QUBOBook/QUBO%20and%20Ising%20Models.html
function vrp_6nodes()
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
  Q, beta = vrp_6nodes()

  E, psi = minimize(Q, beta; iterations = 40, cutoff = 1e-12, mindim = 2)
  x = TenSolver.sample(psi)

  # Known Solution
  @test E ≈ 5.0

  @test [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] in psi
  @test [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] in psi
  @test x in ( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
             , [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
             )
end
