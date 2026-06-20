@testset "Backend dispatch" begin
  @test_throws ArgumentError TenSolver.solution_space([0.0 0.0; 0.0 0.0]; backend=:dmrg)

  @test_throws ArgumentError TenSolver.minimize([1.0;;]; iteration=5)

  E, psi = TenSolver.minimize(
    [1.0;;];
    iterations=3,
    time_limit=0.0,
    vtol=1e-8,
    check_variance_every_iteration=1,
    inidim=2,
    maxdim=[2, 4],
    mindim=1,
    noise=[0.0],
    eigsolve_krylovdim=3,
    eigsolve_maxiter=2,
    eigsolve_tol=1e-14,
    verbosity=0,
  )

  @test E ≈ 0.0
  @test TenSolver.sample(psi) == [0]
end
