import DynamicPolynomials

@testset "GenericTensorNetworks backend" begin
  has_gtn = true
  try
    import GenericTensorNetworks
    import ProblemReductions
  catch err
    has_gtn = false
    @test_skip "GenericTensorNetworks extension dependencies are not available: $err"
  end

  if has_gtn
    @testset "QUBO optimum, count, and enumeration" begin
      Q = [0.0 2.0; 0.0 0.0]
      l = [-1.0, -1.0]
      c = 3.0
      optimum_configs = [[1, 0], [0, 1]]

      E, sol = TenSolver.minimize(Q, l, c; backend=TenSolver.GTNBackend())
      @test sol isa TenSolver.GTNSolution
      @test E ≈ 2.0
      @test TenSolver.sample(sol) in optimum_configs
      @test sol.objective ≈ E

      Emax, max_sol = TenSolver.maximize(Q, l, c; backend=TenSolver.GTNBackend())
      @test Emax ≈ 3.0
      @test TenSolver.sample(max_sol) in [[0, 0], [1, 1]]
      @test max_sol.objective ≈ Emax

      E_count, count_sol = TenSolver.solution_space(Q, l, c; property=:count)
      @test E_count ≈ 2.0
      @test count_sol.metadata["count"] ≈ 2

      E_configs, config_sol = TenSolver.solution_space(Q, l, c; property=:configs)
      @test E_configs ≈ 2.0
      @test sort(config_sol.configs) == sort(optimum_configs)

      E_kbest, kbest_sol = TenSolver.solution_space(Q, l, c; backend=TenSolver.GTNBackend(property=:kbest_sizes, k=2))
      @test E_kbest ≈ 2.0
      @test sort(kbest_sol.metadata["size"]) ≈ [-1.0, -1.0]

      E_single_k, single_k_sol = TenSolver.solution_space(Q, l, c; backend=TenSolver.GTNBackend(property=:single, k=2))
      @test E_single_k ≈ 2.0
      @test sort(single_k_sol.configs) == sort(optimum_configs)

      @test_throws ArgumentError TenSolver.solution_space(Q, l, c; backend=TenSolver.GTNBackend(property=:configs, k=2))
      @test_throws ArgumentError TenSolver.solution_space(Q, l, c; backend=TenSolver.GTNBackend(property=:count, k=2))
    end

    @testset "Higher-order polynomial objective" begin
      DynamicPolynomials.@polyvar x[1:3]
      p = 1.0 * x[1] * x[2] * x[3] - 2.0 * x[1] + 0.5 * x[2] + 4.0
      obj(bits) = p(x => bits)
      exact, exact_config = brute_force(obj, Float64, 3)

      E, sol = TenSolver.minimize(p; backend=TenSolver.GTNBackend())
      @test E ≈ exact
      @test TenSolver.sample(sol) == exact_config
    end
  end
end
