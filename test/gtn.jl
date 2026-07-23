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

      E, sol = TenSolver.minimize(Q, l, c; backend=:gtn)
      @test sol isa TenSolver.GTNSolution
      @test E ≈ 2.0
      @test TenSolver.sample(sol) in optimum_configs

      Q_active = [0.0 -5.0; 0.0 0.0]
      l_active = [2.0, 2.0]
      E_active, active_sol = TenSolver.minimize(Q_active, l_active; backend=:gtn)
      @test E_active ≈ -1.0
      @test TenSolver.sample(active_sol) == [1, 1]

      Emax, max_sol = TenSolver.maximize(Q, l, c; backend=:gtn)
      @test Emax ≈ 3.0
      @test TenSolver.sample(max_sol) in [[0, 0], [1, 1]]

      E_size, size_sol = TenSolver.minimize(Q, l, c; backend=:gtn, property=:size)
      @test E_size ≈ 2.0
      @test size_sol.metadata["size"] ≈ -1.0
      @test_throws ArgumentError TenSolver.sample(size_sol)

      E_count, count_sol = TenSolver.minimize(Q, l, c; backend=:gtn, property=:count)
      @test E_count ≈ 2.0
      @test count_sol.metadata["count"] ≈ 2

      E_configs, config_sol = TenSolver.minimize(Q, l, c; backend=:gtn, property=:configs)
      @test E_configs ≈ 2.0
      @test sort(config_sol.configs) == sort(optimum_configs)

      E_kbest, kbest_sol = TenSolver.minimize(
        Q,
        l,
        c;
        backend=:gtn,
        property=:kbest_sizes,
        k=2,
      )
      @test E_kbest ≈ 2.0
      @test sort(kbest_sol.metadata["size"]) ≈ [-1.0, -1.0]

      E_single_k, single_k_sol = TenSolver.minimize(
        Q,
        l,
        c;
        backend=:gtn,
        property=:single,
        k=2,
      )
      @test E_single_k ≈ 2.0
      @test sort(single_k_sol.configs) == sort(optimum_configs)

      @test_throws ArgumentError TenSolver.minimize(
        Q,
        l,
        c;
        backend=:gtn,
        property=:configs,
        k=2,
      )
      @test_throws ArgumentError TenSolver.minimize(
        Q,
        l,
        c;
        backend=:gtn,
        property=:count,
        k=2,
      )

      @test_throws ArgumentError TenSolver.minimize(Q, l, c; backend=:gtn, domain=[-1, 1])
      @test_throws ArgumentError TenSolver.minimize(
        Q,
        l,
        c;
        backend=:gtn,
        constraints=AbstractConstraint[ExactlyOneConstraint([1, 2], 1)],
      )
      @test_throws ArgumentError TenSolver.minimize(Q, l, c; backend=:gtn, verbosity=0)
    end

    @testset "Higher-order polynomial objective" begin
      DynamicPolynomials.@polyvar x[1:3]
      p = 1.0 * x[1] * x[2] * x[3] - 2.0 * x[1] + 0.5 * x[2] + 4.0
      obj(bits) = p(x => bits)
      exact, exact_config = brute_force(obj, 3)

      E, sol = TenSolver.minimize(p; backend=:gtn)
      @test E ≈ exact
      @test TenSolver.sample(sol) == exact_config

      repeated = x[1]^2 + x[1] - 2.0 * x[2]
      repeated_energy, repeated_sol = TenSolver.minimize(repeated; backend=:gtn)
      @test repeated_energy ≈ -2.0
      @test TenSolver.sample(repeated_sol) == [0, 1]
    end
  end
end
