import DynamicPolynomials
import GenericTensorNetworks
import ProblemReductions

@testset "GenericTensorNetworks backend" begin
  @testset "QUBO optimum, count, and enumeration" begin
    Q = [0.0 2.0; 0.0 0.0]
    l = [-1.0, -1.0]
    c = 3.0
    optimum_configs = [[1, 0], [0, 1]]

    E, sol = TenSolver.minimize(Q, l, c; backend = :gtn)
    @test sol isa TenSolver.GTNSolution
    @test E ≈ 2.0
    @test TenSolver.sample(sol) in optimum_configs

    Q_active = [0.0 -5.0; 0.0 0.0]
    l_active = [2.0, 2.0]
    E_active, active_sol = TenSolver.minimize(Q_active, l_active; backend = :gtn)
    @test E_active ≈ -1.0
    @test TenSolver.sample(active_sol) == [1, 1]

    Emax, max_sol = TenSolver.maximize(Q, l, c; backend = :gtn)
    @test Emax ≈ 3.0
    @test TenSolver.sample(max_sol) in [[0, 0], [1, 1]]

    E_size, size_sol = TenSolver.minimize(Q, l, c; backend = :gtn, property = :size)
    @test E_size ≈ 2.0
    @test size_sol.metadata["size"] ≈ -1.0
    @test_throws ArgumentError TenSolver.sample(size_sol)

    E_count, count_sol = TenSolver.minimize(Q, l, c; backend = :gtn, property = :count)
    @test E_count ≈ 2.0
    @test count_sol.metadata["count"] ≈ 2

    E_configs, config_sol =
      TenSolver.minimize(Q, l, c; backend = :gtn, property = :configs)
    @test E_configs ≈ 2.0
    @test sort(config_sol.configs) == sort(optimum_configs)

    E_kbest, kbest_sol =
      TenSolver.minimize(Q, l, c; backend = :gtn, property = :kbest_sizes, k = 2)
    @test E_kbest ≈ 2.0
    @test sort(kbest_sol.metadata["size"]) ≈ [-1.0, -1.0]

    E_single_k, single_k_sol =
      TenSolver.minimize(Q, l, c; backend = :gtn, property = :single, k = 2)
    @test E_single_k ≈ 2.0
    @test sort(single_k_sol.configs) == sort(optimum_configs)

    @test_throws ArgumentError TenSolver.minimize(
      Q,
      l,
      c;
      backend = :gtn,
      property = :configs,
      k = 2,
    )
    @test_throws ArgumentError TenSolver.minimize(
      Q,
      l,
      c;
      backend = :gtn,
      property = :count,
      k = 2,
    )

    @test_throws ArgumentError TenSolver.minimize(
      Q,
      l,
      c;
      backend = :gtn,
      domain = [-1, 1],
    )
    @test_throws ArgumentError TenSolver.minimize(
      Q,
      l,
      c;
      backend = :gtn,
      constraints = AbstractConstraint[AssignmentConstraint([1, 2], [1], :(==), 1),],
    )
    @test_throws ArgumentError TenSolver.minimize(Q, l, c; backend = :gtn, verbosity = 0)
  end

  @testset "Normalized Boolean domains" begin
    Q = [-2.0 0.0; 0.0 1.0]
    l = [1.0, -4.0]
    DynamicPolynomials.@polyvar x[1:2]
    p = -2.0 * x[1]^2 + x[1] + x[2]^2 - 4.0 * x[2]
    # Both objectives have the unique Boolean optimum -4 at [1, 1],
    # with both diagonal terms active before domain reduction.
    for domain in (0:1, [1, 0, 1], [[1, 0, 1], [0, 1]])
      for objective in ((Q, l), (p,))
        energy, solution = TenSolver.minimize(objective...; backend = :gtn, domain)
        @test energy ≈ -4.0
        @test TenSolver.sample(solution) == [1, 1]
      end
    end

    for domain in ([[0, 1], [-1, 1]], [[0], [0, 1]])
      for objective in ((Q, l), (p,))
        @test_throws ArgumentError TenSolver.minimize(
          objective...;
          backend = :gtn,
          domain,
        )
      end
    end
  end

  @testset "Higher-order polynomial objective" begin
    DynamicPolynomials.@polyvar x[1:3]
    p = 1.0 * x[1] * x[2] * x[3] - 2.0 * x[1] + 0.5 * x[2] + 4.0
    obj(bits) = p(x => bits)
    exact, exact_config = brute_force(obj, 3)

    E, sol = TenSolver.minimize(p; backend = :gtn)
    @test E ≈ exact
    @test TenSolver.sample(sol) == exact_config

    repeated = x[1]^2 + x[1] - 2.0 * x[2]
    repeated_energy, repeated_sol = TenSolver.minimize(repeated; backend = :gtn)
    @test repeated_energy ≈ -2.0
    @test TenSolver.sample(repeated_sol) == [0, 1]
  end

  @testset "Polynomial reduction removes variables" begin
    DynamicPolynomials.@polyvar x[1:2]
    p = x[1]^2 - x[1] - 2.0 * x[2]
    # The two optima [0, 1] and [1, 1] must not collapse to a single one-bit state.
    for property in (:single, :count, :configs)
      @test_throws ArgumentError TenSolver.minimize(p; backend = :gtn, domain = 0:1, property)
    end
    @test_throws ArgumentError TenSolver.minimize(x[1]^2 - x[1]; backend = :gtn, domain = 0:1)
  end
end
