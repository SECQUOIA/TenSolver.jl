@testset "QUBO Correctness" begin
  dim = 5

  @testset "Ultra simple sanity checks" begin
    @testset "Zero matrix" begin
      E, psi = minimize([0.0 0; 0 0])

      @test E ≈ 0.0
      for x in [[i, j] for i in 0:1, j in 0:1]
        @test x in psi
      end
    end

    @testset "Zero matrix + constant" begin
      E, psi = minimize([0.0 0; 0 0], 3.0)

      @test E ≈ 3.0
      for x in [[i, j] for i in 0:1, j in 0:1]
        @test x in psi
      end
    end

    @testset "Zero matrix + zero linear" begin
      E, psi = minimize([0.0 0; 0 0], [0.0, 0.0])

      @test E ≈ 0.0
      for x in [[i, j] for i in 0:1, j in 0:1]
        @test x in psi
      end
    end

    @testset "Zero matrix + linear" begin
      E, psi = minimize([0.0 0; 0 0], [1.0, -1.0])

      @test E ≈ -1.0
      @test TenSolver.sample(psi) == [0, 1]
    end

    @testset "Zero matrix + linear + const" begin
      E, psi = minimize([0.0 0; 0 0], [1.0, -1.0], 3.0)

      @test E ≈ 2.0
      @test TenSolver.sample(psi) == [0, 1]
    end

    @testset "Identity" begin
      E, psi = minimize([1.0 0.0; 0.0 1.0])

      @test E ≈ 0.0
      @test TenSolver.sample(psi) == [0, 0]
    end

    @testset "Max: Identity" begin
      E, psi = maximize([1.0 0.0; 0.0 1.0])

      @test E ≈ 2.0
      @test TenSolver.sample(psi) == [1, 1]
    end

    @testset "Max: Identity + const" begin
      E, psi = maximize([1.0 0.0; 0.0 1.0], 3.0)

      @test E ≈ 5.0
      @test TenSolver.sample(psi) == [1, 1]
    end

    @testset "Max: Zero matrix + linear + const" begin
      E, psi = maximize([0.0 0.0; 0.0 0.0], [1.0, -1.0], 3.0)

      @test E ≈ 4.0
      @test TenSolver.sample(psi) == [1, 0]
    end
  end

  @testset "Iteration stats tracking" begin
    @testset "Distribution carries stats" begin
      E, psi = minimize([1.0 0; 0 -1.0]; iterations=5)
      @test length(psi.energies)      == 5
      @test length(psi.bond_dims)     == 5
      @test length(psi.elapsed_times) == 5
      @test all(isfinite, psi.energies)
      @test all(>=(0), psi.elapsed_times)
    end

    @testset "on_iteration callback is called" begin
      calls = Int[]
      cb = (psi; iteration, kw...) -> begin
        push!(calls, iteration)
        # exercises the convenience constructor Distribution(::MPS)
        dist = TenSolver.Distribution(psi)
        @test dist isa TenSolver.Distribution
      end
      minimize([1.0 0; 0 -1.0]; iterations=5, on_iteration=cb)
      @test calls == collect(1:5)
    end

    @testset "callback_every" begin
      calls = Int[]
      cb = (psi; iteration, kw...) -> push!(calls, iteration)
      minimize([1.0 0; 0 -1.0]; iterations=9, on_iteration=cb, callback_every=3)
      @test calls == [3, 6, 9]
    end

    @testset "callback_every < 1 throws" begin
      @test_throws ArgumentError minimize([1.0 0; 0 -1.0]; iterations=1, callback_every=0)
      @test_throws ArgumentError minimize([1.0 0; 0 -1.0]; iterations=1, callback_every=-1)
    end

    @testset "callback receives live MPS (no copy)" begin
      collected = []
      cb = (psi; kw...) -> push!(collected, copy(psi))
      minimize([1.0 0; 0 -1.0]; iterations=3, on_iteration=cb)
      @test length(collected) == 3
      for i in 1:length(collected), j in (i+1):length(collected)
        @test collected[i] !== collected[j]

        # Check for shallow copy
        for k in 1:length(collected[i])
          @test collected[i][k] !== collected[j][k]
        end
      end
    end
  end

  @testset "Pure quadratic" begin
    Q = randn(dim, dim)

    # TenSolver solution
    e, psi = TenSolver.minimize(Q)
    x = TenSolver.sample(psi)

    # Is the sampled solution part of the ground state?
    @test x in psi

    # Does the ground energy match the solution?
    @test dot(x, Q, x) ≈ e

    for i in 1:10
      y = rand(Bool, dim)
      @test dot(y, Q, y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #

    e0, x0 = brute_force(x -> dot(x, Q, x), Float64, dim)
    # Same minimum value
    @test e ≈ e0
    # Same solution
    @test x == x0
    # Ground state
    @test x0 in psi
  end

  @testset "Quad+Lin" begin
    Q = 2*randn(dim, dim)
    l = 2*randn(dim)
    c = randn()

    # Objective function
    obj(x) = dot(x, Q, x) + dot(l, x) + c

    # TenSolver solution
    e, psi = TenSolver.minimize(Q, l, c)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test obj(x) ≈ e

    for i in 1:10
      y = rand(Bool, dim)
      @test obj(y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #
    e0, x0 = brute_force(obj, Float64, dim)
    # Same minimum value
    @test e ≈ e0
    # Solution is sampleable
    @test x0 in psi
  end
end
