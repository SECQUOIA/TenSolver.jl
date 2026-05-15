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

    @testset "Single variable" begin
      E, psi = minimize(reshape([1.0], 1, 1); verbosity=0)

      @test E ≈ 0.0
      @test TenSolver.sample(psi) == [0]

      E, psi = minimize(reshape([0.0], 1, 1); verbosity=0)

      @test E ≈ 0.0
      @test [0] in psi
      @test [1] in psi
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
    @testset "Solution carries stats" begin
      E, psi = minimize([1.0 0; 0 -1.0]; iterations=5)
      @test psi isa TenSolver.Solution
      @test length(psi.energies)      == 5
      @test length(psi.bond_dims)     == 5
      @test length(psi.elapsed_times) == 5
      @test all(isfinite, psi.energies)
      @test issorted(psi.elapsed_times)
      @test all(>(0), psi.bond_dims)
      @test isfinite(last(psi.energies))
      @test last(psi.energies) ≈ E
    end

    @testset "on_iteration callback is called" begin
      calls = Int[]
      cb = (psi; iteration, kw...) -> push!(calls, iteration)
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

    @testset "callback receives a fresh MPS each iteration" begin
      # DMRG allocates a new MPS object per iteration rather than mutating
      # in-place, so each callback invocation receives a distinct object.
      ids = UInt[]
      cb = (psi; kw...) -> push!(ids, objectid(psi))
      minimize([1.0 0; 0 -1.0]; iterations=3, on_iteration=cb)
      @test length(ids) == 3
      @test length(unique(ids)) == 3
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

  @testset "Issue #19 exact polish escapes local minima" begin
    rows = [
      1, 1, 2, 1, 2, 3, 5, 1, 2, 3, 4, 5, 6, 5, 6, 7, 9, 9, 10, 8, 9, 10, 11,
      9, 10, 11, 12, 1, 2, 4, 7, 1, 2, 3, 1, 2, 3, 4, 14, 4, 5, 6, 7, 14, 8, 9,
      10, 11, 12, 12, 13, 12, 13, 1, 2, 3, 4, 14, 5, 6, 7, 9, 10, 11, 12, 1, 4,
      7, 1, 4, 7, 25, 2, 4, 7, 2, 4, 7, 27, 3, 4, 7, 3, 4, 7, 29, 1, 2, 14, 1,
      2, 14, 31, 7, 14, 9, 13, 10, 13, 11, 13,
    ]
    cols = [
      2, 3, 3, 4, 4, 4, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 10, 11, 11, 12, 12, 12,
      12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 16, 16, 17,
      17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 20, 20, 21, 21, 21, 22, 22,
      23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 28,
      28, 28, 28, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 32, 32, 32, 32, 33,
      33, 34, 34, 35, 35, 36, 36,
    ]
    vals = [
      341.9864, 256.4898, 256.4898, -85.4966, -85.4966, 85.4966, 256.4898, 85.4966, 85.4966,
      85.4966, -85.4966, 256.4898, 256.4898, -85.4966, -85.4966, -85.4966, 256.4898, 256.4898,
      256.4898, -85.4966, 85.4966, 85.4966, 85.4966, -170.9932, -170.9932, -170.9932, 170.9932,
      85.4966, 85.4966, 256.4898, 85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966,
      -85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966,
      -85.4966, -85.4966, -85.4966, 85.4966, -85.4966, -85.4966, 85.4966, 85.4966, 85.4966,
      85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966,
      85.4966, 85.4966, -85.4966, 85.4966, 85.4966, -85.4966, 85.4966, 85.4966, 85.4966, -85.4966,
      85.4966, 85.4966, -85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966, 85.4966,
      85.4966, 85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966, -85.4966, 85.4966,
      -85.4966, -85.4966, 85.4966, -85.4966, 85.4966, -85.4966, 85.4966, -85.4966,
    ]

    Q = zeros(36, 36)
    for (i, j, v) in zip(rows, cols, vals)
      Q[i, j] = v
    end

    l = zeros(36)
    l[
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
       11, 12, 13, 14, 16, 17, 18, 20,
       21, 22, 23, 24, 25, 26, 27, 28,
       29, 30, 31, 32, 33, 34, 35, 36]
    ] = [
      -42.7194, -42.7084, -85.3741, 175.952, 48.9413, 52.4653, -210.389, 91.6596, 86.2356, 86.3366,
      86.7676, 6.99, 172.326, -42.7483, 85.4966, 85.4966, 85.4966, -42.7483,
      -42.7483, -42.7483, -42.7483, -42.7483, -42.7483, -42.7483, -42.7483, -42.7483,
      -128.245, -128.245, 128.245, 128.245, 128.245, 42.7483, 42.7483, 42.7483,
    ]
    c = 641.2245
    x0 = zeros(Int, 36)

    polished = TenSolver._branch_bound_qubo(Q, l, c, x0, dot(x0, Q, x0) + dot(l, x0) + c)

    @test !isnothing(polished)
    E, x = polished
    @test E ≈ 11.7099
    @test dot(x, Q, x) + dot(l, x) + c ≈ E
  end
end
