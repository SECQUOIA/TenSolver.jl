@testset "QUBO Correctness" begin
  dim = 5

  qubo_bandwidth(Q) = begin
    bw = 0
    for i in axes(Q, 1), j in (i + 1):last(axes(Q, 2))
      if abs(Q[i, j] + Q[j, i]) > 0
        bw = max(bw, j - i)
      end
    end
    bw
  end

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

  @testset "Q-matrix preprocessing" begin
    @testset "Permutation reduces coupling bandwidth" begin
      path = zeros(5, 5)
      for i in 1:4
        path[i, i + 1] = 1.0
        path[i + 1, i] = 1.0
      end

      scramble = [1, 3, 5, 2, 4]
      Q = path[scramble, scramble]
      permutation = TenSolver.qmatrix_permutation(Q)

      @test sort(permutation) == collect(1:5)
      @test qubo_bandwidth(Q[permutation, permutation]) < qubo_bandwidth(Q)
    end

    @testset "Samples are returned in original variable order" begin
      Q = [0.0 0.0 -2.0;
           0.0 0.0  0.0;
          -2.0 0.0  0.0]
      l = [0.5, 1.0, 0.5]

      E, psi = minimize(Q, l; preprocess=true, iterations=3, verbosity=0)

      @test E ≈ -3.0
      @test TenSolver.sample(psi) == [1, 0, 1]
      @test [1, 0, 1] in psi
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
