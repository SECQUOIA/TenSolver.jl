@testset "QUBO Correctness" begin
  dim = 5

  @testset "Ill-formed input" begin
    # Should throw when the matrix is not square
    Q = randn(dim, dim - 2)
    @test_throws DimensionMismatch minimize(Q)

    Q = randn(dim - 2, dim)
    @test_throws DimensionMismatch minimize(Q)
  end

  @testset "Ultra simple sanity checks" begin
    @testset "Zero matrix" begin
      E, psi = minimize([0.0 0; 0 0]; verbosity = 0)

      @test E ≈ 0.0
      for x in [[i, j] for i in 0:1, j in 0:1]
        @test x in psi
      end
    end

    @testset "Zero matrix + constant" begin
      E, psi = minimize([0.0 0; 0 0], 3.0; verbosity = 0)

      @test E ≈ 3.0
      for x in [[i, j] for i in 0:1, j in 0:1]
        @test x in psi
      end
    end

    @testset "Zero matrix + zero linear" begin
      E, psi = minimize([0.0 0; 0 0], [0.0, 0.0]; verbosity = 0)

      @test E ≈ 0.0
      for x in [[i, j] for i in 0:1, j in 0:1]
        @test x in psi
      end
    end

    @testset "Zero matrix + linear" begin
      E, psi = minimize([0.0 0; 0 0], [1.0, -1.0]; verbosity = 0)

      @test E ≈ -1.0
      @test TenSolver.sample(psi) == [0, 1]
    end

    @testset "Zero matrix + linear + const" begin
      E, psi = minimize([0.0 0; 0 0], [1.0, -1.0], 3.0; verbosity = 0)

      @test E ≈ 2.0
      @test TenSolver.sample(psi) == [0, 1]
    end

    @testset "Single variable" begin
      Q = reshape([-2.0], 1, 1)
      E, psi = minimize(Q; verbosity = 0)

      @test E ≈ -2.0
      @test TenSolver.sample(psi) == [1]
      @test psi.stats.energies    == [-2.0]
      @test psi.stats.bond_dims   == [1]
      @test length(psi.stats.elapsed_times) == 1
    end

    @testset "Single variable with linear and constant terms" begin
      Q = reshape([0.0], 1, 1)
      E, psi = minimize(Q, [-3.0], 2.0; verbosity = 0)

      @test E ≈ -1.0
      @test TenSolver.sample(psi) == [1]
    end

    @testset "Single variable degeneracy" begin
      Q = reshape([0.0], 1, 1)
      E, psi = minimize(Q; verbosity = 0)

      @test E ≈ 0.0
      @test [0] in psi
      @test [1] in psi
    end

    @testset "Max: Single variable" begin
      Q = reshape([2.0], 1, 1)
      E, psi = maximize(Q; verbosity = 0)

      @test E ≈ 2.0
      @test TenSolver.sample(psi) == [1]
    end

    @testset "Identity" begin
      E, psi = minimize([1.0 0.0; 0.0 1.0]; verbosity = 0)

      @test E ≈ 0.0
      @test TenSolver.sample(psi) == [0, 0]
    end

    @testset "One variable" begin
      E, psi = minimize(reshape([2.0], 1, 1), [-3.0]; verbosity=0)

      @test E ≈ -1.0
      @test TenSolver.sample(psi) == [1]
      @test [1] in psi
    end

    @testset "Max: Identity" begin
      E, psi = maximize([1.0 0.0; 0.0 1.0]; verbosity = 0)

      @test E ≈ 2.0
      @test TenSolver.sample(psi) == [1, 1]
    end

    @testset "Max: Identity + const" begin
      E, psi = maximize([1.0 0.0; 0.0 1.0], 3.0; verbosity = 0)

      @test E ≈ 5.0
      @test TenSolver.sample(psi) == [1, 1]
    end

    @testset "Max: Zero matrix + linear + const" begin
      E, psi = maximize([0.0 0.0; 0.0 0.0], [1.0, -1.0], 3.0; verbosity = 0)

      @test E ≈ 4.0
      @test TenSolver.sample(psi) == [1, 0]
    end
  end

  @testset "Iteration stats tracking" begin
    @testset "Solution carries stats" begin
      E, psi = minimize(
        [1.0 0; 0 -1.0];
        iterations=5,
        check_variance_every_iteration=2,
        vtol=-Inf,
        verbosity=0,
      )
      @test psi isa TenSolver.Solution
      @test length(psi.stats.energies)      == 5
      @test length(psi.stats.bond_dims)     == 5
      @test length(psi.stats.elapsed_times) == 5
      @test length(psi.stats.variances)     == 5
      @test all(isfinite, psi.stats.energies)
      @test issorted(psi.stats.elapsed_times)
      @test all(>(0), psi.stats.bond_dims)
      @test map(isnothing, psi.stats.variances) == [true, false, true, false, true]
      @test all(isfinite, filter(!isnothing, psi.stats.variances))
      @test isfinite(last(psi.stats.energies))
      @test last(psi.stats.energies) ≈ E

      @test isempty(psi.stats.max_bonds.projections)
      @test psi.stats.max_bonds.objective == psi.stats.max_bonds.hamiltonian
    end

    @testset "on_iteration callback preserves its documented signature" begin
      calls = NamedTuple[]
      function cb(psi; iteration, objective, bond_dim, elapsed_time)
        push!(calls, (; iteration, objective, bond_dim, elapsed_time))
      end
      minimize([1.0 0; 0 -1.0]; iterations=5, on_iteration=cb, verbosity = 0)
      @test getproperty.(calls, :iteration) == collect(1:5)
      @test all(call -> isfinite(call.objective), calls)
      @test all(call -> call.bond_dim > 0, calls)
      @test issorted(getproperty.(calls, :elapsed_time))
    end

    @testset "callback_every" begin
      calls = Int[]
      cb = (psi; iteration, kw...) -> push!(calls, iteration)
      minimize([1.0 0; 0 -1.0]; iterations=9, on_iteration=cb, callback_every=3, verbosity = 0)
      @test calls == [3, 6, 9]
    end

    @testset "iteration cadence < 1 throws" begin
      @test_throws ArgumentError minimize([1.0 0; 0 -1.0]; verbosity = 0, iterations=1, callback_every=0)
      @test_throws ArgumentError minimize([1.0 0; 0 -1.0]; verbosity = 0, iterations=1, callback_every=-1)
      @test_throws ArgumentError minimize([1.0 0; 0 -1.0]; verbosity = 0, iterations=1, check_variance_every_iteration=0)
      @test_throws ArgumentError minimize([1.0 0; 0 -1.0]; verbosity = 0, iterations=1, check_variance_every_iteration=-1)
    end

    @testset "callback receives a fresh MPS each iteration" begin
      # DMRG allocates a new MPS object per iteration rather than mutating
      # in-place, so each callback invocation receives a distinct object.
      ids = UInt[]
      cb = (psi; kw...) -> push!(ids, objectid(psi))
      minimize([1.0 0; 0 -1.0]; iterations=3, on_iteration=cb, verbosity = 0)
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
      original_bandwidth = bandwidth(Q)
      permuted_bandwidth = bandwidth(Q[permutation, permutation])

      @test sort(permutation) == collect(1:5)
      @test original_bandwidth == 3
      @test permuted_bandwidth == 1
    end

    @testset "Permutation cutoff" begin
      Q = zeros(3, 3)
      Q[3, 1] = 1e-9

      @test TenSolver.qmatrix_permutation(Q; cutoff=0) == [3, 1, 2]
      @test TenSolver.qmatrix_permutation(Q; cutoff=1e-8) == [1, 2, 3]
    end

    @testset "Preprocessing preserves solution and original variable order" begin
      Q = [0.0 0.0 -2.0;
           0.0 0.0  0.0;
          -2.0 0.0  0.0]
      l = [0.5, 1.0, 0.5]

      Random.seed!(1)
      E0, psi0 = minimize(Q, l; preprocess=false, iterations=3, verbosity=0)
      x0 = TenSolver.sample(psi0)

      Random.seed!(1)
      E, psi = minimize(Q, l; preprocess=true, iterations=3, verbosity=0)
      x = TenSolver.sample(psi)

      @test E0 ≈ -3.0
      @test E ≈ -3.0
      @test x0 == [1, 0, 1]
      @test x == [1, 0, 1]
      @test [1, 0, 1] in psi
    end
  end

  @testset "Pure quadratic" begin
    Q = randn(dim, dim)

    # TenSolver solution
    e, psi = TenSolver.minimize(Q; verbosity = 0)
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

    e0, x0 = brute_force(x -> dot(x, Q, x), dim)
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
    e, psi = TenSolver.minimize(Q, l, c; verbosity = 0)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test obj(x) ≈ e

    for i in 1:10
      y = rand(Bool, dim)
      @test obj(y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #
    e0, x0 = brute_force(obj, dim)
    # Same minimum value
    @test e ≈ e0
    # Solution is sampleable
    @test x0 in psi
  end
end
