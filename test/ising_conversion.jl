using SparseArrays

@testset "Ising conversion" begin
  bitstrings(n) = [collect(bits) for bits in Iterators.product(ntuple(_ -> 0:1, n)...)]

  qubo_value(Q, l, c, x) = dot(x, Q, x) + dot(l, x) + c

  function argmin_set(xs, values; atol=0)
    best = minimum(values)
    return Set(
      Tuple(xs[i])
      for i in eachindex(xs)
      if iszero(atol) ? values[i] == best : isapprox(values[i], best; atol, rtol=1e-8)
    )
  end

  function assert_conversion(Q, l=zeros(eltype(Q), size(Q, 1)), c=zero(eltype(Q)); exact=true)
    model = TenSolver.qubo_to_ising(Q, l, c)
    qubo = TenSolver.ising_to_qubo(model)
    xs = bitstrings(size(Q, 1))

    rows, cols, _ = findnz(model.J)
    @test all(rows[i] < cols[i] for i in eachindex(rows))
    @test qubo.Q isa SparseMatrixCSC

    bool_values = map(x -> qubo_value(Q, l, c, x), xs)
    ising_values = map(x -> TenSolver.ising_energy(model, TenSolver.bool_to_spin(x)), xs)

    for (x, bool_energy, ising_energy) in zip(xs, bool_values, ising_values)
      s = TenSolver.bool_to_spin(x)
      @test TenSolver.spin_to_bool(s) == x
      @test TenSolver.bool_to_spin(TenSolver.spin_to_bool(s)) == s

      qubo_roundtrip_energy = qubo_value(qubo.Q, qubo.l, qubo.c, x)
      if exact
        @test bool_energy == ising_energy
        @test bool_energy == qubo_roundtrip_energy
      else
        @test isapprox(bool_energy, ising_energy; atol=1e-10, rtol=1e-8)
        @test isapprox(bool_energy, qubo_roundtrip_energy; atol=1e-10, rtol=1e-8)
      end
    end

    atol = exact ? 0 : 1e-8
    @test argmin_set(xs, bool_values; atol) == argmin_set(xs, ising_values; atol)
  end

  rational_matrix(n) = [((2 * i - 3 * j) // (i + j + 1)) for i in 1:n, j in 1:n]

  @testset "Dense random pure quadratic" begin
    rng = MersenneTwister(36)
    for n in 1:6
      Q = randn(rng, n, n)
      assert_conversion(Q, zeros(n), 0.0; exact=false)
    end
  end

  @testset "Dense non-symmetric rational QUBO" begin
    for n in 1:6
      Q = rational_matrix(n)
      l = [((-1)^i * i) // (n + 1) for i in 1:n]
      assert_conversion(Q, l, 3 // 7)
    end
  end

  @testset "Dense symmetric rational QUBO" begin
    for n in 1:6
      A = rational_matrix(n)
      Q = (A + transpose(A)) ./ 2
      l = [i // (2n + 1) for i in 1:n]
      assert_conversion(Q, l, -5 // 11)
    end
  end

  @testset "Diagonal-only QUBO" begin
    for n in 1:6
      Q = diagm(0 => [i // 3 for i in 1:n])
      l = [(-i) // 5 for i in 1:n]
      assert_conversion(Q, l, 2 // 9)
    end
  end

  @testset "Sparse QUBO" begin
    for n in 2:6
      Q = sparse([1, n, 2], [n, 1, 2], [3 // 5, -1 // 4, 2 // 3], n, n)
      l = sparsevec([1, n], [1 // 6, -2 // 7], n)
      assert_conversion(Q, l, -1 // 8)
    end
  end

  @testset "Zero QUBO" begin
    for n in 1:6
      assert_conversion(zeros(Rational{Int}, n, n), zeros(Rational{Int}, n), 0 // 1)
    end
  end

  @testset "One-variable edge case" begin
    Q = reshape([3 // 2], 1, 1)
    l = [-5 // 4]
    assert_conversion(Q, l, 7 // 3)
  end

  @testset "Input validation" begin
    @test_throws DimensionMismatch TenSolver.qubo_to_ising(ones(2, 3))
    @test_throws DimensionMismatch TenSolver.qubo_to_ising(ones(2, 2), ones(3))
    @test_throws DimensionMismatch TenSolver.ising_to_qubo(ones(2, 3), ones(2))
    @test_throws DimensionMismatch TenSolver.ising_to_qubo(ones(2, 2), ones(3))
    @test_throws ArgumentError TenSolver.qubo_to_ising(reshape([1.0], 1, 1); convention=:binary)
    @test_throws ArgumentError TenSolver.bool_to_spin([0, 2])
    @test_throws ArgumentError TenSolver.spin_to_bool([-1, 0])
  end
end
