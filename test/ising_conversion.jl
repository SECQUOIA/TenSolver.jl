using SparseArrays

@testset "Ising conversion" begin
  QT = TenSolver.QUBOTools

  bitstrings(n) = [collect(bits) for bits in Iterators.product(ntuple(_ -> 0:1, n)...)]

  qubo_value(Q, l, c, x) = dot(x, Q, x) + dot(l, x) + c
  spin_value(J, h, offset, s) = dot(s, J, s) + dot(h, s) + offset

  function form_parts(form)
    n, l, Q, scale, offset, sense, domain = form
    @test n == length(l) == size(Q, 1) == size(Q, 2)
    @test scale == one(scale)
    @test sense === QT.Min
    return (; n, l, Q, scale, offset, sense, domain)
  end

  function assert_upper_triangular(Q)
    rows, cols, _ = findnz(Q)
    @test all(rows[i] < cols[i] for i in eachindex(rows))
  end

  function argmin_set(xs, values; atol=0)
    best = minimum(values)
    return Set(
      Tuple(xs[i])
      for i in eachindex(xs)
      if iszero(atol) ? values[i] == best : isapprox(values[i], best; atol, rtol=1e-8)
    )
  end

  function assert_conversion(Q, l=zeros(eltype(Q), size(Q, 1)), c=zero(eltype(Q)); exact=true)
    ising = TenSolver.qubo_to_ising(Q, l, c)
    qubo = TenSolver.ising_to_qubo(ising)
    xs = bitstrings(size(Q, 1))

    ising_parts = form_parts(ising)
    qubo_parts = form_parts(qubo)
    @test ising isa QT.AbstractForm
    @test qubo isa QT.AbstractForm
    @test ising_parts.domain === QT.SpinDomain
    @test qubo_parts.domain === QT.BoolDomain
    @test qubo_parts.Q isa SparseMatrixCSC
    assert_upper_triangular(ising_parts.Q)
    assert_upper_triangular(qubo_parts.Q)

    bool_values = map(x -> qubo_value(Q, l, c, x), xs)
    ising_values = map(x -> QT.value(TenSolver.bool_to_spin(x), ising), xs)

    for (x, bool_energy, ising_energy) in zip(xs, bool_values, ising_values)
      s = TenSolver.bool_to_spin(x)
      @test TenSolver.spin_to_bool(s) == x
      @test TenSolver.bool_to_spin(TenSolver.spin_to_bool(s)) == s

      qubo_roundtrip_energy = QT.value(x, qubo)
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

    Q = [
      0//1   -3//4   2//5
      1//6    0//1  -5//7
      0//1    3//8   1//9
    ]
    l = [2//3, -1//5, 4//7]
    assert_conversion(Q, l, -11//13)
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

  @testset "QUBOTools normalizes Ising inputs" begin
    function assert_ising_roundtrip(J, h=zeros(eltype(J), size(J, 1)), offset=zero(eltype(J)))
      qubo = TenSolver.ising_to_qubo(J, h, offset)
      qubo_parts = form_parts(qubo)
      xs = bitstrings(size(J, 1))

      @test qubo_parts.domain === QT.BoolDomain
      assert_upper_triangular(qubo_parts.Q)

      for x in xs
        spin = TenSolver.bool_to_spin(x)
        @test spin_value(J, h, offset, spin) == QT.value(x, qubo)
      end
    end

    assert_ising_roundtrip(
      [0 0; 2 0],
      zeros(Int, 2),
      0,
    )
    assert_ising_roundtrip(
      [0 2; 2 0],
      [1, -1],
      3,
    )
    assert_ising_roundtrip(
      [5 3; -3 7],
      zeros(Int, 2),
      0,
    )

    qubo = TenSolver.ising_to_qubo([5 0; 0 7], zeros(Int, 2), 0)
    @test QT.value([0, 0], qubo) == 12
  end

  @testset "Input validation" begin
    @test_throws DimensionMismatch TenSolver.qubo_to_ising(ones(2, 3))
    @test_throws DimensionMismatch TenSolver.qubo_to_ising(ones(2, 2), ones(3))
    @test_throws DimensionMismatch TenSolver.ising_to_qubo(ones(2, 3), ones(2))
    @test_throws DimensionMismatch TenSolver.ising_to_qubo(ones(2, 2), ones(3))
    @test_throws ArgumentError TenSolver.qubo_to_ising(reshape([1.0], 1, 1); convention=:binary)
    @test_throws ArgumentError TenSolver.bool_to_spin([0, 2])
    @test_throws ArgumentError TenSolver.spin_to_bool([-1, 0])

    spin_form = TenSolver.qubo_to_ising(reshape([1.0], 1, 1))
    bool_form = TenSolver.ising_to_qubo(spin_form)
    @test_throws ArgumentError TenSolver.qubo_to_ising(spin_form)
    @test_throws ArgumentError TenSolver.ising_to_qubo(bool_form)
  end
end
