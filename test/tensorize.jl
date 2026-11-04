import ITensorMPS: maxlinkdim
import SparseArrays: sparse

function quadratic_mpo_matrix(H)
  sites = ITensorMPS.siteinds(first, H; plev = 0)
  dimension = prod(ITensors.dim, sites)
  return reshape(Array(prod(H), ITensors.prime.(sites)..., sites...), dimension, dimension)
end

function assert_quadratic_mpo(Q, l, domain; atol = 1e-10)
  H = TenSolver.tensorize(Q, l; domain)
  expected = [dot(x, Q, x) + dot(l, x) for x in map(collect, Iterators.product(domain...))]
  if all(iszero, expected)
    # Empty-storage zero MPOs need not materialize a dense array.
    @test iszero(norm(H))
  else
    @test quadratic_mpo_matrix(H) ≈ Diagonal(vec(expected)) atol = atol rtol =
      20eps(eltype(Q))
  end
  @test all(t -> eltype(t) == eltype(Q), H)
  return H
end

# In theory (with exact real numbers),
# we are guaranteed a maximum bond dimension of N+1.
# Experimentally, it seems that we could tighten those bounds
# to max_bd <= min(rank(Q) + 2, bandwidth + 1)

@testset "MPO Construction" begin
  dim = 5

  @testset "Bond dimension UB" begin
    @testset "Tridiagonal" begin
      for domain in TenSolver.Domains{Float64}.([[0, 1], [-1, 1], [-2, 3, 5]], dim)
        Q = randn(dim, dim)
        H = TenSolver.tensorize(Tridiagonal(Q); domain)

        @test maxlinkdim(H) <= 3
      end
    end

    @testset "Full rank" begin
      for domain in TenSolver.Domains{Float64}.([[0, 1], [-1, 1], [-2, 3, 5]], dim)
        Q = randn(dim, dim)
        H = TenSolver.tensorize(Q; domain)

        @test maxlinkdim(H) <= ceil(bandwidth(Q)/2) + 2
      end
    end
  end
end

@testset "Quadratic MPO coefficients and domains" begin
  Q = [2.0 3.0 0.0 -2.0; -1.0 -4.0 1.0 0.0; 0.0 2.0 5.0 0.0; 4.0 0.0 0.0 -1.0]
  l = [-1.0, 0.5, 2.0, 0.0]
  for T in (Float32, Float64)
    atol = T == Float32 ? 1e-4 : 1e-10
    for ds in (
      fill([0, 1], 4),
      fill([-1, 1], 4),
      fill([-2, 0, 3], 4),
      [[-2, 0, 3], [-1, 1], [0, 2], [1]],
      [[-0.5, 0.25, 7], [-1, 1], [0, 2], [1]],
    )
      domain = TenSolver.Domains{T}(ds, 4)
      for A in (Q, sparse(Q), Tridiagonal(Q), Diagonal(diag(Q)), zeros(4, 4))
        assert_quadratic_mpo(T.(A), T.(l), domain; atol)
      end
    end
  end

  @testset "Single site, zero and disconnected interactions" begin
    for T in (Float32, Float64)
      domain = TenSolver.Domains{T}([-2, 1, 3], 1)
      assert_quadratic_mpo(reshape(T[2], 1, 1), T[-3], domain)
      assert_quadratic_mpo(zeros(T, 1, 1), zeros(T, 1), domain)
      domain = TenSolver.Domains{T}(0:1, 6)
      A = zeros(T, 6, 6)
      A[1, 2] = 2
      A[4, 6] = -3
      assert_quadratic_mpo(A, zeros(T, 6), domain; atol = T == Float32 ? 1e-5 : 1e-10)
      H = assert_quadratic_mpo(zeros(T, 6, 6), T.(1:6), domain)
      @test maxlinkdim(H) <= 2
      Hzero = assert_quadratic_mpo(zeros(T, 6, 6), zeros(T, 6), domain)
      @test maxlinkdim(Hzero) == 1
    end
  end

  @testset "Coefficient filtering precedes MPO compression" begin
    # Each orientation is below 0.5, but their sum is retained. The other
    # pair cancels before filtering; onsite terms exactly at 0.5 are removed.
    A = [0.5 0.4 2.0; 0.4 2.0 0.0; -2.0 0.0 -0.5]
    b = [0.5, 1.0, -0.5]
    domain = TenSolver.Domains{Float64}([-1, 0, 2], 3)
    H = TenSolver.tensorize(A, b; domain, cutoff = 0.5)
    expected = [0.8x[1] * x[2] + 2x[2]^2 + x[2] for x in Iterators.product(domain...)]
    @test quadratic_mpo_matrix(H) ≈ Diagonal(vec(expected)) atol = 1e-12

    # The generic constructor's relative discarded-weight cutoff is 1e-15.
    # This small second singular direction is therefore omitted at cut 2.
    A = zeros(4, 4)
    A[1, 3] = 1
    A[2, 4] = 1e-9
    domain = TenSolver.Domains{Float64}(0:1, 4)
    H = TenSolver.tensorize(A; domain)
    expected = [x[1] * x[3] for x in Iterators.product(domain...)]
    @test quadratic_mpo_matrix(H) ≈ Diagonal(vec(expected)) atol = 1e-12
    generic = invoke(TenSolver.tensorize, Tuple{AbstractArray{Float64}}, A; domain)
    @test quadratic_mpo_matrix(H) ≈ quadratic_mpo_matrix(generic) atol = 1e-12
    @test maxlinkdim(H) == maxlinkdim(generic)

    tiny = eps(Float64) / 2
    H = TenSolver.tensorize(fill(tiny, 4, 4), fill(tiny, 4); domain)
    # A pair combines to eps(), which is also skipped during OpSum assembly.
    @test iszero(norm(H))
    generic = invoke(
      TenSolver.tensorize,
      Tuple{AbstractArray{Float64},AbstractArray{Float64}},
      fill(tiny, 4, 4),
      fill(tiny, 4);
      domain,
    )
    @test iszero(norm(generic))

    # A linear objective must not evaluate an absent quadratic term: x^2
    # overflows here although the retained linear energies remain finite.
    large_domain = TenSolver.Domains{Float64}([0.0, 1.0, 1e200], 1)
    H = TenSolver.tensorize(zeros(1, 1), [2.0]; domain = large_domain)
    @test quadratic_mpo_matrix(H) == Diagonal([0.0, 2.0, 2e200])
  end

  @testset "Generic OpSum agreement and shape checks" begin
    domain = TenSolver.Domains{Float64}([-2, 0, 3], 4)
    H = TenSolver.tensorize(Q, l; domain)
    generic = invoke(
      TenSolver.tensorize,
      Tuple{AbstractArray{Float64},AbstractArray{Float64}},
      Q,
      l;
      domain,
    )
    @test quadratic_mpo_matrix(H) ≈ quadratic_mpo_matrix(generic) atol = 1e-10
    @test maxlinkdim(H) == maxlinkdim(generic)
    @test_throws DimensionMismatch TenSolver.tensorize(zeros(3, 4), l; domain)
    @test_throws DimensionMismatch TenSolver.tensorize(Q, zeros(3); domain)
    @test_throws DimensionMismatch TenSolver.tensorize(
      Q,
      l;
      domain = TenSolver.Domains{Float64}(0:1, 3),
    )
  end
end
