import ITensorMPS: maxlinkdim

# In theory (with exact real numbers),
# we are guaranteed a maximum bond dimension of N+1.
# Experimentally, it seems that we could tighten those bounds
# to max_bd <= min(rank(Q) + 2, bandwidth + 1)

@testset "MPO Construction" begin
  dim = 5

  @testset "Bond dimension UB" begin
    @testset "Tridiagonal" begin
      for domain in [[0, 1], [-1, 1], [-2, 3, 5]]
        Q = randn(dim, dim)
        H = TenSolver.tensorize(Tridiagonal(Q); domain)

        @test maxlinkdim(H) <= 3
      end
    end

    @testset "Full rank" begin
      for domain in [[0, 1], [-1, 1], [-2, 3, 5]]
        Q = randn(dim, dim)
        H = TenSolver.tensorize(Q; domain)

        @test maxlinkdim(H) <= ceil(bandwidth(Q)/2) + 2
      end
    end
  end
end
