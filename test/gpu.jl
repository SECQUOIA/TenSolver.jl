import CUDA
import DynamicPolynomials as DP

@testset "CUDA explicit domains" begin
  if CUDA.functional()
    Q = [2.0 0.25; 0.25 -1.0]
    l = [-3.0, 2.0]
    c = 4.0
    DP.@polyvar x[1:2]
    p = 2x[1]^3 - 3x[1] - x[2] + c

    # Independent enumeration of the four binary/spin assignments gives the
    # matrix extrema. For p, 2x1^3 - 3x1 is (-10, 0, 45) at (-2, 0, 3).
    # The heterogeneous linear objective chooses each variable independently.
    fixtures = (
      ((Q, l, c), 0:1, 3.0, [1, 0], 5.0, [0, 1]),
      ((Q, l, c), [-1, 1], -0.5, [1, -1], 9.5, [-1, 1]),
      ((p,), [-2, 0, 3], -9.0, [-2, 3], 51.0, [3, -2]),
      (([2.0, -3.0], c), [[-1, 2], [0.0, 0.5, 1.5]], -2.5, [-1, 1.5], 8.0, [2, 0]),
    )
    for (args, domain, emin, xmin, emax, xmax) in fixtures
      for (solve, expected, assignment) in ((minimize, emin, xmin), (maximize, emax, xmax))
        E, psi = solve(args...; domain, device = CUDA.cu, verbosity = 0, noise = 0.0)
        @test E ≈ expected
        @test all(t -> ITensors.data(t) isa CUDA.CuArray, psi.tensor)
        @test all(==(assignment), sample(psi, 5))
      end
    end

    @test_throws UndefKeywordError minimize(Q; device = CUDA.cu, verbosity = 0)
    @test_throws UndefKeywordError maximize(Q; device = CUDA.cu, verbosity = 0)
  else
    @info "Skipping CUDA smoke test: no functional NVIDIA GPU is available"
    @test_skip CUDA.functional()
  end
end
