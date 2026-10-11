include(joinpath(@__DIR__, "../../benchmarks/peps_common.jl"))

@testset "PEPS benchmark smoke" begin
  @test PEPSBenchmarks.has_peps()
  @test isnothing(Base.get_extension(TenSolver, :TenSolverSpinGlassPEPSExt))

  # Two independent Boolean pairs: the minima are -0.875 at (1, 0)
  # and -1.125 at (1, 1). Adding 0.125 gives -1.875.
  Q = [0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.25; 0.0 0.0 0.0 0.0]
  l = [-0.875, -0.125, -0.375, -1.0]
  reference = (; value = -1.875, state = [1, 0, 1, 1])

  for topology in (TenSolver.SquareGrid(2, 2), TenSolver.KingGrid(2, 2))
    problem = PEPSBenchmarks.BenchmarkProblem("pair-fixture", topology, Q, l, 0.125)
    exact = PEPSBenchmarks.brute_force(problem; max_variables = 4)
    @test exact == reference
    result = PEPSBenchmarks.peps_result(
      problem,
      Ref(reference);
      beta = 2.0,
      maxdim = 4,
      iterations = 1,
      max_states = 16,
      cutoff_prob = 0.0,
      contraction = :svd,
      transformations = :identity,
    )
    @test result.status == "ok"
    @test result.objective ≈ reference.value
    @test result.gap ≈ 0.0 atol = 1e-12
    @test 1 <= result.states <= 16
    @test isfinite(result.runtime) && result.runtime >= 0
    @test result.discarded <= 0
    @test result.transform == string(PEPSBenchmarks.SpinGlassPEPS.rotation(0))
    @test isempty(result.note)
  end
end
