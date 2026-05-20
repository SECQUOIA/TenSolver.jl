@testset "Benchmark helpers" begin
  include(joinpath(@__DIR__, "..", "benchmarks", "peps_common.jl"))

  square = PEPSBenchmarks.square_problem(2, 2; seed = 11)
  king = PEPSBenchmarks.king_problem(2, 2; seed = 12)

  @test square.topology == TenSolver.SquareGrid(2, 2)
  @test king.topology == TenSolver.KingGrid(2, 2)
  @test size(square.Q) == (4, 4)
  @test length(square.l) == 4

  exact = PEPSBenchmarks.brute_force(square; max_variables = 4)
  @test exact !== nothing
  @test length(exact.state) == 4
  @test PEPSBenchmarks.objective_value(square, exact.state) ≈ exact.value

  large = PEPSBenchmarks.square_problem(5, 5; seed = 13)
  @test PEPSBenchmarks.brute_force(large; max_variables = 4) === nothing
end
