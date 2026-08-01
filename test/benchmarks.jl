using LinearAlgebra

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

  (; J, h, offset) = PEPSBenchmarks.peps_form(square)
  spins = TenSolver.bool_to_spin(exact.state)
  @test dot(spins, J, spins) + dot(h, spins) + offset ≈ exact.value
  @test all(in((-1, 1)), spins)

  large = PEPSBenchmarks.square_problem(5, 5; seed = 13)
  @test PEPSBenchmarks.brute_force(large; max_variables = 4) === nothing
end
