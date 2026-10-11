import JuMP

@testset "JuMP interface and preprocess attribute" begin
  Q = [0.0 0.0 -2.0;
       0.0 0.0  0.0;
      -2.0 0.0  0.0]
  l = [0.5, 1.0, 0.5]
  c = 1.25

  m = JuMP.Model(TenSolver.Optimizer)
  JuMP.set_silent(m)
  JuMP.set_attribute(m, "preprocess", true)
  JuMP.set_attribute(m, "iterations", 3)
  @JuMP.variable(m, x[1:3], Bin)
  @JuMP.objective(m, Min, dot(x, Q, x) + dot(l, x) + c)

  JuMP.optimize!(m)

  @test JuMP.objective_value(m) ≈ -1.75
  @test JuMP.value.(x) == [1, 0, 1]
end

@testset "JuMP QUBO bridge selects binary domain" begin
  # Values at (0,0), (1,0), (0,1), (1,1) are respectively
  # 3.75, 1.25, 13.75, 16.25, including scaling and the offset.
  for (sense, expected, assignment) in (
    (JuMP.MOI.MIN_SENSE, 1.25, [1, 0]),
    (JuMP.MOI.MAX_SENSE, 16.25, [1, 1]),
  ), preprocess in (false, true)
    m = JuMP.Model(TenSolver.Optimizer)
    JuMP.set_silent(m)
    JuMP.set_attribute(m, "preprocess", preprocess)
    JuMP.set_attribute(m, "iterations", 3)
    JuMP.set_attribute(m, "num_reads", 7)
    JuMP.set_attribute(m, TenSolver.QUBODrivers.FinalNumberOfReads(), 5)
    @JuMP.variable(m, x[1:2], Bin)
    JuMP.set_objective_sense(m, sense)
    JuMP.set_objective_function(m, 2.5 * (2x[1]^2 - 3x[1] + 4x[2] + 2x[1] * x[2]) + 3.75)

    JuMP.optimize!(m)

    @test JuMP.objective_value(m) ≈ expected
    @test JuMP.value.(x) == assignment
    raw = JuMP.unsafe_backend(m)
    sampleset = TenSolver.QUBOTools.solution(raw)
    @test sum(TenSolver.QUBOTools.reads(s) for s in sampleset) == 5
    @test TenSolver.QUBOTools.metadata(sampleset)["reads"]["number_of_reads"] == 7
    @test TenSolver.QUBOTools.metadata(sampleset)["reads"]["final_number_of_reads"] == 5
    for result in 1:JuMP.result_count(m)
      @test JuMP.value.(x; result) == assignment
      @test JuMP.objective_value(m; result) ≈ expected
    end
  end
end
