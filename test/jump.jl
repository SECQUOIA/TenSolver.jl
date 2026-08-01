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
