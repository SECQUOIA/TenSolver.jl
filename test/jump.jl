import JuMP

@testset "JuMP interface" begin
  dim = 5
  Q = 2*randn(dim, dim)
  l = 2*randn(dim)
  c = randn()
  obj(x) = dot(x, Q, x) + dot(l, x) + c

  m = JuMP.Model(TenSolver.Optimizer)
  @JuMP.variable(m, x[1:dim], Bin)
  @JuMP.objective(m, Min, dot(x, Q, x) + dot(l, x) + c)

  JuMP.optimize!(m)

  e = JuMP.objective_value(m)

  # ~:~ Exact solution ~:~ #
  e0, x0 = brute_force(obj, dim)
  # Same minimum value
  @test e ≈ e0
  # Same solution
  @test JuMP.value.(x) == x0
end

@testset "JuMP preprocess attribute" begin
  Q = [0.0 0.0 -2.0;
       0.0 0.0  0.0;
      -2.0 0.0  0.0]
  l = [0.5, 1.0, 0.5]

  m = JuMP.Model(TenSolver.Optimizer)
  JuMP.set_silent(m)
  JuMP.set_attribute(m, "preprocess", true)
  JuMP.set_attribute(m, "iterations", 3)
  @JuMP.variable(m, x[1:3], Bin)
  @JuMP.objective(m, Min, dot(x, Q, x) + dot(l, x))

  JuMP.optimize!(m)

  @test JuMP.objective_value(m) ≈ -3.0
  @test JuMP.value.(x) == [1, 0, 1]
end
