import JuMP
import QUBODrivers

@testset "QUBODrivers.jl" begin
  QUBODrivers.test(TenSolver.Optimizer)
end


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
  e0, x0 = brute_force(obj, Float64, dim)
  # Same minimum value
  @test e ≈ e0
  # Same solution
  @test JuMP.value.(x) == x0
end
