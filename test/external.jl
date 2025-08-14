import JuMP: MOI

@testset "QUBODrivers.jl" begin
  import QUBODrivers

  QUBODrivers.test(TenSolver.Optimizer)
end

@testset "Aqua.jl" begin
  import Aqua

  Aqua.test_all(
    TenSolver;
    ambiguities = (exclude=[MOI.supports],) ,
    piracies=(treat_as_own=[TenSolver.ITensors.state],)
  )
end
