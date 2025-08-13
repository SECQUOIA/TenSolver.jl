import Aqua

@testset "Aqua.jl" begin
  Aqua.test_all(
    TenSolver;
    ambiguities = false,
    piracies=false,
  )
end
