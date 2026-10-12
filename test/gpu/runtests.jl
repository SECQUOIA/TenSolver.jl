using Test, Adapt, CUDA, ITensors, TenSolver

if !(CUDA.functional())
  error("The GPU tests require a functional CUDA device")
end
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "..", "device_solve.jl"))

@testset "CUDA solves" begin
  @testset "Float64 storage" begin
    test_device_solve(x -> adapt(CuArray, x), t -> ITensors.data(t) isa CuArray{Float64})
  end
  @testset "CUDA.cu Float32 storage" begin
    test_device_solve(
      CUDA.cu,
      t -> ITensors.data(t) isa CuArray{Float32};
      atol = 1e-5,
      broken_tie_sampling = true,
    )
  end
end
