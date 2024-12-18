using Test, Random, LinearAlgebra

using  TenSolver

# Makes ITensor slower but catches more errors. Good for development.
import ITensors
ITensors.enable_debug_checks()

filepath(x) = joinpath(dirname(@__FILE__), x)

# Exact solvers. No testset on this file
include(filepath("utils.jl"))


@testset "Ill-formed input" begin
  dim = 4

  # Should throw when the matrix is not square
  Q = randn(dim, dim - 2)
  @test_throws BoundsError solve(Q)
end

# Traditional interface
include(filepath("qubo.jl"))
# JuMP interface
include(filepath("jump.jl"))

# Cases from papers
include(filepath("cases/vrp.jl"))
