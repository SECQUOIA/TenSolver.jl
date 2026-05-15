using Test, Random, LinearAlgebra

using  TenSolver

# Makes ITensor slower but catches more errors. Good for development.
import ITensors
ITensors.enable_debug_checks()

filepath(x) = joinpath(dirname(@__FILE__), x)

# Exact solvers. No testset on this file
include(filepath("utils.jl"))

include(filepath("ising_conversion.jl"))
include(filepath("backend.jl"))
include(filepath("peps_backend.jl"))

@testset "Ill-formed input" begin
  dim = 4

  # Should throw when the matrix is not square
  Q = randn(dim, dim - 2)
  @test_throws DimensionMismatch minimize(Q)

  Q = randn(dim - 2, dim)
  @test_throws DimensionMismatch minimize(Q)
end

# Traditional interface
include(filepath("qubo.jl"))
include(filepath("pubo.jl"))
# Iteration Log utilities
include(filepath("log.jl"))
# JuMP interface
include(filepath("jump.jl"))
# QUBODrivers.jl and Aqua.jl test suites
include(filepath("external.jl"))

# Cases from papers
include(filepath("cases/vrp.jl"))

# HDF5 snapshot callback
include(filepath("hdf5.jl"))

# Documentation tests
include(filepath("doctests.jl"))
