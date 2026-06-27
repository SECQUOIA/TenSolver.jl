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

#----------------------------------------------------------#
#                         Test sets                        #
#----------------------------------------------------------#

# Ising <-> QUBO utilities
include(filepath("ising_conversion.jl"))

# Binary constraint API
include(filepath("constraints.jl"))

# Traditional interface
include(filepath("qubo.jl"))
include(filepath("pubo.jl"))

# JuMP interface
include(filepath("jump.jl"))

# Iteration Log utilities
include(filepath("log.jl"))

# Cases from papers
@testset "Real Models" begin
  include(filepath("cases/pharma.jl"))
  include(filepath("cases/vrp.jl"))
end

# HDF5 snapshot callback
include(filepath("hdf5.jl"))

# Benchmark helper smoke tests
include(filepath("benchmarks.jl"))

# QUBODrivers.jl and Aqua.jl test suites
include(filepath("external.jl"))

# Documentation tests
include(filepath("doctests.jl"))
