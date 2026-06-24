using Test, Random, LinearAlgebra

using  TenSolver

# Makes ITensor slower but catches more errors. Good for development.
import ITensors
ITensors.enable_debug_checks()

filepath(x) = joinpath(dirname(@__FILE__), x)

# Exact solvers. No testset on this file
include(filepath("utils.jl"))


#----------------------------------------------------------#
#                         Test sets                        #
#----------------------------------------------------------#

# Ising <-> QUBO utilities
include(filepath("ising_conversion.jl"))

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

# QUBODrivers.jl and Aqua.jl test suites
include(filepath("external.jl"))

# Documentation tests
include(filepath("doctests.jl"))
