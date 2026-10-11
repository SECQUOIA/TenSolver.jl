using Test, Random, LinearAlgebra
using TenSolver

# This lane requires the optional package and extension; an import failure fails CI.
@test realpath(dirname(dirname(pathof(TenSolver)))) == realpath(joinpath(@__DIR__, "../.."))
# Exercise benchmark loading in a fresh session before explicitly importing PEPS.
include("benchmarks.jl")
import SpinGlassPEPS
@test !isnothing(Base.get_extension(TenSolver, :TenSolverSpinGlassPEPSExt))
include("../utils.jl")
include("../peps_backend.jl")
include("jump.jl")
