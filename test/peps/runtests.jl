using Test, Random, LinearAlgebra
using TenSolver
import SpinGlassPEPS

# This lane requires the optional package and extension; an import failure fails CI.
@test realpath(dirname(dirname(pathof(TenSolver)))) == realpath(joinpath(@__DIR__, "../.."))
@test !isnothing(Base.get_extension(TenSolver, :TenSolverSpinGlassPEPSExt))
include("../utils.jl")
include("../peps_backend.jl")
