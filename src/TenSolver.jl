module TenSolver

import ITensors, ITensorMPS
import QUBODrivers, QUBOTools

using LinearAlgebra

include("solution.jl")
export solve

include("solver.jl")
export sample


## ~:~ Welcome to the QUBOVerse ~:~ ##
# The functions below allow us to solve QUBO JuMP models
# with the solvers in this package.

QUBODrivers.@setup Optimizer begin
    name    = "TenSolver"
    version = v"0.1.0"
    # attributes = begin
    #     Cutoff["cutoff"]::Float64 = 1E-8
    # end
end

function QUBODrivers.sample(sampler::Optimizer{T}) where {T}
    n, L, Q, α, β = QUBOTools.qubo(sampler, :sparse; sense = :min)

    # Solve
    e, x = solve(Q, L)

    λ = α * (e + β)
    ψ = sample(x)
    s = QUBOTools.Sample{T,Int}(ψ, λ)

    return QUBOTools.SampleSet{T,Int}([s]; sense = :min, domain = :bool)
end

end # module TenSolver
