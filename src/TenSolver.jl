module TenSolver

import ITensors, ITensorMPS
import QUBODrivers, QUBOTools

using LinearAlgebra

include("solution.jl")
export sample

include("solver.jl")
export minimize


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
    # min_x a*(x'Qx + l'x + b)
    #  s.t. x in {0, 1}^n
    n, l, Q, a, b = QUBOTools.qubo(sampler, :sparse; sense = :min)

    # Solve
    energy, psi = minimize(Q, l, b)

    obj = a * energy
    x   = sample(psi)
    s   = QUBOTools.Sample{T,Int}(x, obj)

    return QUBOTools.SampleSet{T,Int}([s]; sense = :min, domain = :bool)
end

end # module TenSolver
