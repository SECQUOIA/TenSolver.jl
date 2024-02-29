module TenSolver

import TensorNetworks
import ITensors
import QUBODrivers
import QUBOTools

using LinearAlgebra

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
    e, x = TensorNetworks.solve_qubo(Q + diagm(L))

    λ = α * (e + β)
    ψ = ITensors.sample(x) .- 1
    s = QUBOTools.Sample{T,Int}(ψ, λ)

    return QUBOTools.SampleSet{T,Int}([s]; sense = :min, domain = :bool)
end

end # module TenSolver
