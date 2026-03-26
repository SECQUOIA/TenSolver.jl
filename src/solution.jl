import ITensors, ITensorMPS
import ITensorMPS: MPS, siteinds

struct Distribution{T <: Real}
    tensor        :: MPS
    energies      :: Vector{T}
    bond_dims     :: Vector{Int}
    elapsed_times :: Vector{Float64}
end

# Convenience constructor for wrapping a bare MPS (e.g. in callbacks)
Distribution(tensor::MPS) = Distribution{Float64}(tensor, Float64[], Int[], Float64[])

# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample(psi)

Sample a bitstring from a (quantum) probability distribution.
"""
sample(psi::Distribution) = ITensorMPS.sample!(psi.tensor) .- 1

sample(psi::Distribution, n :: Integer) = [sample(psi) for _ in 1:n]

"""
    in(xs, psi::Distribution [; cutoff)

Whether the vector `xs` has a positive probability of being sampleable from `psi`.
When setting `cutoff`, it will be used as the minimum probability considered positive.
"""
function Base.in(bs, psi::Distribution; cutoff = 1e-8)
  return prob(psi, bs) > cutoff
end

function prob(psi::Distribution, bs)
  return abs2(coeff(psi, bs))
end

function coeff(psi::Distribution, bs)
  tn    = psi.tensor
  sites = siteinds(tn)
  psi0  = MPS(sites, string.(Int.(bs)))

  return inner(psi0,  tn)
end
