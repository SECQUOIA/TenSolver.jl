import ITensors, ITensorMPS
import ITensorMPS: MPS, siteinds

struct Distribution
  tensor :: MPS
end

# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample(psi)

Sample a bitstring from a (quantum) probability distribution.
"""
sample(psi::Distribution) = ITensorMPS.sample!(psi.tensor) .- 1


function Base.in(bs, psi::Distribution; cutoff = 1e-8)
  tn = psi.tensor
  sites = siteinds(tn)

  psi0  = MPS(sites, [iszero(i) ? "0" : "1" for i in bs])

  return abs(dot(psi0, tn)) > cutoff
end
