import ITensors, ITensorMPS
import ITensorMPS: MPS

struct Distribution
  tensor :: MPS
end

# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample(psi)

Sample a bitstring from a (quantum) probability distribution.
"""
sample(psi::Distribution) = ITensors.sample!(psi.tensor) .- 1
