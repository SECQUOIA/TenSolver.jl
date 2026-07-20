#=
  Utility functions for testing.
  Here we have slower but exact solvers.
=#

"""
    brute_force(f, n[, constraints]; domain = 0:1)

A QUBO solver using a brute force approach instead of Tensor networks.
Despite being painfully slow, this is useful as a sanity check.
"""
function brute_force(obj, n, constraints = AbstractConstraint[]; domain = 0:1)
  best = +Inf
  solution = Vector{Float64}[]

  for bits in Iterators.product(fill(domain, n)...)
    x = collect(bits)

    if is_feasible(x, constraints)
      value = obj(x)
      if value < best
        best = value
        solution = x
      end
    end
  end

  isempty(solution) && throw(ArgumentError("no feasible bitstring"))
  return best, solution
end

function mpo_matrix_element(H, sites, bra_bits, ket_bits)
  bra = ITensorMPS.MPS(sites, string.(bra_bits))
  ket = ITensorMPS.MPS(sites, string.(ket_bits))
  return real(ITensors.inner(bra', H, ket))
end

function mps_amplitude(psi, sites, bits)
  basis = ITensorMPS.MPS(sites, string.(bits))
  return real(ITensors.inner(basis, psi))
end
