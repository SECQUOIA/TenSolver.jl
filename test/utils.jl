#=
  Utility functions for testing.
  Here we have slower but exact solvers.
=#

"""
    brute_force(f, n)

A QUBO solver using a brute force approach instead of Tensor networks.
Despite being painfully slow, this is useful as a sanity check.
"""
function brute_force(f::Function, T, n::Int64)
  min_now  = +Inf
  solution = Vector{T}[]

  for i in 0:(2^n-1)
    # The Boolean vector corresponding to the natural i
    bs = [convert(T, parse(Bool, d)) for d in last(bitstring(i), n)]
    z  = f(bs)

    if z < min_now
      solution = bs
      min_now  = z
    end

  end

  return min_now, solution
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
