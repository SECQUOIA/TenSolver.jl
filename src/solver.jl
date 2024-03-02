import ITensors

function issquare(A :: AbstractMatrix)
  nrows, ncols = size(A)
  return nrows == ncols
end

# Projection on |1>, or equivalently, (I - σ_z) / 2
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"P1",::SiteType"Qubit") = [0 0 ; 0 1]

"""
    tensorize_qubo(Q, sites)

Turn a square matrix into an equivalent MPO Hamiltonian acting on Qubit sites.

    Q --> H = Σ Q_ij P_i P_j
"""
function tensorize_qubo(Q :: AbstractMatrix{T}, sites; cutoff = 1e-8) where {T}
  dim = length(sites)

  os = ITensors.OpSum()
  # Construct the Hamiltonian H = Σ Q_ij P_i P_j
  # The less operators in the sum, the fastest we can calculate an MPO.
  # We use the following simmetries to simplify the construction:
  # - For x in Bool, x^2 = x.
  #   Thus, the Hamiltonian can be linear in the diagonal.
  # - P_i commutes with P_j.
  #   Thus, we're able to represent Q_ij and Q_ji with a single operator.
  for i in 1:dim, j in i:dim
    if j == i # Diagonal part
      coeff = Q[i, i]
      if abs(coeff) > cutoff   # Not representing ~zero coeffs produces a speedup
        os += (coeff, "P1", i)
      end
    else      # Upper and Lower parts together
      coeff = Q[j, i] + Q[i, j]
      if abs(coeff) > cutoff
        os += (coeff, "P1", i, "P1", j)
      end
    end
  end

  return ITensors.MPO(T, os, sites)
end

"""
    solve_qubo(Q)

Solve the Quadratic Unconstrained Binary Optimization problem

    min <b|Q|b> s.t. b_i in {0, 1}

This function uses DMRG with tensor networks the calculate the optimal solution,
by finding the ground state of the Hamiltonian

    H = Σ Q_ij P_iP_j

where P_i acts locally on the i-th qubit as [0 0; 0 1], i.e, the projection on |1>.
"""
function solve_qubo(Q :: AbstractMatrix{T}
                   ; cutoff  = 1e-8
                   , nsweeps :: Int = 5
                   , maxdim  = [10, 20, 100, 100, 200]
                   ) where {T}
  dim   = size(Q)[1]
  sites = ITensors.siteinds("Qubit", dim)
  H     = tensorize_qubo(Q, sites; cutoff)

  # Initial product state
  psi0  = ITensors.MPS(sites, "+")  # ⨂ (|0> + |1>) / √2

  energy, psi = ITensors.dmrg(H, psi0; nsweeps, maxdim, cutoff)

  return energy, psi
end


# Sample from |ψ> in the {0, 1} world instead of 1-based Julia index world.
"""
    sample_solution!(psi)

Sample a bitstring from a Tensor in MPS-form,
representing a distribution of Qubits.
"""
sample_solution(psi) = ITensors.sample!(psi) .- 1


"""
  brute_force_qubo(Q)

A version of `solve_qubo` that uses a brute force approach instead of Tensor networks.
Despite being painfully slow, this is useful as a sanity check.
"""
function brute_force_qubo(Q :: Matrix{T}) where T
  @assert issquare(Q)
  n   = size(Q)[1]

  min_now   = +Inf
  solution = Vector{Float64}[]

  for i in 0:(2^n-1)
    # The Boolean vector corresponding to the natural i
    digits  = [parse(Bool, d) for d in last(bitstring(i), n)]
    bs      = digits #map(d -> convert(T, d), digits)
    z       = bs'Q*bs

    if z < min_now
      solution = bs
      min_now  = z
    end

  end

  return min_now, solution
end

