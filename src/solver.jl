using LinearAlgebra
import Combinatorics: multiset_permutations

using ITensorMPS, ITensors
import ITensorMPS: MPS, MPO, dmrg, OpSum, OpName, SiteType, StateName


function issquare(A :: AbstractMatrix)
  nrows, ncols = size(A)
  return nrows == ncols
end

# Diagonal matrix whose eigenvalues are the ordered feasible values for an integer variable.
# For qubits, this is a projection on |1>. Or equivalently, (I - σ_z) / 2.
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"D",::SiteType"Qudit", d::Int) = diagm(0:(d-1))

ITensors.state(::StateName"full", ::SiteType"Qudit", s::Index) = (d = dim(s); fill(1/sqrt(d), d))

"""
    tensorize(Q, l, sites)

Turn a quadratic objective function acting on bitstrings,

    min_x x'Qx + l'x
     s.t. x_i in K_i ⊆ Z

Into an equivalent MPO Hamiltonian acting on Qudit sites.
The conversion consists of exchanging each variable `x_i`
for a matrix `P_i` whose eigenvalues represent its feasible set `K_i`.

    ∑ Q_ij x_i x_j + ∑ l_i x_i --> H = Σ Q_ij D_i D_j + ∑ l_i D_i
"""
function tensorize( sites
                  , Q :: Union{AbstractMatrix{T}, Nothing} = nothing
                  , l :: Union{AbstractVector{T}, Nothing} = nothing
                  ; cutoff = 1e-8
                  ) where {T}
  nbits = length(sites)
  os = OpSum{T}()
  # Construct the Hamiltonian H = Σ Q_ij P_i P_j
  # The less operators in the sum, the fastest we can calculate an MPO.
  # We use the following symmetries to simplify the construction:
  # - For x in Bool, x^2 = x.
  #   Thus, the Hamiltonian can be linear in the diagonal.
  # - P_i commutes with P_j.
  #   Thus, we're able to represent Q_ij and Q_ji with a single operator.

  # Linear term / Diagonal part
  if l !== nothing
    for i in 1:nbits
      coeff = l[i]

      if abs(coeff) > cutoff   # Not representing ~zero coeffs produces a speedup
        os .+= (coeff, "D", i)
      end
    end
  end

  # Loop on upper triangular part
  if Q !== nothing
    for i in 1:nbits, j in i:nbits
      coeff = sum(K -> Q[K...], multiset_permutations([i, j], 2))

      if abs(coeff) > cutoff
        if i == j && dim(sites[i]) == 2 # Code optimization for bits: x^2 = x
          os .+= (coeff, "D", i)
        else
          os .+= (coeff, "D", i, "D", j)
        end
      end
    end
  end

  return MPO(T, os, sites)
end

"""
    solve(Q [, l, c ; device, cutoff, kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem

    min  b'Qb + l'b + c
    s.t. b_i in {0, 1}

This function uses DMRG with tensor networks to calculate the optimal solution,
by finding the ground state (least eigenspace) of the Hamiltonian

    H = Σ Q_ij D_iD_j + Σ l_i D_i

where D_i acts locally on the i-th qubit as [0 0; 0 1], i.e, the projection on |1>.

The optional keyword `device` controls whether the solver should run on CPU or GPU.
For using a GPU, you can import the respective package, e.g. CUDA.jl,
and pass its accelerator as argument.

```julia
import CUDA
solve(Q; device = CUDA.cu)

import Metal
solve(Q; device = Metal.mtl)
```
"""
function solve( Q :: AbstractMatrix{T}
              , l :: Union{AbstractVector{T}, Nothing} = nothing
              , c :: T = zero(T)
              ; cutoff = 1e-8
              , iterations :: Int = 10
              , maxdim = [10, 20, 100, 100, 200]
              , device :: Function = identity
              , kwargs...
              ) where {T}
  particles = size(Q)[1]

  # Quantization
  sites = ITensors.siteinds("Qudit", particles; dim = 2)
  H     = tensorize(sites, Q, l; cutoff)

  # Initial product state
  # Slight entanglement to help DMRG avoid local minima
  psi0 = random_mps(T, sites; linkdims=2)

  energy, psi = dmrg(device(H), device(psi0)
                    ; nsweeps = iterations
                    , maxdim, cutoff, kwargs...)

  return energy + c, Distribution(psi)
end
