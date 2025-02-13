using LinearAlgebra
import Combinatorics: multiset_permutations

using ITensorMPS, ITensors
import ITensorMPS: MPS, MPO, dmrg, OpSum, OpName, SiteType, StateName
import ITensorMPS: AbstractObserver


function issquare(A :: AbstractMatrix)
  nrows, ncols = size(A)
  return nrows == ncols
end

function maybe(f::Function, mx::Union{T, Nothing}; default=nothing) where T
  return isnothing(mx) ? default : f(mx)
end

variance(H::MPO, x::MPS) = inner(H, x, H, x) - inner(x, H, x)^2

mutable struct ConvergenceObserver <: AbstractObserver
  atol            :: Float64
  rtol            :: Float64
  variance_tol    :: Float64
  time_limit      :: Float64
  init_time       :: Float64
  previous_energy :: Float64

  ConvergenceObserver(atol, rtol, vtol=0.0, time_limit = +Inf) = new(atol, rtol, vtol, time_limit, time() , +Inf)
end

function ITensorMPS.checkdone!(o::ConvergenceObserver; energy, sweep, psi, outputlevel)
  stagnated = isapprox(energy, o.previous_energy; atol = o.atol, rtol = o.rtol)
  o.previous_energy = energy
  var = o.variance_tol > 0 ? variance(o.H, psi) : Inf

  return stagnated || var < o.variance_tol || time() - o.init_time > o.time_limit
end


# Diagonal matrix whose eigenvalues are the ordered feasible values for an integer variable.
# For qubits, this is a projection on |1>. Or equivalently, (I - σ_z) / 2.
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"D",::SiteType"Qudit", d::Int) = diagm(0:(d-1))

ITensors.state(::StateName"full", ::SiteType"Qudit", s::Index) = (d = dim(s); fill(1/sqrt(d), d))

"""
    tensorize(sites, p::AbstractArray{T, n})

Turn a quadratic/linear function acting on bitstrings
into an equivalent MPO Hamiltonian acting on Qudit sites.
The conversion consists of exchanging each integer variable `x_i`
for a matrix `P_i` whose eigenvalues represent its feasible set `K_i`.

    ∑ Q_ij x_i x_j + ∑ l_i x_i --> H = Σ Q_ij D_i D_j + ∑ l_i D_i
"""
function tensorize end

function tensorize(sites, Q::AbstractArray{T}, Qs...; cutoff = 1e-8) where T
  os = OpSum{T}()

  tensorize!(os, sites, Q; cutoff)
  for x in Qs
    tensorize!(os, sites, x; cutoff)
  end

  return isempty(os) ? MPO(T,sites) : MPO(T, os, sites)
end

tensorize!(os, sites, ::Nothing; cutoff = 1e-8) = os

# Construct the Hamiltonian H = Σ Q_ij P_i P_j
# The less operators in the sum, the fastest we can calculate an MPO.
# We use the following symmetries to simplify the construction:
# - For x in Bool, x^2 = x.
#   Thus, the Hamiltonian can be linear in the diagonal.
# - P_i commutes with P_j.
#   Thus, we're able to represent Q_ij and Q_ji with a single operator.
function tensorize!( os:: OpSum
                   , sites
                   , Q :: AbstractArray{T, 2}
                   ; cutoff = 1e-8
                   ) where {T}
  nbits = length(sites)

  for i in 1:nbits, j in (i+1):nbits
    coeff = sum(K -> Q[K...], multiset_permutations([i, j], 2))

    if abs(coeff) > cutoff
      if i == j && dim(sites[i]) == 2 # Code optimization for bits: x^2 = x
        os .+= (coeff, "D", i)
      else
        os .+= (coeff, "D", i, "D", j)
      end
    end
  end

  return os
end

function tensorize!( os:: OpSum
                  , sites
                  , l :: AbstractArray{T, 1}
                  ; cutoff = 1e-8
                  ) where {T}
  nbits = length(sites)

  # Linear term / Diagonal part
  for i in 1:nbits
    coeff = l[i]

    if abs(coeff) > cutoff   # Not representing ~zero coeffs produces a speedup
      os .+= (coeff, "D", i)
    end
  end

  return os
end

"""
    minimize(Q::Matrix[, l::Vector[, c::Number ; device, cutoff, kwargs...)

Solve the Quadratic Unconstrained Binary Optimization (QUBO) problem

    min  b'Qb + l'b + c
    s.t. b_i in {0, 1}

Return the optimal value `E` and a probability distribution `ψ` over optimal solutions.
You can use [`sample`](@ref) to get an actual bitstring from `ψ`.

This function uses DMRG with tensor networks to calculate the optimal solution,
by finding the ground state (least eigenspace) of the Hamiltonian

    H = Σ Q_ij D_iD_j + Σ l_i D_i

where D_i acts locally on the i-th qubit as [0 0; 0 1], i.e, the projection on |1>.

Keyword arguments:

- `iterations :: Int` - Maximum iterations the solver should run. Defaults to `10`.
- `cutoff :: Float64` - Any absolute value below this threshold is considered zero. Defaults to `1e-8`.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `maxdim` - The maximum allowed bond dimension.
  Integer or array of integer specifying the bond dimension per iteration.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `mindim` - The minimum allowed bond dimension, if possible.
  Integer or array of integer specifying the bond dimension per iteration.
- `time_limit :: Float64` - If specified, determines the maximum running time in seconds.
  It only determines whether a new iteration should start or not, thus the solver may run for longer if the threshold happens during an iteration.
- `device = cpu` - Accelerator device used during computation.
  See the section below for how to run on GPUs.
- `vtol :: Float64` - If specified, determines the variance tolerance before the algorithm stops.
  The variance test determines whether DMRG converged to an eigenstate (not necessarily the ground state),
  but is expensive to calculate.
- `noise` - A float or array of floats (per iteration) specifying the noise term added to the system to help with convergence.
  It is recommended to use a large noise (~ 1e-5) on the initial iterations and let it go to zero on later iterations.
- `eigsolve_krylovdim :: Int = 3` - Maximum Krylov space dimension used in the local eigensolver.
- `eigsolve_tol :: Float64 = 1e-14` - Eigensolver tolerance.
- `eigsolve_maxiter :: Int = 1` - Maximum iterations for eigensolver.

Running on GPU:

The optional keyword `device` controls whether the solver should run on CPU or GPU.
For using a GPU, you can import the respective package, e.g. CUDA.jl,
and pass its accelerator as argument.

```julia
import CUDA
minimize(Q; device = CUDA.cu)

import Metal
minimize(Q; device = Metal.mtl)
```


See also [`maximize`](@ref).
"""
function minimize( Q :: AbstractMatrix{T}
                 , l :: Union{AbstractVector{T}, Nothing} = nothing
                 , c :: T = zero(T)
                 ; cutoff     :: Float64  = 1e-8  #  a cutoff of 1E-5 gives sensible accuracy; a cutoff of 1E-8 is high accuracy; and a cutoff of 1E-12 is near exact accuracy. (https://itensor.org/docs.cgi?page=tutorials/dmrg_params)
                 , iterations :: Int      = 10
                 , device     :: Function = cpu
                 , time_limit :: Float64  = +Inf
                 , atol       :: Float64  = cutoff
                 , rtol       :: Float64  = atol
                 , vtol       :: Float64  = 0.0
                 , verbosity              = 1
                 # DMRG keywords
                 , maxdim                 = [10, 20, 50, 100, 100, 200]
                 , mindim                 = 1
                 , noise                  = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 0.0] # 1E-5 is a lot of noise and 1E-12 is a minimal amount of noise that can still be considered non-zero.
                 , eigsolve_krylovdim :: Int     = 3
                 , eigsolve_maxiter   :: Int     = 1
                 , eigsolve_tol       :: Float64 = 1e-14
                 # Work in progress
                 , preprocess :: Bool     = false
                 ) where {T}
  particles = size(Q)[1]

  # Quantization
  sites = ITensors.siteinds("Qudit", particles; dim = 2)
  H = tensorize(sites, Q, isnothing(l) ? diag(Q) : diag(Q) + l; cutoff)

  # Initial product state
  # Slight entanglement to help DMRG avoid local minima
  if preprocess
    Tri  = Tridiagonal(Q)
    psi0 = minimize(Tri; preprocess = false, mindim=10, cutoff, atol, rtol, vtol, iterations, time_limit, maxdim, noise, device, kwargs...)[2].tensor
    @show psi0
    for i in 1:particles
      psi0[i] *= ITensors.delta(T, siteind(psi0, i), sites[i])
    end
  else
    psi0 = random_mps(T, sites; linkdims=10)
  end
  observer = ConvergenceObserver(atol, rtol, vtol, time_limit)

  energy, tn = dmrg(device(H), device(psi0)
                    ; nsweeps     = iterations
                    , observer    = observer
                    , ishermitian = true
                    , outputlevel = verbosity
                    , cutoff
                    , maxdim
                    , mindim
                    , noise
                    , eigsolve_krylovdim
                    , eigsolve_tol
                    , eigsolve_maxiter
                    , eigsolve_verbosity = 0
               )
  # The calculated energy has approximation errors compared to the true solution.
  # It makes more sense to sample a solution and calculate the true objective function applied to it.
  psi = Distribution(tn)
  x = sample(psi)
  obj = dot(x, Q, x) + c + maybe(l -> dot(l,x), l; default=zero(T))

  return obj, psi
end

"""
    minimize(Q::Matrix, c::Number; kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem with no linear term.

    min  b'Qb + c
    s.t. b_i in {0, 1}

See also [`maximize`](@ref).
"""
minimize(Q :: AbstractMatrix{T}, c :: T; kwargs...) where T = minimize(Q, nothing, c; kwargs...)

"""
    maximize(Q::Matrix[, l::Vector[, c::Number; kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem
for maximization.

    max  b'Qb + l'b + c
    s.t. b_i in {0, 1}

See also [`minimize`](@ref).
"""
function maximize(qs... ; kwargs...)
  # Flip the sign of all non-nothing elements
  # max p(x) = - min -p(x)
  mqs = map(q -> maybe(-, q), qs)
  E, psi = minimize(mqs...; kwargs...)

  return -E, psi
end
