import ITensors: inner
import ITensorMPS: MPS, MPO, OpSum, @OpName_str, @SiteType_str, @StateName_str

import ITensors, ITensorMPS

import MultivariatePolynomials: AbstractPolynomial, coefficient, monomial, terms, variables, effective_variables, powers, isconstant

# Diagonal matrix whose eigenvalues are the ordered feasible values for an integer variable.
# For qubits, this is a projection on |1>. Or equivalently, (I - σ_z) / 2.
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"D",::SiteType"Qudit", d::Int) = diagm(0:(d-1))

function ITensors.state(::StateName"full", ::SiteType"Qudit", s::ITensorMPS.Index)
  d = ITensorMPS.dim(s)
  return fill(1/sqrt(d), d)
end

#----------------------------------------------------------#
# Interface                                                #
#----------------------------------------------------------#


"""
    DMRGBackend()

Select TenSolver's default ITensorMPS DMRG backend.
"""
struct DMRGBackend <: AbstractTenSolverBackend end

normalize_backend(::Val{:dmrg}) = default_backend

"""
    minimize(::DRMGBackend, Q::Matrix[, l::Vector[, c::Number ; kwargs...)

This function uses DMRG with tensor networks to calculate the optimal solution,
by finding the ground state (least eigenspace) of the Hamiltonian

    H = Σ Q_ij D_iD_j + Σ l_i D_i

where D_i acts locally on the i-th qubit as [0 0; 0 1], i.e, the projection on |1>.


Backend-specific keyword arguments:

- `constraints :: AbstractVector{<:AbstractConstraint}` - Experimental native Julia hard constraints.
  Defaults to `AbstractConstraint[]`. In constrained DMRG solves, TenSolver lowers each constraint to
  a projection MPO, solves the projected Hamiltonian, and returns a feasible sampled bitstring.
  For polynomial objectives, constraints are expressed in the same order as their `effective_variables`.
  If the constraints admit no solution at all, the solve does not error: it logs a warning and
  returns `+Inf` together with an infeasible [`Solution`](@ref) (see [`is_feasible`](@ref)).
- `maxdim` - The maximum allowed bond dimension.
  Integer or array of integer specifying the bond dimension per iteration.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `mindim` - The minimum allowed bond dimension, if possible.  Defaults to `1`.
  Integer or array of integer specifying the bond dimension per iteration.
- `vtol :: Float64` - If specified, determines the variance tolerance before the algorithm stops.
  The variance test determines whether DMRG converged to an eigenstate (not necessarily the ground state),
  but is expensive to calculate.
- `noise` - A float or array of floats (per iteration) specifying the noise term added to the system to help with convergence.
  It is recommended to use a large noise (~ 1e-5) on the initial iterations and let it go to zero on later iterations.
- `eigsolve_krylovdim :: Int = 3` - Maximum Krylov space dimension used in the local eigensolver.
- `eigsolve_tol :: Float64 = 1e-14` - Eigensolver tolerance.
- `eigsolve_maxiter :: Int = 1` - Maximum iterations for eigensolver.
"""
function minimize(::DMRGBackend, Q::AbstractMatrix{T}, l::AbstractVector{T}, c::T
  ; cutoff=1e-8
  , preprocess::Bool=false
  , domain_dim::Integer = 2
  , kwargs...
) where T
  Qp, lp, permutation = preprocess ? preprocess_qubo(Q, l, cutoff) : (Q, l, collect(1:size(Q, 1)))
  H      = tensorize(Qp, lp; cutoff, dim = domain_dim)
  obj(x) = dot(x, Q, x) + dot(l, x) + c

  return minimize_mpo(H, c, obj ; cutoff, permutation, domain_dim, kwargs...)
end

"""
    minimize(p::AbstractPolynomial ; kwargs...)

Solve the Polynomial Unconstrained Binary Optimization problem

    min p(b)
    s.t. b_i in {0, 1}

See also [`maximize`](@ref).
"""
function minimize(
  ::DMRGBackend,
  p::AbstractPolynomial{T}
  ;
  cutoff=1e-8,
  domain_dim::Integer = 2,
  kwargs...,
) where T
  H      = tensorize(p; cutoff, dim = domain_dim)
  cte    = constant_term(p)
  vs     = effective_variables(p)
  obj(x) = real(p(vs => x))

  return minimize_mpo(H, cte, obj ; cutoff, domain_dim, kwargs...)
end


#----------------------------------------------------------#
# The actual computations                                  #
#----------------------------------------------------------#

# Structural check for freshly tensorized zero objectives; this is not a
# semantic zero-operator test after arbitrary MPO algebra.
is_zero_tensor(H::MPO; cutoff) = norm(H) <= cutoff
is_zero_tensor(p::MPS; cutoff) = norm(p) <= cutoff

expectation(H::MPO, x::MPS) = real(inner(x', H, x))

variance(H::MPO, x::MPS) = real(inner(H, x, H, x)) - expectation(H, x)^2

# Strict upper triangular part of an array
function upper_indices(a)
  return (Tuple(ci)
            for ci in CartesianIndices(size(a))
            if issorted(Tuple(ci); lt = <))
end

function constant_term(p::AbstractPolynomial{T}) where T
  ts  = terms(p)
  idx = findfirst(isconstant, ts)

  return isnothing(idx) ? zero(T) : coefficient(ts[idx])
end

# Returns `nothing` when the projected state has no feasible amplitude,
# which at the initial state proves the constraints are infeasible.
function constrained_initial_state(T, sites, projections; cutoff, inidim)
  psi = ITensorMPS.random_mps(T, sites; linkdims=inidim)
  return isempty(projections) ? psi : project_feasible_state(psi, projections; cutoff)
end

function project_feasible_state(psi::MPS, projections; cutoff)
  projected = project_state(psi, projections; cutoff)
  return is_zero_tensor(projected; cutoff) ? nothing : normalize(projected)
end

# Infeasibility is a solver outcome, not an argument error: report it as a
# status like other optimization packages (objective +Inf, empty Solution),
# so it maps onto MOI.INFEASIBLE at the JuMP layer. See the discussion in #94.
function infeasible_result(::Type{T}) where {T}
  @warn "constraints define an empty feasible subspace"
  F = float(T)
  return real(T)(+Inf), infeasible_solution(F)
end


"""
    tensorize(p)

Turn a polynomial function action on bitstrings
into an equivalent MPO Hamiltonian acting on Qudit sites.
The conversion consists of exchanging each integer variable `x_i`
for a matrix `P_i` whose eigenvalues represent its feasible set `K_i`.

    ∑ Q_ij x_i x_j + ∑ l_i x_i --> H = Σ Q_ij D_i D_j + ∑ l_i D_i
"""
function tensorize end

function tensorize(Q::AbstractArray{T}, rest::Vararg{AbstractArray{T}}; cutoff = zero(T), dim::Integer) where T
  Qs = [Q, rest...]
  if !allequal(Iterators.flatmap(size, Qs))
    throw(DimensionMismatch("All arrays should act on the same number of variables.\nEncountered dimensions $(collect(map(size, Qs)))."))
  end

  N = size(Q, 1)
  sites = ITensors.siteinds("Qudit", N; dim = dim)
  os = OpSum{T}()

  for t in Qs
    for idx in upper_indices(t)
      # Due to commutativity, we can group all terms that have the same indices.
      # This speeds up the calculations.
      # e.g., Q_ij x_i x_j + Q_ji x_j x_i = (Q_ij + Q_ji) x_i x_j
      coeff = sum(k -> t[k...], multiset_permutations(idx, ndims(t)))

      if abs(coeff) > cutoff
        op   = Iterators.flatmap(v -> ("D", v), idx)
        os .+= (coeff, op...)
      end
    end
  end

  return isempty(os) ? MPO(T,sites) : MPO(T, os, sites)
end

function tensorize(p::AbstractPolynomial{T}; cutoff = zero(T), dim::Integer) where T
  N = length(effective_variables(p))
  sites = ITensors.siteinds("Qudit", N; dim = dim)
  os = OpSum{T}()

  # Map: var name => index
  indices = Dict(v => i for (i, v) in enumerate(effective_variables(p)))

  for t in terms(p)
    coeff = coefficient(t)

    if abs(coeff) > cutoff && ! isconstant(t)
      op = Iterators.flatten(
        map(powers(t)) do p
          v, e = p
          Iterators.flatten(Iterators.repeated(("D", indices[v]), e))
        end
      )
      os .+= (coeff, op...)
    end
  end

  return isempty(os) ? MPO(T,sites) : MPO(T, os, sites)
end

function minimize_mpo( H :: MPO
                     , c :: T
                     , obj
                     ; device      = cpu
                     , cutoff      = 1e-8  #  a cutoff of 1E-5 gives sensible accuracy; a cutoff of 1E-8 is high accuracy; and a cutoff of 1E-12 is near exact accuracy. (https://itensor.org/docs.cgi?page=tutorials/dmrg_params)
                     , verbosity   = 1
                     , constraints = AbstractConstraint[]
                     , domain_dim :: Integer
                     # Stopping criteria
                     , iterations :: Union{Nothing, Int} = nothing
                     , time_limit = +Inf
                     , vtol       = cutoff
                     , check_variance_every_iteration = 10
                     # DMRG keywords
                     , inidim     = 40
                     , maxdim     = [10, 10, 10, 20, 50, 100, 100, 200, 300, 300, 400, 400, 800, 900, 1000]
                     , mindim     = 1
                     , noise      = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 0.0] # 1E-5 is a lot of noise and 1E-12 is a minimal amount of noise that can still be considered non-zero.
                     , eigsolve_krylovdim :: Int     = 3
                     , eigsolve_maxiter   :: Int     = 2
                     , eigsolve_tol       :: Float64 = 1e-14
                     # Iteration callback
                     , on_iteration     :: Union{Nothing, Function} = nothing
                     , callback_every   :: Int = 1
                     , permutation :: Vector{Int} = collect(1:length(H))
                     ) where {T}
  callback_every >= 1 || throw(ArgumentError("`callback_every` must be >= 1, got $callback_every"))
  initial_time      = time()
  energies_log      = T[]
  bond_dims_log     = Int[]
  elapsed_times_log = Float64[]

  # Quantization
  sites = ITensorMPS.siteinds(first, H; plev=0)

  # Constraints
  projections = map(
    device,
    projection_mpos(T, constraints, sites; permutation),
  )

  # Hamiltonian construction
  H = device(H)
  H = is_zero_tensor(H; cutoff) ? H : project_hamiltonian(H, projections; cutoff)

  # Initial state
  psi = constrained_initial_state(T, sites, projections; cutoff, inidim)
  if isnothing(psi)
    return infeasible_result(T)
  end
  psi = device(psi)

  @debug(
    "Constraint projection MPO construction finished",
    projection_max_bond = map(ITensorMPS.maxlinkdim, projections),
    projected_hamiltonian_max_bond = ITensorMPS.maxlinkdim(H),
    projected_initial_max_bond = ITensorMPS.maxlinkdim(psi),
    time=(time() - initial_time),
  )

  iterlog_header(verbosity)
  var    = T(Inf)
  energy = T(Inf)

  for i in Iterators.countfrom(1)
    energy, psi = groundstate(H, device(psi)
                      ; projections
                      , nsweeps     = 1
                      , ishermitian = true
                      , outputlevel = 0
                      , cutoff
                      , maxdim = maxdim[min(i, length(maxdim))]
                      , mindim = mindim[min(i, length(mindim))]
                      , noise  = noise[min(i, length(noise))]
                      , eigsolve_krylovdim
                      , eigsolve_tol
                      , eigsolve_maxiter
                      , eigsolve_verbosity = 0
                 )

    # Get metadata #
    if i % check_variance_every_iteration == 0
      vtime = time()
      var = variance(H, psi)
      @debug "Calculate variance" variance=var time=(time() - vtime())
    end

    elapsed_time = time() - initial_time

    bond_dim = ITensorMPS.maxlinkdim(psi)

    iterlog_iteration(
      verbosity,
      i,
      energy + c,
      bond_dim,
      i % check_variance_every_iteration == 0 ? var : nothing,
      elapsed_time,
    )

    # Per-iteration stats (always collected)
    push!(energies_log,      energy + c)
    push!(bond_dims_log,     bond_dim)
    push!(elapsed_times_log, elapsed_time)

    # Optional callback
    if !isnothing(on_iteration) && i % callback_every == 0
      on_iteration(psi; iteration=i, objective=energy+c, bond_dim, elapsed_time)
    end

    # Stopping criteria #
    if !isnothing(iterations) && i >= iterations
      @debug "Stopping: maximum iterations reached" iteration=i
      break
    elseif elapsed_time > time_limit
      @debug "Stopping: maximum time reached" iteration=i limit=time_limit time=elapsed_time
      break
    elseif var < vtol
      @debug "Stopping: variance below tolerance" iteration=i variance=var tolerance=vtol
      break
    elseif length(H) == 1
      @debug "Stopping: solver is exact for n=1" iteration=i
      break
    end
  end


  if isinf(energy)
    optimal, dist = infeasible_result(T)
  else
    # The calculated energy has approximation errors compared to the true solution.
    # It makes more sense to sample a solution and calculate the true objective function applied to it.
    dist = Solution{T}(psi, energies_log, bond_dims_log, elapsed_times_log, permutation)
    optimal = obj(sample(dist))
  end

  elapsed_time = time() - initial_time
  iterlog_footer(verbosity, optimal, elapsed_time)
  return optimal, dist
end

function groundstate(H::MPO, psi0::MPS; projections, cutoff=1e-8, kwargs...)
  if length(psi0) != 1
    energy, psi = ITensorMPS.dmrg(H, psi0; cutoff, kwargs...)

    # Re-project each iteration so the sampled bitstring is guaranteed feasible.
    # In exact arithmetic the sweep keeps a feasible start feasible (the local
    # eigensolver only ever applies P'HP to a feasible state), but the injected
    # `noise` term and SVD truncation can leak amplitude into the infeasible
    # subspace. That subspace is the kernel of the projections, where P'HP has
    # zero energy, so the leaked amplitude is never penalized back out on its own.
    psi = project_feasible_state(psi, projections; cutoff)
  else
    # ITensorMPS.dmrg does not support single-site systems, so solve the n=1
    # problem by directly comparing basis states. When constrained, we must
    # restrict the search to feasible basis states: the projected Hamiltonian
    # P'HP assigns zero energy to the infeasible subspace (the kernel of the
    # projections), so picking the global lowest-energy basis state would select
    # an infeasible one whenever the feasible objective is positive.
    s = only(ITensorMPS.siteinds(psi0))
    Ht = only(H)
    pt = only(psi0)

    feasible = filter(a -> !iszero(pt[s => a]), 1:ITensorMPS.dim(s))

    if isempty(feasible)
      energy = Inf
      psi = nothing
    else
      energies = [real(Ht[s => a, ITensors.prime(s) => a]) for a in feasible]
      energy = minimum(last, energies)
      optimal = [a for (a, e) in zip(feasible, energies) if e == energy]

      basis = ITensors.ITensor(eltype(pt), s)
      foreach(a -> (basis[s => a] = 1), optimal)
      psi = ITensorMPS.MPS([basis])
      ITensorMPS.normalize!(psi)
    end
  end

  # Unlike the initial state, a mid-run collapse does not prove
  # infeasibility (the start was feasible)
  if isnothing(psi)
    error(
      "constrained DMRG produced a state with zero feasible amplitude; constraints may be infeasible or the projection cutoff may be too large",
    )
  end

  return energy, psi
end
