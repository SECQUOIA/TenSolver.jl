import ITensors: inner
import ITensorMPS: MPS, MPO, dmrg, OpSum, @OpName_str, @SiteType_str, @StateName_str, dim

import ITensors, ITensorMPS

# Diagonal matrix whose eigenvalues are the ordered feasible values for an integer variable.
# For qubits, this is a projection on |1>. Or equivalently, (I - σ_z) / 2.
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"D",::SiteType"Qudit", d::Int) = diagm(0:(d-1))

ITensors.state(::StateName"full", ::SiteType"Qudit", s::ITensorMPS.Index) = (d = dim(s); fill(1/sqrt(d), d))

expectation(H, x) = inner(x', H, x)
variance(H::MPO, x::MPS) = real(inner(H, x, H, x) - expectation(H, x)^2)

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
- `feasible_sample_retries :: Int` - Maximum attempts to draw a feasible bitstring from a constrained
  final state before throwing a diagnostic error. Defaults to `100`.
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
function minimize(
  ::DMRGBackend,
  Q::AbstractMatrix{T},
  l::Union{AbstractVector{T}, Nothing}=nothing,
  c::T=zero(T)
  ;
  cutoff=1e-8,
  preprocess::Bool=false,
  constraints=AbstractConstraint[],
  kwargs...,
) where T
  constraint_vec = normalize_constraints(constraints)
  Qp, lp, permutation = preprocess ? preprocess_qubo(Q, l, cutoff) : (Q, l, collect(1:size(Q, 1)))
  H      = tensorize(Qp, isnothing(lp) ? diag(Qp) : diag(Qp) + lp; cutoff)
  obj(x) = dot(x, Q, x) + c + maybe(l -> dot(l,x), l; default=zero(T))

  if isempty(constraint_vec)
    return minimize_mpo(H, c, obj; cutoff, permutation, kwargs...)
  end

  tensor_constraints = constraint_reindex(constraint_vec, permutation)
  H_eff, initial_state, projections = constrained_projection_problem(T, H, tensor_constraints; cutoff)

  return minimize_mpo(
    H_eff,
    c,
    obj;
    cutoff,
    permutation,
    initial_state,
    solution_projection=projections,
    sample_validator=x -> is_feasible(x, constraint_vec),
    kwargs...,
  )
end

"""
    minimize(p::AbstractPolynomial ; kwargs...)

Solve the Polynomial Unconstrained Binary Optimization problem

    min p(b)
    s.t. b_i in {0, 1}

See also [`maximize`](@ref).
"""
function minimize(::DMRGBackend, p::AbstractPolynomial{T}; cutoff=1e-8, constraints=AbstractConstraint[], kwargs...) where T
  constraint_vec = normalize_constraints(constraints)
  H   = tensorize(p)
  cte = constant(p)
  vs = variables(p)

  if isempty(constraint_vec)
    return minimize_mpo(H, cte, a -> p(vs => a); cutoff, kwargs...)
  end

  H_eff, initial_state, projections = constrained_projection_problem(T, H, constraint_vec; cutoff)

  return minimize_mpo(
    H_eff,
    cte,
    a -> p(vs => a);
    cutoff,
    initial_state,
    solution_projection=projections,
    sample_validator=x -> is_feasible(x, constraint_vec),
    kwargs...,
  )
end


#----------------------------------------------------------#
# The actual computations                                  #
#----------------------------------------------------------#

function normalize_constraints(constraints)
  return collect(AbstractConstraint, constraints)
end

function constrained_projection_problem(::Type{T}, H, constraints; cutoff) where {T}
  sites = ITensorMPS.siteinds(first, H; plev=0)
  projections = projection_mpos(T, constraints, sites)
  initial_state = constrained_initial_state(sites, projections; cutoff)
  H_eff = is_zero_mpo(H; cutoff) ? H : project_hamiltonian(H, projections; cutoff)

  @debug(
    "Constraint projection MPO construction finished",
    projection_max_bond=map(ITensorMPS.maxlinkdim, projections),
    projected_hamiltonian_max_bond=ITensorMPS.maxlinkdim(H_eff),
    projected_initial_max_bond=ITensorMPS.maxlinkdim(initial_state),
  )

  return H_eff, initial_state, projections
end

# Structural check for freshly tensorized zero objectives; this is not a
# semantic zero-operator test after arbitrary MPO algebra.
is_zero_mpo(H::MPO; cutoff=0) = norm(H) < cutoff

function constrained_initial_state(sites, projections; cutoff)
  psi = ITensorMPS.MPS(sites, fill("full", length(sites)))
  return project_feasible_state(
    psi,
    projections;
    cutoff,
    error_type=ArgumentError,
    message="constraints define an empty feasible subspace; no binary vector satisfies all constraints",
  )
end

function project_feasible_state(psi::MPS, projections; cutoff, error_type, message)
  projected = project_state(psi, projections; cutoff)

  if iszero(norm(projected))
    throw(error_type(message))
  end

  return normalize(projected)
end

function sampled_objective(dist, obj; sample_validator, feasible_sample_retries)
  feasible_sample_retries >= 1 ||
    throw(ArgumentError("`feasible_sample_retries` must be >= 1, got $feasible_sample_retries"))

  if isnothing(sample_validator)
    return obj(sample(dist))
  end

  local last_sample
  for _ in 1:feasible_sample_retries
    x = sample(dist)
    sample_validator(x) && return obj(x)
    last_sample = x
  end

  throw(ErrorException("failed to sample a feasible constrained solution after $(feasible_sample_retries) attempts; last sample was $(last_sample)"))
end


# Strict upper triangular part of an array
function upper_indices(a)
  return (Tuple(ci)
            for ci in CartesianIndices(size(a))
            if issorted(Tuple(ci); lt = (<=)))
end

function constant(p::AbstractPolynomial{T}) where T
  ts  = terms(p)
  idx = findfirst(isconstant, ts)

  return isnothing(idx) ? zero(T) : coefficient(ts[idx])
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

function tensorize(Q::AbstractArray{T}, rest::Vararg{AbstractArray{T}}; cutoff = zero(T)) where T
  Qs = [Q, rest...]
  if !allequal(Iterators.flatmap(size, Qs))
    throw(DimensionMismatch("All arrays should act on the same number of variables.\nEncountered dimensions $(collect(map(size, Qs)))."))
  end

  N = size(Q, 1)
  sites = ITensors.siteinds("Qudit", N; dim = 2)
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

function tensorize(p::AbstractPolynomial{T}; cutoff = zero(T)) where T
  N = length(effective_variables(p))
  sites = ITensors.siteinds("Qudit", N; dim = 2)
  os = OpSum{T}()

  # Map: var name => index
  indices = Dict(v => i for (i, v) in enumerate(effective_variables(p)))

  for t in terms(p)
    coeff = coefficient(t)

    if abs(coeff) > cutoff && ! isconstant(t)
      vars = effective_variables(t)
      op   = Iterators.flatmap(v -> ("D", indices[v]), vars)
      os .+= (coeff, op...)
    end
  end

  return isempty(os) ? MPO(T,sites) : MPO(T, os, sites)
end


function single_variable_minimize(
  ::Type{T},
  sites,
  obj,
  initial_time;
  device,
  verbosity,
  on_iteration,
  callback_every,
  sample_validator=nothing,
) where {T}
  feasible_samples = [
    x for x in ([0], [1])
    if isnothing(sample_validator) || sample_validator(x)
  ]

  if isempty(feasible_samples)
    throw(ArgumentError("constraints define an empty feasible subspace; no binary vector satisfies all constraints"))
  end

  feasible_values = [obj(x) for x in feasible_samples]

  energy, state = if length(feasible_samples) == 2 && feasible_values[1] == feasible_values[2]
    (T(feasible_values[1]), ["full"])
  else
    idx = argmin(feasible_values)
    (T(feasible_values[idx]), string.(feasible_samples[idx]))
  end

  psi = ITensorMPS.MPS(sites, state) |> device
  elapsed_time = time() - initial_time
  bond_dim = ITensorMPS.maxlinkdim(psi)

  iterlog_header(verbosity)
  iterlog_iteration(verbosity, 1, energy, bond_dim, 0.0, elapsed_time)

  dist = Solution{T}(psi, T[energy], Int[bond_dim], Float64[elapsed_time])
  if !isnothing(on_iteration) && 1 % callback_every == 0
    on_iteration(psi; iteration=1, objective=energy, bond_dim, elapsed_time)
  end

  iterlog_footer(verbosity, energy, elapsed_time)

  return energy, dist
end

function minimize_mpo( H :: MPO
                     , c :: T
                     , obj
                     ; device     :: Function = cpu
                     , cutoff     = 1e-8  #  a cutoff of 1E-5 gives sensible accuracy; a cutoff of 1E-8 is high accuracy; and a cutoff of 1E-12 is near exact accuracy. (https://itensor.org/docs.cgi?page=tutorials/dmrg_params)
                     , verbosity  = 1
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
                     , on_iteration :: Union{Nothing, Function} = nothing
                     , callback_every   :: Int = 1
                     , permutation :: Vector{Int} = Int[]
                     , initial_state :: Union{Nothing, MPS} = nothing
                     , solution_projection = nothing
                     , sample_validator = nothing
                     , feasible_sample_retries :: Integer = 100
                     ) where {T}
  callback_every >= 1 || throw(ArgumentError("`callback_every` must be >= 1, got $callback_every"))
  feasible_sample_retries >= 1 ||
    throw(ArgumentError("`feasible_sample_retries` must be >= 1, got $feasible_sample_retries"))
  initial_time      = time()
  energies_log      = T[]
  bond_dims_log     = Int[]
  elapsed_times_log = Float64[]

  # Quantization
  sites = ITensorMPS.siteinds(first, H; plev=0)
  solution_permutation = isempty(permutation) ? collect(1:length(sites)) : permutation
  solution_projections = isnothing(solution_projection) ? nothing : map(device, projection_sequence(solution_projection))

  if length(sites) == 1
    return single_variable_minimize(
      T,
      sites,
      obj,
      initial_time;
      device,
      verbosity,
      on_iteration,
      callback_every,
      sample_validator,
    )
  end

  H     = device(H)
  psi   = isnothing(initial_state) ? ITensorMPS.random_mps(T, sites; linkdims=inidim) : initial_state
  psi   = device(psi)

  @debug("MPO construction finished",
    time=(time() - initial_time),
    max_bond = ITensorMPS.maxlinkdim(H),
    num_coefficients = sum(prod(m) for m in H),
  )

  iterlog_header(verbosity)
  var = Inf
  local energy, psi

  for i in Iterators.countfrom(1)
    energy, psi = dmrg(H, device(psi)
                      ; nsweeps     = 1
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

    if !isnothing(solution_projections)
      psi = project_feasible_state(
        psi,
        solution_projections;
        cutoff,
        error_type=ErrorException,
        message="constrained DMRG produced a state with zero feasible amplitude; constraints may be infeasible or the projection cutoff may be too large",
      )
      energy = real(expectation(H, psi))
    end

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
      @debug "Stopping: variance below tolerance" variance=var tolerance=vtol
      break
    end
  end

  # The calculated energy has approximation errors compared to the true solution.
  # It makes more sense to sample a solution and calculate the true objective function applied to it.
  dist = Solution{T}(psi, energies_log, bond_dims_log, elapsed_times_log, solution_permutation)
  optimal = sampled_objective(
    dist,
    obj;
    sample_validator,
    feasible_sample_retries,
  )
  elapsed_time = time() - initial_time

  iterlog_footer(verbosity, optimal, elapsed_time)

  return optimal, dist
end
