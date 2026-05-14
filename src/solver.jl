using LinearAlgebra
import Combinatorics: multiset_permutations

import ITensors: inner
import ITensorMPS: MPS, MPO, dmrg, OpSum, @OpName_str, @SiteType_str, @StateName_str, dim

import ITensors, ITensorMPS

import MultivariatePolynomials: AbstractPolynomial, coefficient, monomial, terms, variables, effective_variables, isconstant

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

maybe(f::Function, mx::Nothing; default=nothing) = default
maybe(f::Function, mx; default=nothing) = f(mx)

expectation(H, x) = inner(x', H, x)
variance(H::MPO, x::MPS) = real(inner(H, x, H, x) - expectation(H, x)^2)

# Diagonal matrix whose eigenvalues are the ordered feasible values for an integer variable.
# For qubits, this is a projection on |1>. Or equivalently, (I - σ_z) / 2.
# This looks like type piracy,
# but is, in fact, ITensors' way to extend the OpSum mechanism.
ITensors.op(::OpName"D",::SiteType"Qudit", d::Int) = diagm(0:(d-1))

ITensors.state(::StateName"full", ::SiteType"Qudit", s::ITensorMPS.Index) = (d = dim(s); fill(1/sqrt(d), d))

"""
    tensorize(p)

Turn a polynomial function action on bitstrings
into an equivalent MPO Hamiltonian acting on Qudit sites.
The conversion consists of exchanging each integer variable `x_i`
for a matrix `P_i` whose eigenvalues represent its feasible set `K_i`.

    ∑ Q_ij x_i x_j + ∑ l_i x_i --> H = Σ Q_ij D_i D_j + ∑ l_i D_i
"""
function tensorize end

function tensorize(model::PseudoBooleanModel{T}; cutoff = zero(T)) where T
  N = length(model)
  sites = ITensors.siteinds("Qudit", N; dim = 2)
  os = OpSum{T}()

  for (vars, coeff) in model.terms
    if abs(coeff) > cutoff
      op = Iterators.flatmap(v -> ("D", v), vars)
      os .+= (coeff, op...)
    end
  end

  return isempty(os) ? MPO(T, sites) : MPO(T, os, sites)
end

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
- `on_iteration :: Function` - Called after each recorded iteration as
  `f(psi::MPS; iteration, objective, bond_dim, elapsed_time)`.
  `objective` is the expected objective function ⟨ψ|H|ψ⟩ at this iteration.
  Use to collect statistics or serialize intermediate states.
  `psi` is the MPS for that iteration.
  Default: `nothing` (no callback).
- `callback_every :: Int` - Invoke the callback every N iterations. Must be >= 1. Default: `1`.

The returned `Solution` carries per-iteration stats in `.energies`, `.bond_dims`, and `.elapsed_times`.

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
function minimize end

function minimize(model::PseudoBooleanModel; backend=DMRGBackend(), cutoff=1e-8, kwargs...)
  selected_backend = backend isa AbstractBackend ? backend : backend_from_attribute(backend)
  return _solve_backend(selected_backend, model; cutoff, kwargs...)
end

function _solve_backend(::DMRGBackend, model::PseudoBooleanModel{T}; cutoff=1e-8, kwargs...) where {T}
  if length(model) == 1
    return _minimize_single_variable(model; cutoff, kwargs...)
  end

  H = tensorize(model; cutoff)
  return _minimize(H, model.constant, x -> evaluate(model, x); cutoff, kwargs...)
end

function minimize(Q :: AbstractMatrix{T} , l :: Union{AbstractVector{T}, Nothing} = nothing , c :: T = zero(T); cutoff=1e-8, backend=DMRGBackend(), kwargs...) where T
  model = pseudoboolean(Q, l, c; cutoff=zero(T))
  return minimize(model; backend, cutoff, kwargs...)
end

function minimize(p::AbstractPolynomial{T}; cutoff=1e-8, backend=DMRGBackend(), kwargs...) where T
  model = pseudoboolean(p; cutoff=zero(T))
  return minimize(model; backend, cutoff, kwargs...)
end

function _check_no_unknown_solver_kwargs(kwargs)
  if !isempty(kwargs)
    unknown = join(("`$key`" for key in keys(kwargs)), ", ")
    throw(ArgumentError("Unsupported solver keyword(s): $unknown"))
  end
end

function _minimize_single_variable(
  model::PseudoBooleanModel{T};
  cutoff = 1e-8,
  device :: Function = cpu,
  verbosity = 1,
  on_iteration :: Union{Nothing, Function} = nothing,
  callback_every :: Int = 1,
  # Accepted for API compatibility with the DMRG path. This exact path does not iterate.
  iterations :: Union{Nothing, Int} = nothing,
  time_limit = +Inf,
  vtol = cutoff,
  check_variance_every_iteration = 10,
  inidim = 40,
  maxdim = [10, 10, 10, 20, 50, 100, 100, 200, 300, 300, 400, 400, 800, 900, 1000],
  mindim = 1,
  noise = [1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 0.0],
  eigsolve_krylovdim :: Int = 3,
  eigsolve_maxiter :: Int = 2,
  eigsolve_tol :: Float64 = 1e-14,
  kwargs...
) where {T}
  callback_every >= 1 || throw(ArgumentError("`callback_every` must be >= 1, got $callback_every"))
  _check_no_unknown_solver_kwargs(kwargs)

  initial_time = time()
  objective_zero = evaluate(model, [0])
  objective_one = evaluate(model, [1])
  optimal = min(objective_zero, objective_one)

  state = if abs(objective_zero - objective_one) <= cutoff
    "full"
  elseif objective_zero < objective_one
    "0"
  else
    "1"
  end

  sites = ITensors.siteinds("Qudit", 1; dim = 2)
  psi = MPS(sites, [state]) |> device
  elapsed_time = time() - initial_time
  solution = Solution{T}(psi, T[optimal], Int[1], Float64[elapsed_time])

  iterlog_header(verbosity)
  iterlog_iteration(verbosity, 1, optimal, 1, nothing, elapsed_time)
  iterlog_footer(verbosity, optimal, elapsed_time)

  if !isnothing(on_iteration)
    on_iteration(psi; iteration=1, objective=optimal, bond_dim=1, elapsed_time)
  end

  return optimal, solution
end

function _minimize( H :: MPO
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
                  ) where {T}
  callback_every >= 1 || throw(ArgumentError("`callback_every` must be >= 1, got $callback_every"))
  initial_time      = time()
  energies_log      = T[]
  bond_dims_log     = Int[]
  elapsed_times_log = Float64[]

  # Quantization
  sites = ITensorMPS.siteinds(first, H; plev=0)
  H     = device(H)
  psi   = ITensorMPS.random_mps(T, sites; linkdims=inidim) |> device

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
  dist = Solution{T}(psi, energies_log, bond_dims_log, elapsed_times_log)
  x    = sample(dist)
  optimal = obj(x)
  elapsed_time = time() - initial_time

  iterlog_footer(verbosity, optimal, elapsed_time)

  return optimal, dist
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

  return -E, _with_objective(psi, -E)
end
