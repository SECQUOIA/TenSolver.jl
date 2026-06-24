"""
    DMRGBackend()

Select TenSolver's default ITensorMPS DMRG backend.
"""
struct DMRGBackend <: AbstractTenSolverBackend end

normalize_backend(::Val{:dmrg}) = default_backend

function minimize(::DMRGBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); cutoff=1e-8, preprocess::Bool=false, kwargs...) where T
  Qp, lp, permutation = preprocess ? preprocess_qubo(Q, l, cutoff) : (Q, l, collect(1:size(Q, 1)))
  H      = tensorize(Qp, isnothing(lp) ? diag(Qp) : diag(Qp) + lp; cutoff)
  obj(x) = dot(x, Q, x) + c + maybe(l -> dot(l,x), l; default=zero(T))

  return minimize_mpo(H, c, obj; cutoff, permutation, kwargs...)
end

function minimize(::DMRGBackend, p::AbstractPolynomial{T}; cutoff=1e-8, kwargs...) where T
  H   = tensorize(p)
  cte = constant(p)
  vs = variables(p)
  return minimize_mpo(H, cte, a -> p(vs => a); cutoff, kwargs...)
end

function single_variable_minimize(::Type{T}, sites, obj, initial_time; device, verbosity, on_iteration, callback_every) where {T}
  x0 = [0]
  x1 = [1]
  e0 = obj(x0)
  e1 = obj(x1)

  energy, state = if e0 == e1
    (T(e0), ["full"])
  elseif e0 < e1
    (T(e0), string.(x0))
  else
    (T(e1), string.(x1))
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
                     ) where {T}
  callback_every >= 1 || throw(ArgumentError("`callback_every` must be >= 1, got $callback_every"))
  initial_time      = time()
  energies_log      = T[]
  bond_dims_log     = Int[]
  elapsed_times_log = Float64[]

  # Quantization
  sites = ITensorMPS.siteinds(first, H; plev=0)
  solution_permutation = isempty(permutation) ? collect(1:length(sites)) : permutation

  if length(sites) == 1
    return single_variable_minimize(T, sites, obj, initial_time; device, verbosity, on_iteration, callback_every)
  end

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
  dist = Solution{T}(psi, energies_log, bond_dims_log, elapsed_times_log, solution_permutation)
  x    = sample(dist)
  optimal = obj(x)
  elapsed_time = time() - initial_time

  iterlog_footer(verbosity, optimal, elapsed_time)

  return optimal, dist
end
