using LinearAlgebra
import Combinatorics: multiset_permutations

import MultivariatePolynomials: AbstractPolynomial, coefficient, monomial, terms, variables, effective_variables, isconstant

maybe(f::Function, mx::Nothing; default=nothing) = default
maybe(f::Function, mx; default=nothing) = f(mx)

ifnotnothing(a, b) = maybe(_ -> b, a)

"""
    AbstractTenSolverBackend

Abstract solver backend marker for TenSolver implementations.

Backends must provide backend-specific `minimize` methods for the normalized
optimization inputs they support. Matrix backends implement
`minimize(::MyBackend, Q::AbstractMatrix, l, c; kwargs...)`; polynomial
backends implement `minimize(::MyBackend, p::AbstractPolynomial; kwargs...)`.
Extensions that support symbolic selection must also define
`normalize_backend(::Val{:my_backend}) = MyBackend(...)`.

The default implementation is [`DMRGBackend`](@ref).

# See also
[`DMRGBackend`](@ref), [`normalize_backend`](@ref).
"""
abstract type AbstractTenSolverBackend end

"""
    normalize_backend(backend)

Normalize a user-facing backend selector into a backend object.

Backends can support `backend = :my_backend` by defining
`normalize_backend(::Val{:my_backend}) = MyBackend(...)`.
"""
function normalize_backend end

function backend_error(backend)
  if backend === :peps
    return ArgumentError("backend :peps is not available. Install/load the PEPS extension or use backend = :dmrg.")
  end

  return ArgumentError("No backend-specific `minimize` method is available for backend $(repr(backend)). Use backend = :dmrg or provide a backend-specific `minimize` method.")
end

normalize_backend(backend::AbstractTenSolverBackend) = backend
normalize_backend(backend::Symbol) = normalize_backend(Val(backend))
function normalize_backend(::Val{backend}) where {backend}
  throw(backend_error(backend))
end
normalize_backend(backend) = throw(backend_error(backend))


#
# Backends
#
include("backends/dmrg.jl")
include("backends/peps.jl")
const default_backend = DMRGBackend()

#=======================================================================#
# Minimization and Maximization                                         #
#=======================================================================#

"""
    minimize([Q::Matrix], [l::Vector], [c::Number] ; domain, kwargs...)
    minimize(p::AbstractPolynomial ; domain, kwargs...)

Solve a polynomial discrete optimization problem

    min  p(x)
    s.t. x_i in domain
         constraints

In the matrix version, the objective is limited to quadratic forms x -> x'Qx + l'x + c.
Missing arguments (quadratic, linear or constant term)
are allowed and taken to be zero.

Return the optimal value `E` and a probability distribution `ψ` over optimal solutions.
You can use [`sample`](@ref) to get an actual solution vector from `ψ`.

There are multiple backends available, selected through the keyword `backend`.
By default, it uses DMRG to calculate the optimal solution.


Keyword arguments:

- `constraints :: AbstractVector{<:AbstractConstraint}` - Experimental native Julia hard constraints.
  Defaults to `AbstractConstraint[]`. In constrained DMRG solves, TenSolver lowers each constraint to
  a projection MPO, solves the projected Hamiltonian, and returns a feasible sampled assignment.
  For polynomial objectives, constraints are expressed in the same order as their `effective_variables`.
  If the constraints admit no solution at all, the solve does not error: it logs a warning and
  returns `+Inf` together with an infeasible [`Solution`](@ref) (see [`is_feasible`](@ref)).
- `domain` - Possible variable values. Defaults to `[0, 1]`.
  Unconstrained DMRG optimization accepts any finite collection of real values;
  individual constraint types can impose narrower requirements. Use `[-1, 1]`
  for Ising spins. Domains are sorted and deduplicated before solving.
- `iterations :: Int` - Maximum iterations the solver should run. Defaults to `10`.
- `cutoff :: Float64` - Any absolute value below this threshold is considered zero. Defaults to `1e-8`.
  You can use this keyword to control the solver's accuracy vs resources trade-off.
- `time_limit :: Float64` - If specified, determines the maximum running time in seconds.
  It only determines whether a new iteration should start or not, thus the solver may run for longer if the threshold happens during an iteration.
- `device = cpu` - Accelerator device used during computation.
  See the section below for how to run on GPUs.
- `preprocess :: Bool` - Defaults to `false`. If `true`, permute QUBO variables before constructing the MPS Hamiltonian
  so coupled variables are closer in the one-dimensional tensor order. Samples are returned in the
  caller's original variable order. This is an experimental feature and may be subject to changes.
- `on_iteration :: Function` - Called after each recorded iteration as
  `f(psi::MPS; iteration, objective, bond_dim, elapsed_time)`.
  `objective` is the expected objective function ⟨ψ|H|ψ⟩ at this iteration.
  Use to collect statistics or serialize intermediate states.
  `psi` is the MPS for that iteration.
  Default: `nothing` (no callback).
- `callback_every :: Int` - Invoke the callback every N iterations. Must be >= 1. Default: `1`.
- `backend` - Solver backend. Defaults to the current DMRG implementation.
  Use `backend = :dmrg` or `backend = DMRGBackend()` to select it explicitly.
  Other backends are reserved for optional extensions.

  Other keywords might be available depending on the chosen backend.
  See the documentation for each backend for comprehensive lists.

  Some keywords, such as `constraints` and `domain`,
  may have limited support depending on the backend.

The [`DMRGSolution`](@ref) returned by the default backend carries
per-iteration convergence data in `solution.stats`. Its former top-level
fields `solution.energies`, `solution.bond_dims`, and
`solution.elapsed_times` remain available as deprecated aliases.

Provably infeasible constrained models are reported as a status:
`minimize` logs a warning and returns `+Inf` (the minimum over an empty feasible set)
together with an infeasible [`Solution`](@ref), which cannot be sampled.
Check it with [`is_feasible`](@ref).

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

function minimize(backend::AbstractTenSolverBackend, args...; kwargs...)
  throw(backend_error(backend))
end

function minimize(
  p::AbstractPolynomial{T}
  ;
  backend=default_backend,
  domain::AbstractVector = 0:1,
  kwargs...,
) where T
  domain = validate_solve_domain(domain)
  p      = simplify_polynomial(p, domain)
  return minimize(normalize_backend(backend), p; domain, kwargs...)
end

function minimize(
  Q :: AbstractMatrix{T},
  l :: AbstractVector{T} = zeros(T, size(Q, 1)),
  c :: T = zero(T)
  ;
  backend = default_backend,
  domain::AbstractVector = 0:1,
  kwargs...,
) where {T<:Real}
  domain  = validate_solve_domain(domain)
  Q, l, c = simplify_polynomial(Q, l, c, domain)
  return minimize(normalize_backend(backend), Q, l, c; domain, kwargs...)
end

function minimize(l :: AbstractVector{T}, c :: T = zero(T); kwargs...) where {T<:Real}
  return minimize(zeros(T, size(l, 1), size(l, 1)), l, c; kwargs...)
end

function minimize(Q :: AbstractMatrix{T}, c :: T; kwargs...) where {T<:Real}
  return minimize(Q, zeros(T, size(Q, 1)), c; kwargs...)
end

"""
    maximize(Q::Matrix[, l::Vector[, c::Number; kwargs...)
    maximize(p::AbstractPolynomial; kwargs...)

Solve the Quadratic Unconstrained Binary Optimization problem
for maximization.

    max  b'Qb + l'b + c
    s.t. b_i in {0, 1}

All keywords accepted by [`minimize`](@ref) can also be used for maximization problems.
Provably infeasible constrained models return `-Inf` (the supremum over an
empty feasible set) together with an infeasible [`Solution`](@ref).

See also [`minimize`](@ref).
"""
function maximize(qs... ; kwargs...)
  # Flip the sign of all non-nothing elements
  # max p(x) = - min -p(x)
  mqs = map(q -> maybe(-, q), qs)
  E, psi = minimize(mqs...; kwargs...)

  return -E, psi
end

#=====================================================================#
# Domain validation                                                   #
#=====================================================================#

function validate_solve_domain(domain)
  # Preprocessing to dedeplicate domain values
  domain = (ismutable(domain) ? unique! : unique)(sort(domain))

  if !applicable(iterate, domain)
    throw(ArgumentError("`domain` must be an iterable collection of values."))
  elseif !applicable(length, domain)
    throw(ArgumentError("`domain` must have a finite length."))
  elseif isempty(domain)
    throw(ArgumentError("`domain` must contain at least one value."))
  elseif !all(u -> u isa Real, domain)
    throw(ArgumentError("`domain` values must be values of a real type."))
  elseif !allunique(domain)
    throw(ArgumentError("`domain` values must be unique."))
  end

  return domain
end

function simplify_polynomial(p::AbstractPolynomial, domain)
  # A finite domain xi in U = {u1, ..., ud} is equivalent
  # to the root set of a single variable polynomial
  # q(x) = (xi - u1)...(xi - ud)
  rooted(x) = prod(x - a for a in domain)
  # By dividing p // q, we get
  # p(x) = m(x)q(x) + r(x).
  # Notice that for any a in U, q(a) = 0, and
  # p(a) = m(a)*0 + r(a) = r(a).
  # Thus, we transform p -> r as a degree reduction procedure.
  return mapfoldl(rooted, rem, effective_variables(p); init = p)
end

function simplify_polynomial(Q::AbstractMatrix, l, c, domain)
  # A variable x in {a, b} satifies
  #   (x - a)(x - b) = 0
  #   x^2 = (a + b)x - ab
  #   Thus, we exchange the diagonal terms x^2 by linear and constant terms.
  if length(domain) == 2
    s, p = sum(domain), prod(domain)

    l = l .+ s .* diag(Q)
    c = c  - p  * sum(diag(Q))
    Q = Q .- Diagonal(view(Q, diagind(Q)))
  end

  return Q, l, c
end
