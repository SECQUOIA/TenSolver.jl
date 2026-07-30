"""
    AbstractConstraint

Supertype for conditions over a vector `x` addressed by 1-based site indices.


# API

These are an interface for feasibility constraints.
Any concrete subtype is expected to implement
- [`is_feasible`](@ref)
- [`constraint_sites`](@ref)
- [`constraint_to_dfa`](@ref)

Constraint types are experimental. They currently provide TenSolver's
Julia lowering target for projection-MPO constrained solves; future JuMP/MOI
integration may change which constraint abstraction is considered stable public API.

See also [`SumConstraint`](@ref), [`SumModConstraint`](@ref), [`NotEqualsConstraint`](@ref),
[`AssignmentConstraint`](@ref), and [`RelationConstraint`](@ref).
"""
abstract type AbstractConstraint end

"""
    SumConstraint{T} <: AbstractConstraint
    SumConstraint(sites, weights, relation, rhs)
    SumConstraint(sites, weights, rhs; relation)

Weighted-sum constraint over a vector `x`:

    sum(weights[i] * x[sites[i]] for i in eachindex(sites)) relation rhs.

`sites` must be unique positive integers,
`weights` must be the same length as `sites` and only contain nonnegative integers,
and `relation` must be one of `:(==)`, `:(!=)`, `:(<=)`, or `:(>=)`.

Warning: The `==` and `!=` relations use exact arithmetic comparison.
"""
struct SumConstraint{T<:Real} <: AbstractConstraint
  weights::Dict{Int,T}
  relation::Symbol
  rhs::T

  function SumConstraint{T}(sites, weights, relation, rhs::T) where {T<:Real}
    site_vec   = validate_sites(sites)
    weight_vec = validate_weights(weights)
    validate_same_length(site_vec, weight_vec, "sites", "weights")
    relation   = validate_relation(relation)
    rhs        = validate_rhs(rhs)

    weight_map = Dict{Int,T}(zip(site_vec, weight_vec))

    return new{T}(weight_map, relation, rhs)
  end
end

function SumConstraint(sites, weights, relation, rhs)
  raw_weights = collect(weights)
  isempty(raw_weights) && throw(ArgumentError("weights must not be empty"))

  weight_types = map(typeof, raw_weights)
  T = promote_type(weight_types..., typeof(rhs))

  weight_vec = T.(raw_weights)
  rhs_value = convert(T, rhs)

  # `sites` and `relation` are validated once, inside the inner constructor.
  return SumConstraint{T}(sites, weight_vec, relation, rhs_value)
end

function SumConstraint(sites, weights, rhs; relation)
  return SumConstraint(sites, weights, relation, rhs)
end

"""
    SumModConstraint{T} <: AbstractConstraint
    SumModConstraint(sites, weights, rhs; mod)

Modular weighted-sum constraint over a vector `x`:

    ( sum(weights' * x[sites]) == rhs ) mod m.

`sites` must be unique positive integers,
`weights` must be the same length as `sites`.
"""
struct SumModConstraint{T<:Real} <: AbstractConstraint
  weights::Dict{Int,T}
  rhs::T
  mod::T

  function SumModConstraint{T}(sites, weights, rhs; mod) where {T<:Real}
    site_vec   = validate_sites(sites)
    weight_vec = validate_weights(map(w -> w % mod, weights))
    rhs        = validate_rhs(rhs % mod)
    validate_same_length(site_vec, weight_vec, "sites", "weights")

    weight_map = Dict{Int,T}(zip(site_vec, weight_vec))

    return new{T}(weight_map, rhs, mod)
  end
end

function SumModConstraint(sites, weights, rhs; mod)
  return SumModConstraint{typeof(mod)}(sites, weights, rhs; mod)
end

"""
    NotEqualsConstraint <: AbstractConstraint
    NotEqualsConstraint(sites, values)

Excludes a single assignment over a vector `x`:
at least one component of `x[sites]` must differ from `values`.
Equivalently, the partial assignment `x[sites] == values` is forbidden.

`sites` must be unique positive integers, and `values` must have the same length as `sites`.
"""
struct NotEqualsConstraint{T<:Real} <: AbstractConstraint
  values::Dict{Int, T}

  function NotEqualsConstraint{T}(sites, values::AbstractVector{T}) where {T<:Real}
    site_vec = validate_sites(sites)
    validate_same_length(site_vec, values, "sites", "values")

    value_map = Dict{Int,T}(zip(site_vec, values))

    return new{T}(value_map)
  end
end

function NotEqualsConstraint(sites, values)
  isempty(values) && throw(ArgumentError("values must not be empty"))
  T = promote_type(map(typeof, values)...)

  return NotEqualsConstraint{T}(sites, T.(values))
end

"""
    AssignmentConstraint{T} <: AbstractConstraint
    AssignmentConstraint(sites, values, relation, rhs)

Restrict how many `sites` satisfy `x[site] in values`, i.e.,

    count(x[site] in values for site in sites) relation rhs.

`rhs` must be a nonnegative integer. The count and `rhs` are stored
independently of the numeric element type of `values`.

A common application is to restrict _exactly one_ variable to be a certain value,

```julia
ExactlyOne(sites, value) = AssignmentConstraint(sites, [value], :(==), 1)
```
"""
struct AssignmentConstraint{T<:Real} <: AbstractConstraint
  sites    :: Vector{Int}
  values   :: Set{T}
  relation :: Symbol
  rhs      :: Int

  function AssignmentConstraint{T}(sites, values, relation, rhs) where {T<:Real}
    site_vec = validate_sites(sites)
    relation = validate_relation(relation)
    rhs      = Int(validate_rhs(rhs))

    return new{T}(site_vec, Set(values), relation, rhs)
  end
end

function AssignmentConstraint(sites, values, relation, rhs)
  return AssignmentConstraint{eltype(values)}(sites, values, relation, rhs)
end

"""
    RelationConstraint <: AbstractConstraint
    RelationConstraint(left_site, relation, right_site)

Pairwise constraint over a vector `x`:
`x[left_site] relation x[right_site]`.

`left_site` and `right_site` must be distinct positive integers, and `relation`
must be one of `:(==)`, `:(!=)`, `:(<=)`, or `:(>=)`.
"""
struct RelationConstraint <: AbstractConstraint
  left_site::Int
  relation::Symbol
  right_site::Int

  function RelationConstraint(left_site, relation, right_site)
    left  = validate_site(left_site, "left_site")
    right = validate_site(right_site, "right_site")
    left == right && throw(ArgumentError("relation constraint sites must be distinct"))

    return new(left, validate_relation(relation), right)
  end
end

"""
    is_feasible(x, constraint::AbstractConstraint)

Test whether the vector `x` satisfies a single `constraint`.
"""
function is_feasible end

function is_feasible(x::AbstractVector, constraint::SumConstraint)
  lhs = sum(weight * x[site] for (site, weight) in constraint.weights)
  return relation_holds(lhs, constraint.relation, constraint.rhs)
end

function is_feasible(x::AbstractVector, constraint::SumModConstraint)
  lhs = sum(weight * x[site] for (site, weight) in constraint.weights) % constraint.mod
  return lhs == constraint.rhs
end

function is_feasible(x::AbstractVector, constraint::NotEqualsConstraint)
  return any(x[site] != value for (site, value) in constraint.values)
end

function is_feasible(x::AbstractVector, constraint::AssignmentConstraint)
  lhs = count(site -> x[site] in constraint.values, constraint.sites)
  return relation_holds(lhs, constraint.relation, constraint.rhs)
end

function is_feasible(x::AbstractVector, constraint::RelationConstraint)
  lhs = x[constraint.left_site]
  rhs = x[constraint.right_site]

  return relation_holds(lhs, constraint.relation, rhs)
end

"""
    is_feasible(x, constraints::Vector{AbstractConstraint})

Test whether the vector `x` satisfies every constraint in a vector `constraints`.

Any vector is feasible for an empty constraint vector.
"""
function is_feasible(x::AbstractVector, constraints::AbstractVector{<:AbstractConstraint})
  return all(c -> is_feasible(x, c), constraints)
end

"""
    constraint_sites(constraint::AbstractConstraint)

Access the site indices stored in the `constraint`.
"""
function constraint_sites end

function constraint_sites(constraint::SumConstraint)
  return keys(constraint.weights)
end

function constraint_sites(constraint::SumModConstraint)
  return keys(constraint.weights)
end

function constraint_sites(constraint::NotEqualsConstraint)
  return keys(constraint.values)
end

function constraint_sites(constraint::AssignmentConstraint)
  return constraint.sites
end

function constraint_sites(constraint::RelationConstraint)
  return [constraint.left_site, constraint.right_site]
end

#----------------------------------------------------------#
# Constraint Validation
#----------------------------------------------------------#

const VALID_RELATIONS = (
  Symbol("=="),
  Symbol("!="),
  Symbol("<="),
  Symbol(">="),
)

function validate_site(site, name)
  site isa Integer || throw(ArgumentError("$name must be an integer"))
  site > 0 || throw(ArgumentError("$name must be a positive integer"))

  return Int(site)
end

function validate_sites(sites)
  if isempty(sites)
    throw(ArgumentError("sites must not be empty"))
  end

  validated = [validate_site(site, "sites") for site in sites]
  if !allunique(validated)
    throw(ArgumentError("sites must be unique"))
  end

  return validated
end

function validate_same_length(left, right, left_name, right_name)
  if length(left) != length(right)
    throw(DimensionMismatch("$left_name and $right_name must have the same length"))
  end
end

function validate_weights(weights)
  if isempty(weights)
    throw(ArgumentError("weights must not be empty"))
  end
  # Nonnegativity is a deliberate v1 contract (issue #56 acceptance criteria):
  # it keeps the predicate aligned with the nonnegative projection targets used
  # by the constraint/MPO work tracked in #57. Signed weights (e.g. encoding a
  # difference `x1 - x2 == 0`) are intentionally out of scope here and should be
  # revisited together with that lowering, not relaxed in isolation.
  for (i, weight) in enumerate(weights)
    if weight < 0
      throw(ArgumentError("Found negative weight w[$(i)] = $(repr(weight)). Weights must be nonnegative."))
    end
    if !isinteger(weight)
      throw(ArgumentError("Found noninteger weight w[$(i)] = $(repr(weight)). Weights must be integer."))
    end
  end

  return weights
end

function validate_rhs(rhs)
  if rhs < 0
    throw(ArgumentError("Found negative rhs = $(repr(rhs)). rhs must be nonnegative."))
  end
  if !isinteger(rhs)
    throw(ArgumentError("Found noninteger rhs = $(repr(rhs)). rhs must be integer."))
  end

  return rhs
end

function validate_relation(relation)
  relation in VALID_RELATIONS ||
    throw(ArgumentError("relation must be one of: $(join(string.(VALID_RELATIONS), ", "))"))

  return relation
end

function relation_holds(lhs, relation, rhs)
  relation === Symbol("==") && return lhs == rhs
  relation === Symbol("!=") && return lhs != rhs
  relation === Symbol("<=") && return lhs <= rhs
  relation === Symbol(">=") && return lhs >= rhs

  error("unsupported relation: $relation")
end
