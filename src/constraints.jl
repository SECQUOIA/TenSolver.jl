"""
    AbstractConstraint

Supertype for conditions over a binary vector `x` (entries in `{0, 1}`)
addressed by 1-based site indices.


# API

These are an interface for binary feasibility constraints.
Any concrete subtype is expected to implement
- [`is_feasible`](@ref).
- [`constraint_sites`](@ref).

See also [`SumConstraint`](@ref), [`NotEqualsConstraint`](@ref),
[`ExactlyOneConstraint`](@ref), and [`RelationConstraint`](@ref).
"""
abstract type AbstractConstraint end

"""
    SumConstraint{T} <: AbstractConstraint
    SumConstraint(sites, weights, relation, rhs)
    SumConstraint(sites, weights, rhs; relation)

Weighted-sum constraint over a binary vector `x`:

    sum(weights[i] * x[sites[i]] for i in eachindex(sites)) relation rhs.

`sites` must be unique positive integers,
`weights` must be the same length as `sites` and only contain nonnegative integers,
and `relation` must be one of `:(==)`, `:(!=)`, `:(<=)`, or `:(>=)`.

The `==` and `!=` relations use exact arithmetic comparison. Projection MPO
lowering for `SumConstraint` supports integer-valued weights and right-hand side
values.
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

    weight_map = Dict(site => weight for (site, weight) in zip(site_vec, weight_vec))
    if length(weight_map) != length(site_vec)
      throw(ArgumentError("sites must be unique"))
    end

    return new{T}(weight_map, relation, rhs)
  end
end

function SumConstraint(sites, weights, relation, rhs)
  raw_weights = collect(weights)
  isempty(raw_weights) && throw(ArgumentError("weights must not be empty"))

  weight_types = map(typeof, raw_weights)
  T = promote_type(weight_types..., typeof(rhs))
  T <: Real || throw(ArgumentError("weights and rhs must be real-valued"))

  weight_vec = T.(raw_weights)
  rhs_value = convert(T, rhs)

  # `sites` and `relation` are validated once, inside the inner constructor.
  return SumConstraint{T}(sites, weight_vec, relation, rhs_value)
end

function SumConstraint(sites, weights, rhs; relation)
  return SumConstraint(sites, weights, relation, rhs)
end

"""
    NotEqualsConstraint <: AbstractConstraint
    NotEqualsConstraint(sites, values)

Excludes a single assignment over a binary vector `x`: at least one component of
`x[sites]` must differ from `values`. Equivalently, the partial assignment
`x[sites] == values` is forbidden.

`sites` must be unique positive integers, and `values` must contain only binary
values (`0` or `1`) with the same length as `sites`.
"""
struct NotEqualsConstraint{T<:Real} <: AbstractConstraint
  values::Dict{Int, T}

  function NotEqualsConstraint{T}(sites, values::AbstractVector{T}) where {T<:Real}
    site_vec  = validate_sites(sites)
    value_vec = validate_binary_values(values, "values")
    validate_same_length(site_vec, value_vec, "sites", "values")

    value_map = Dict{Int,T}(site => value for (site, value) in zip(site_vec, value_vec))
    if length(value_map) != length(site_vec)
      throw(ArgumentError("sites must be unique"))
    end

    return new{T}(value_map)
  end
end

function NotEqualsConstraint(sites, values)
  raw_values = collect(values)
  isempty(raw_values) && throw(ArgumentError("values must not be empty"))

  T = promote_type(map(typeof, raw_values)...)
  T <: Real || throw(ArgumentError("values must be real-valued"))

  return NotEqualsConstraint{T}(sites, T.(raw_values))
end

"""
    ExactlyOneConstraint <: AbstractConstraint
    ExactlyOneConstraint(sites)
    ExactlyOneConstraint(sites, value)

Requires exactly one constrained entry to equal the target binary `value`:

    count(x[site] == value for site in sites) == 1.

`value` defaults to `1`. `sites` must be unique positive integers, and `value`
must be `0` or `1`.
"""
struct ExactlyOneConstraint <: AbstractConstraint
  sites::Vector{Int}
  value::Int

  function ExactlyOneConstraint(sites, value)
    return new(validate_sites(sites), validate_binary_value(value, "value"))
  end
end

ExactlyOneConstraint(sites) = ExactlyOneConstraint(sites, 1)

"""
    RelationConstraint <: AbstractConstraint
    RelationConstraint(left_site, relation, right_site)

Pairwise constraint over a binary vector `x`:
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

Test whether the binary vector `x` (entries in `{0, 1}`, 1-based indexing)
satisfies a single `constraint <: AbstractConstraint`.

Throws an `ArgumentError` if `x` contains a non-binary entry and a `BoundsError`
if a constraint references a site outside `x`.
"""
function is_feasible end

function is_feasible(x::AbstractVector, constraint::SumConstraint)
  lhs = sum(weight * binary_at(x, site) for (site, weight) in constraint.weights)
  return relation_holds(lhs, constraint.relation, constraint.rhs)
end

function is_feasible(x::AbstractVector, constraint::NotEqualsConstraint)
  return any(binary_at(x, site) != value for (site, value) in constraint.values)
end

function is_feasible(x::AbstractVector, constraint::ExactlyOneConstraint)
  return count(binary_at(x, site) == constraint.value for site in constraint.sites) == 1
end

function is_feasible(x::AbstractVector, constraint::RelationConstraint)
  lhs = binary_at(x, constraint.left_site)
  rhs = binary_at(x, constraint.right_site)

  return relation_holds(lhs, constraint.relation, rhs)
end

"""
    is_feasible(x, constraints::Vector{AbstractConstraint})

Test whether the binary vector `x`
satisfies every constraint in a vector `constraints`.
Any vector is feasible for an empty constraint vector.

Throws an `ArgumentError` if `x` contains a non-binary entry and a `BoundsError`
if a constraint references a site outside `x`.
"""
function is_feasible(x::AbstractVector, constraints::AbstractVector{<:AbstractConstraint})
  return all(constraint -> is_feasible(x, constraint), constraints)
end

"""
    constraint_sites(constraint::AbstractConstraint)

Access the site indices stored in the `constraint`.
"""
function constraint_sites end

# TODO: Why do we always have to sort it?
function constraint_sites(constraint::SumConstraint)
  return sort!(collect(keys(constraint.weights)))
end

function constraint_sites(constraint::NotEqualsConstraint)
  return sort!(collect(keys(constraint.values)))
end

function constraint_sites(constraint::ExactlyOneConstraint)
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
  relation isa Symbol || throw(ArgumentError("relation must be a Symbol"))
  relation in VALID_RELATIONS ||
    throw(ArgumentError("relation must be one of: $(join(string.(VALID_RELATIONS), ", "))"))

  return relation
end

function validate_binary_values(values, name)
  if isempty(values)
    throw(ArgumentError("$name must not be empty"))
  end
  if any(!in((0, 1)), values)
    throw(ArgumentError("$name must contain only binary values 0 or 1"))
  end

  return values
end

function validate_binary_value(value, name)
  if !(isequal(value, 0) || isequal(value, 1))
    throw(ArgumentError("$name must be a binary value 0 or 1"))
  end

  return Int(value)
end

function binary_at(x, site)
  checkbounds(x, site)
  value = x[site]
  if !(value in (0, 1))
    throw(ArgumentError("x must contain only binary values 0 or 1"))
  end

  return Int(value)
end

function relation_holds(lhs, relation, rhs)
  relation === Symbol("==") && return lhs == rhs
  relation === Symbol("!=") && return lhs != rhs
  relation === Symbol("<=") && return lhs <= rhs
  relation === Symbol(">=") && return lhs >= rhs

  error("unsupported relation: $relation")
end
