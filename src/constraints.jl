abstract type AbstractConstraint end

struct SumConstraint{T<:Real} <: AbstractConstraint
  sites::Vector{Int}
  weights::Vector{T}
  relation::Symbol
  rhs::T

  function SumConstraint{T}(
    sites::AbstractVector{<:Integer},
    weights::AbstractVector{T},
    relation::Symbol,
    rhs::T,
  ) where {T<:Real}
    site_vec = _validate_sites(sites)
    weight_vec = _validate_weights(weights)
    _validate_same_length(site_vec, weight_vec, "sites", "weights")
    relation = _validate_relation(relation)

    return new{T}(site_vec, weight_vec, relation, rhs)
  end
end

function SumConstraint(sites, weights, relation, rhs)
  site_vec = _validate_sites(sites)
  raw_weights = collect(weights)
  isempty(raw_weights) && throw(ArgumentError("weights must not be empty"))

  weight_types = map(typeof, raw_weights)
  T = promote_type(weight_types..., typeof(rhs))
  T <: Real || throw(ArgumentError("weights and rhs must be real-valued"))

  weight_vec = T.(raw_weights)
  rhs_value = convert(T, rhs)

  return SumConstraint{T}(site_vec, weight_vec, _validate_relation(relation), rhs_value)
end

struct NotEqualsConstraint <: AbstractConstraint
  sites::Vector{Int}
  values::Vector{Int}

  function NotEqualsConstraint(sites, values)
    site_vec = _validate_sites(sites)
    value_vec = _validate_binary_values(values, "values")
    _validate_same_length(site_vec, value_vec, "sites", "values")

    return new(site_vec, value_vec)
  end
end

struct ExactlyOneConstraint <: AbstractConstraint
  sites::Vector{Int}

  function ExactlyOneConstraint(sites)
    return new(_validate_sites(sites))
  end
end

struct RelationConstraint <: AbstractConstraint
  left_site::Int
  relation::Symbol
  right_site::Int

  function RelationConstraint(left_site, relation, right_site)
    left = _validate_site(left_site, "left_site")
    right = _validate_site(right_site, "right_site")
    left == right && throw(ArgumentError("relation constraint sites must be distinct"))

    return new(left, _validate_relation(relation), right)
  end
end

sum_constraint(sites, weights, relation, rhs) = SumConstraint(sites, weights, relation, rhs)
sum_constraint(sites, weights, rhs; relation=Symbol("==")) = SumConstraint(sites, weights, relation, rhs)
not_equals_constraint(sites, values) = NotEqualsConstraint(sites, values)
exactly_one_constraint(sites) = ExactlyOneConstraint(sites)
relation_constraint(left_site, relation, right_site) = RelationConstraint(left_site, relation, right_site)

function is_feasible(x::AbstractVector, constraint::SumConstraint)
  lhs = sum(
    constraint.weights[i] * _binary_at(x, constraint.sites[i])
    for i in eachindex(constraint.sites, constraint.weights)
  )

  return _relation_holds(lhs, constraint.relation, constraint.rhs)
end

function is_feasible(x::AbstractVector, constraint::NotEqualsConstraint)
  return any(
    _binary_at(x, constraint.sites[i]) != constraint.values[i]
    for i in eachindex(constraint.sites, constraint.values)
  )
end

function is_feasible(x::AbstractVector, constraint::ExactlyOneConstraint)
  return sum(_binary_at(x, site) for site in constraint.sites) == 1
end

function is_feasible(x::AbstractVector, constraint::RelationConstraint)
  lhs = _binary_at(x, constraint.left_site)
  rhs = _binary_at(x, constraint.right_site)

  return _relation_holds(lhs, constraint.relation, rhs)
end

function is_feasible(x::AbstractVector, constraints::AbstractVector{<:AbstractConstraint})
  return all(constraint -> is_feasible(x, constraint), constraints)
end

const _VALID_RELATIONS = (
  Symbol("=="),
  Symbol("!="),
  Symbol("<"),
  Symbol("<="),
  Symbol(">"),
  Symbol(">="),
)

function _validate_site(site, name)
  site isa Integer || throw(ArgumentError("$name must be an integer"))
  site > 0 || throw(ArgumentError("$name must be a positive integer"))

  return Int(site)
end

function _validate_sites(sites)
  site_vec = collect(sites)
  isempty(site_vec) && throw(ArgumentError("sites must not be empty"))

  validated = [_validate_site(site, "sites") for site in site_vec]
  length(unique(validated)) == length(validated) ||
    throw(ArgumentError("sites must be unique"))

  return validated
end

function _validate_same_length(left, right, left_name, right_name)
  length(left) == length(right) ||
    throw(DimensionMismatch("$left_name and $right_name must have the same length"))

  return nothing
end

function _validate_weights(weights)
  weight_vec = collect(weights)
  isempty(weight_vec) && throw(ArgumentError("weights must not be empty"))
  all(weight -> weight >= 0, weight_vec) ||
    throw(ArgumentError("weights must be nonnegative"))

  return weight_vec
end

function _validate_relation(relation)
  relation isa Symbol || throw(ArgumentError("relation must be a Symbol"))
  relation in _VALID_RELATIONS ||
    throw(ArgumentError("relation must be one of: $(join(string.(_VALID_RELATIONS), ", "))"))

  return relation
end

function _validate_binary_values(values, name)
  value_vec = collect(values)
  isempty(value_vec) && throw(ArgumentError("$name must not be empty"))
  all(value -> value == 0 || value == 1, value_vec) ||
    throw(ArgumentError("$name must contain only binary values 0 or 1"))

  return Int.(value_vec)
end

function _binary_at(x, site)
  checkbounds(x, site)
  value = x[site]
  value == 0 || value == 1 ||
    throw(ArgumentError("x must contain only binary values 0 or 1"))

  return Int(value)
end

function _relation_holds(lhs, relation, rhs)
  relation === Symbol("==") && return lhs == rhs
  relation === Symbol("!=") && return lhs != rhs
  relation === Symbol("<") && return lhs < rhs
  relation === Symbol("<=") && return lhs <= rhs
  relation === Symbol(">") && return lhs > rhs
  relation === Symbol(">=") && return lhs >= rhs

  error("unsupported relation: $relation")
end
