integer(::Type{T}) where {T<:Integer} = T
integer(::Type) = Int

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
struct SumConstraint{T<:Integer} <: AbstractConstraint
  weights::Dict{Int,T}
  relation::Symbol
  rhs::T

  function SumConstraint{T}(sites, weights, relation, rhs) where {T<:Integer}
    @argcheck allunique(sites)
    @argcheck all(>(0), sites)
    @argcheck length(weights) == length(sites) DimensionMismatch
    @argcheck all(>=(0), weights)  # WIP: this is only necessary at the DFA level
    @argcheck rhs >= 0
    @argcheck relation in VALID_RELATIONS

    # Helps reduce the bond dimension
    weights = T.(weights)
    rhs     = T(rhs)
    g       = gcd(rhs, weights...)
    @. weights = div(weights, g)
    rhs        = div(rhs, g)

    weight_map = Dict{Int,T}(zip(sites, weights))
    filter!(p -> !iszero(p.second), weight_map)

    return new{T}(weight_map, relation, rhs)
  end
end

function SumConstraint(sites, weights, relation, rhs)
  T = integer(promote_type(typeof(rhs), eltype(weights)))
  return SumConstraint{T}(sites, weights, relation, rhs)
end

function SumConstraint(sites, weights, rhs; relation)
  return SumConstraint(sites, weights, relation, rhs)
end

"""
    SumModConstraint{T} <: AbstractConstraint
    SumModConstraint(sites, weights, rhs; mod)

Modular weighted-sum constraint over a vector `x`:

    sum(weights[i] * x[sites[i]] for i in eachindex(sites)) ≡ rhs (mod m).

`sites` must be unique positive integers,
`weights` must be the same length as `sites`,
`weights` and `rhs` must be integer-valued,
and `mod` must be a positive integer.

Weights and the rhs are stored as their least nonnegative residues modulo `mod`.
"""
struct SumModConstraint{T<:Integer} <: AbstractConstraint
  weights::Dict{Int,T}
  rhs::T
  mod::T

  function SumModConstraint{T}(sites, weights, rhs; mod) where {T<:Integer}
    @argcheck all(>(0), sites)
    @argcheck allunique(sites)
    @argcheck length(weights) == length(sites) DimensionMismatch
    @argcheck mod >= 1

    # Helps reduce the bond dimension
    weights = T.(weights)
    rhs     = T(rhs)
    mod     = T(mod)

    @. weights = Base.mod(weights, mod)
    rhs        = Base.mod(rhs, mod)

    weight_map = Dict{Int,T}(zip(sites, weights))
    filter!(p -> !iszero(p.second), weight_map)

    return new{T}(weight_map, rhs, mod)
  end
end

function SumModConstraint(sites, weights, rhs; mod)
  T = integer(promote_type(typeof(rhs), typeof(mod), eltype(weights)))
  return SumModConstraint{T}(sites, T.(weights), convert(T, rhs); mod=convert(T, mod))
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
    @argcheck all(>(0), sites)
    @argcheck allunique(sites)
    @argcheck length(values) == length(sites) DimensionMismatch

    value_map = Dict{Int,T}(zip(sites, values))

    return new{T}(value_map)
  end
end

function NotEqualsConstraint(sites, values)
  return NotEqualsConstraint{eltype(values)}(sites, values)
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
    @argcheck all(>(0), sites)
    @argcheck allunique(sites)
    @argcheck rhs >= 0
    @argcheck relation in VALID_RELATIONS

    return new{T}(sites, Set(values), relation, Int(rhs))
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

  function RelationConstraint(left, relation, right)
    @argcheck left  > 0
    @argcheck right > 0
    @argcheck left != right
    @argcheck relation in VALID_RELATIONS

    return new(left, relation, right)
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
  lhs = mod(sum(w * x[s] for (s, w) in constraint.weights), constraint.mod)
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
# Valid relations
#----------------------------------------------------------#

const VALID_RELATIONS = (
  Symbol("=="),
  Symbol("!="),
  Symbol("<="),
  Symbol(">="),
)

function relation_holds(lhs, relation, rhs)
  relation === Symbol("==") && return lhs == rhs
  relation === Symbol("!=") && return lhs != rhs
  relation === Symbol("<=") && return lhs <= rhs
  relation === Symbol(">=") && return lhs >= rhs

  error("unsupported relation: $relation")
end
