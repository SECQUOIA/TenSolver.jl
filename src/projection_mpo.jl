# Projection-MPO construction adapted from the CoTenN constraint projection
# design in Sharma, Ritvik, Cheng Peng, Siddharth Dangwal, and Sara Achour,
# "CoTenN: Constrained Optimization with Tensor Networks," PLDI 2026.
#
# Note on ITensors.jl / ITensorMPS.jl helpers: the on-site identity is built with
# `ITensors.delta` rather than a hand-rolled Kronecker tensor. The higher-level
# `OpSum`/`MPO(::OpSum, sites)` (AutoMPO) machinery was considered for the full
# assembly but is intentionally not used here: it heuristically compresses the
# operator and does not guarantee the exact, uncompressed one-path-per-feasible-
# assignment structure this module needs (bond dimension = number of feasible
# assignments). Bounding/compressing that bond dimension is deferred to #58, where
# projections meet solves and the downstream cost is measurable. The remaining
# path-threaded tensors have no direct high-level equivalent, so they are assembled
# from sparse nonzeros via `itensor_from_nonzeros`.

"""
    SparseTensorEntry{T}

One nonzero term in the sparse representation used to assemble a projection
MPO. `coordinates` maps 1-based register sites to 1-based local basis states;
sites omitted from the dictionary are unconstrained by this entry. `value` is
the coefficient carried by that partial assignment.
"""
struct SparseTensorEntry{T}
  coordinates::Dict{Int,Int}
  value::T
end

function validate_projection_sites(sites)
  isempty(sites) && throw(ArgumentError("sites must not be empty"))
  all(site -> ITensors.dim(site) == 2, sites) ||
    throw(ArgumentError("projection MPO sites must be binary Qudit indices"))

  return nothing
end

function validate_constraint_site_bounds(constraint_sites, sites)
  all(site -> 1 <= site <= length(sites), constraint_sites) ||
    throw(BoundsError(sites, maximum(constraint_sites)))

  return nothing
end

"""
    projection_entries(::Type{T}, constraint::AbstractConstraint)

Enumerate feasible assignments on the sites touched by `constraint`.
"""
function projection_entries(::Type{T}, constraint::AbstractConstraint) where {T}
  cs = constraint_sites(constraint)
  assignments = Iterators.product(fill(0:1, length(cs))...)
  entries = SparseTensorEntry{T}[]

  for assignment in assignments
    x = zeros(Int, maximum(cs))
    coordinates = Dict{Int,Int}()

    for (site, bit) in zip(cs, assignment)
      x[site] = bit
      coordinates[site] = bit + 1
    end

    if is_feasible(x, constraint)
      push!(entries, SparseTensorEntry{T}(coordinates, one(T)))
    end
  end

  return entries
end

function projection_entries_to_dfa(entries, num_sites::Integer, constrained_sites)
  constrained_positions = sort(collect(constrained_sites))
  position_to_depth = Dict(site_position => depth for (depth, site_position) in enumerate(constrained_positions))

  children = [Dict{Int,Int}()]
  states_by_depth = Vector{Vector{Int}}(undef, length(constrained_positions) + 1)
  states_by_depth[1] = [1]
  for depth in 2:length(states_by_depth)
    states_by_depth[depth] = Int[]
  end

  accepting = Set{Int}()

  for entry in entries
    node = 1
    for (depth, site_position) in enumerate(constrained_positions)
      bit = get(entry.coordinates, site_position, 1) - 1
      child = get(children[node], bit, 0)

      if child == 0
        child = length(children) + 1
        push!(children, Dict{Int,Int}())
        push!(states_by_depth[depth + 1], child)
        children[node][bit] = child
      end

      node = child
    end

    push!(accepting, node)
  end

  states = collect(eachindex(children))
  transitions = [Dict{Tuple{Int,Int},Int}() for _ in 1:num_sites]

  for site_position in 1:num_sites
    if !haskey(position_to_depth, site_position)
      for state in states
        transitions[site_position][(state, 0)] = state
        transitions[site_position][(state, 1)] = state
      end
      continue
    end

    depth = position_to_depth[site_position]
    for state in states_by_depth[depth]
      for bit in 0:1
        child = get(children[state], bit, 0)
        child == 0 && continue
        transitions[site_position][(state, bit)] = child
      end
    end
  end

  return DFA(states, [0, 1], 1, accepting, transitions)
end


"""
    constraint_to_dfa([T], constraint, sites)

Build a DFA for `constraint` over `sites`.
"""
function constraint_to_dfa(::Type{T}, constraint::AbstractConstraint, sites) where {T}
  site_vec = collect(sites)
  validate_projection_sites(site_vec)

  cs = constraint_sites(constraint)
  validate_constraint_site_bounds(cs, site_vec)

  return projection_entries_to_dfa(
    projection_entries(T, constraint),
    length(site_vec),
    cs,
  )
end

function sum_constraint_projection_data(constraint::SumConstraint)
  weights_by_site = Dict{Int,Int}()
  for (site, weight) in zip(constraint.sites, constraint.weights)
    weights_by_site[site] = integer_projection_value(weight, "SumConstraint weights")
  end

  rhs = integer_projection_value(constraint.rhs, "SumConstraint rhs")
  return weights_by_site, rhs
end

function constraint_to_dfa(::Type{T}, constraint::SumConstraint, sites) where {T}
  site_vec = collect(sites)
  validate_projection_sites(site_vec)
  validate_constraint_site_bounds(constraint.sites, site_vec)

  weights_by_site, rhs = sum_constraint_projection_data(constraint)
  state_limit = sum_projection_state_limit(constraint.relation, rhs)
  states_after_site = sum_projection_reachable_states(
    weights_by_site,
    length(site_vec),
    constraint.relation,
    rhs,
  )

  states = Int[0]
  for reachable in states_after_site
    append!(states, reachable)
  end
  unique!(sort!(states))

  accepting_states = Set(
    filter(state -> relation_holds(state, constraint.relation, rhs), states_after_site[end])
  )

  transitions = [Dict{Tuple{Int,Int},Int}() for _ in eachindex(site_vec)]
  for site_position in eachindex(site_vec)
    left_states = site_position == firstindex(site_vec) ? [0] : states_after_site[site_position - 1]
    weight = get(weights_by_site, site_position, 0)

    for partial_sum in left_states
      for bit in 0:1
        next_sum = sum_transition_target(partial_sum, weight, bit, state_limit)
        isnothing(next_sum) && continue
        transitions[site_position][(partial_sum, bit)] = next_sum
      end
    end
  end

  return DFA(states, [0, 1], 0, accepting_states, transitions)
end

function integer_projection_value(value::Real, name)
  isinteger(value) ||
    throw(ArgumentError("$name must be integer-valued for SumConstraint projection MPOs"))

  try
    return Int(value)
  catch error
    if error isa InexactError || error isa OverflowError
      throw(ArgumentError("$name must fit in Int for SumConstraint projection MPOs"))
    end
    rethrow()
  end
end

function checked_sum_projection_add(lhs::Int, rhs::Int)
  try
    return Base.checked_add(lhs, rhs)
  catch error
    if error isa OverflowError
      throw(ArgumentError("SumConstraint projection partial sums must fit in Int"))
    end
    rethrow()
  end
end

function sum_projection_state_limit(relation::Symbol, rhs::Int)
  (relation === Symbol("<=") || relation === Symbol("==")) && return rhs

  return nothing
end

function sum_exceeds_state_limit(partial_sum::Int, weight::Int, state_limit::Int)
  return big(partial_sum) + big(weight) > state_limit
end

function sum_transition_target(partial_sum::Int, weight::Int, bit::Integer, state_limit)
  bit == 0 && return partial_sum

  if !isnothing(state_limit) && sum_exceeds_state_limit(partial_sum, weight, state_limit)
    return nothing
  end

  return checked_sum_projection_add(partial_sum, weight)
end

function sum_projection_reachable_states(weights_by_site, num_sites, relation::Symbol, rhs::Int)
  states_after_site = Vector{Vector{Int}}(undef, num_sites)
  state_limit = sum_projection_state_limit(relation, rhs)
  reachable = !isnothing(state_limit) && 0 > state_limit ? Int[] : [0]

  for site_position in 1:num_sites
    weight = get(weights_by_site, site_position, 0)
    next_reachable = Int[]

    for partial_sum in reachable, bit in 0:1
      target = sum_transition_target(partial_sum, weight, bit, state_limit)
      isnothing(target) && continue
      push!(next_reachable, target)
    end

    unique!(sort!(next_reachable))
    states_after_site[site_position] = next_reachable
    reachable = next_reachable
  end

  return states_after_site
end


"""
    projection_mpo([T], constraint, sites)

Build a diagonal projection MPO over the binary Qudit register `sites`.

The diagonal entry is `one(T)` for computational basis states whose bits satisfy
`constraint`, and zero otherwise. Constraint site numbers use the same 1-based
register indexing as `sites`. The generic construction is exact and
uncompressed: each feasible assignment of the constrained sites is represented
as one MPO path.

`SumConstraint` uses a specialized exact integer partial-sum automaton. Its
weights and right-hand side must be integer-valued. Link dimensions grow with
the number of distinct reachable partial sums, which can be exponential in the
number of weighted sites. For nonnegative `<=` and `==` constraints, partial
sums exceeding the right-hand side are pruned from the automaton.
"""
function projection_mpo end


function projection_mpo(::Type{T}, constraint::AbstractConstraint, sites) where {T}
  site_vec = collect(sites)
  return dfa_to_mpo(T, constraint_to_dfa(T, constraint, site_vec), site_vec)
end

projection_mpo(constraint::AbstractConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

"""
    projection_mpos([T], constraints, sites)

Build one projection MPO per constraint over the shared binary Qudit register
`sites`.

This is a convenience wrapper around [`projection_mpo`](@ref) that reuses the
same collected site register for every constraint. `T` controls the numeric
element type of the assembled MPO tensors.
"""
function projection_mpos(::Type{T}, constraints::AbstractVector{<:AbstractConstraint}, sites) where {T}
  site_vec = collect(sites)

  return [projection_mpo(T, constraint, site_vec) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, sites) =
  projection_mpos(Float64, constraints, sites)
