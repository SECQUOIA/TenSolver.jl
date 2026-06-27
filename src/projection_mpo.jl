# Projection-MPO construction adapted from the CoTenN constraint projection
# design in Sharma, Ritvik, Cheng Peng, Siddharth Dangwal, and Sara Achour,
# "CoTenN: Constrained Optimization with Tensor Networks," PLDI 2026.

struct SparseTensorEntry{T}
  coordinates::Dict{Int,Int}
  value::T
end

"""
    itensor_from_nonzeros([T], inds, nonzeros)

Build an `ITensor` over `inds` from sparse `(coordinate, value)` entries.
Coordinates are 1-based tuples matching `inds`.
"""
function itensor_from_nonzeros(::Type{T}, inds, nonzeros) where {T}
  ind_tuple = Tuple(inds)
  tensor = ITensors.ITensor(T, ind_tuple...)

  for (coordinate, value) in nonzeros
    coordinate_tuple = Tuple(coordinate)
    length(coordinate_tuple) == length(ind_tuple) ||
      throw(DimensionMismatch("nonzero coordinate length must match tensor order"))

    selector = map(i -> ind_tuple[i] => coordinate_tuple[i], eachindex(ind_tuple))
    tensor[selector...] = tensor[selector...] + convert(T, value)
  end

  return tensor
end

itensor_from_nonzeros(inds, nonzeros) = itensor_from_nonzeros(Float64, inds, nonzeros)

function identity_tensor(::Type{T}, site) where {T}
  site_prime = ITensors.prime(site)
  nonzeros = (((state, state), one(T)) for state in 1:ITensors.dim(site))

  return itensor_from_nonzeros(T, (site, site_prime), nonzeros)
end

function pass_through_tensor(::Type{T}, site, left_link, right_link) where {T}
  if isnothing(left_link) && isnothing(right_link)
    return identity_tensor(T, site)
  end

  inds = mpo_tensor_indices(site, left_link, right_link)
  num_paths = path_count(left_link, right_link)
  nonzeros = Tuple{Vector{Int},T}[]

  for path in 1:num_paths, state in 1:ITensors.dim(site)
    push!(nonzeros, (mpo_coordinate(state, path, left_link, right_link), one(T)))
  end

  return itensor_from_nonzeros(T, inds, nonzeros)
end

function path_count(left_link, right_link)
  !isnothing(left_link) && return ITensors.dim(left_link)
  !isnothing(right_link) && return ITensors.dim(right_link)

  return 1
end

function mpo_tensor_indices(site, left_link, right_link)
  inds = []
  !isnothing(left_link) && push!(inds, left_link)
  push!(inds, site, ITensors.prime(site))
  !isnothing(right_link) && push!(inds, right_link)

  return Tuple(inds)
end

function mpo_coordinate(state::Integer, path::Integer, left_link, right_link)
  coordinate = Int[]
  !isnothing(left_link) && push!(coordinate, path)
  push!(coordinate, state, state)
  !isnothing(right_link) && push!(coordinate, path)

  return coordinate
end

function tensor_to_mpo(::Type{T}, entries, sites) where {T}
  site_vec = collect(sites)
  validate_projection_sites(site_vec)

  entry_vec = collect(entries)
  validate_sparse_entries(entry_vec, site_vec)

  num_paths = max(length(entry_vec), 1)
  links = [
    ITensors.Index(num_paths, "Link,Projection,l=$i")
    for i in 1:(length(site_vec) - 1)
  ]

  tensors = Vector{ITensors.ITensor}(undef, length(site_vec))
  for site_position in eachindex(site_vec)
    site = site_vec[site_position]
    left_link = site_position == firstindex(site_vec) ? nothing : links[site_position - 1]
    right_link = site_position == lastindex(site_vec) ? nothing : links[site_position]
    if !isempty(entry_vec) && all(entry -> !haskey(entry.coordinates, site_position), entry_vec)
      tensors[site_position] = pass_through_tensor(T, site, left_link, right_link)
      continue
    end

    inds = mpo_tensor_indices(site, left_link, right_link)
    nonzeros = Tuple{Vector{Int},T}[]

    for (path, entry) in enumerate(entry_vec)
      states = get(entry.coordinates, site_position, nothing)
      states = isnothing(states) ? (1, 2) : (states,)
      coefficient = site_position == firstindex(site_vec) ? entry.value : one(T)

      for state in states
        push!(nonzeros, (mpo_coordinate(state, path, left_link, right_link), coefficient))
      end
    end

    tensors[site_position] = itensor_from_nonzeros(T, inds, nonzeros)
  end

  return ITensorMPS.MPO(tensors)
end

function validate_projection_sites(sites)
  isempty(sites) && throw(ArgumentError("sites must not be empty"))
  all(site -> ITensors.dim(site) == 2, sites) ||
    throw(ArgumentError("projection MPO sites must be binary Qudit indices"))

  return nothing
end

function validate_sparse_entries(entries, sites)
  for entry in entries
    for (site_position, coordinate) in entry.coordinates
      checkbounds(sites, site_position)
      1 <= coordinate <= ITensors.dim(sites[site_position]) ||
        throw(BoundsError(1:ITensors.dim(sites[site_position]), coordinate))
    end
  end

  return nothing
end

function validate_constraint_site_bounds(constraint_sites, sites)
  all(site -> 1 <= site <= length(sites), constraint_sites) ||
    throw(BoundsError(sites, maximum(constraint_sites)))

  return nothing
end

function constraint_sites(constraint::NotEqualsConstraint)
  return constraint.sites
end

function constraint_sites(constraint::ExactlyOneConstraint)
  return constraint.sites
end

function constraint_sites(constraint::RelationConstraint)
  return [constraint.left_site, constraint.right_site]
end

function projection_entries(::Type{T}, constraint::NotEqualsConstraint) where {T}
  return projection_entries(T, constraint, constraint_sites(constraint))
end

function projection_entries(::Type{T}, constraint::ExactlyOneConstraint) where {T}
  return projection_entries(T, constraint, constraint_sites(constraint))
end

function projection_entries(::Type{T}, constraint::RelationConstraint) where {T}
  return projection_entries(T, constraint, constraint_sites(constraint))
end

function projection_entries(::Type{T}, constraint::AbstractConstraint, constraint_sites) where {T}
  assignments = Iterators.product(fill(0:1, length(constraint_sites))...)
  entries = SparseTensorEntry{T}[]

  for assignment in assignments
    x = zeros(Int, maximum(constraint_sites))
    coordinates = Dict{Int,Int}()

    for (site, bit) in zip(constraint_sites, assignment)
      x[site] = bit
      coordinates[site] = bit + 1
    end

    if is_feasible(x, constraint)
      push!(entries, SparseTensorEntry{T}(coordinates, one(T)))
    end
  end

  return entries
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

function sum_constraint_projection_data(constraint::SumConstraint)
  weights_by_site = Dict{Int,Int}()
  for (site, weight) in zip(constraint.sites, constraint.weights)
    weights_by_site[site] = integer_projection_value(weight, "SumConstraint weights")
  end

  rhs = integer_projection_value(constraint.rhs, "SumConstraint rhs")

  return weights_by_site, rhs
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

function sum_state_position_map(states)
  return Dict(state => position for (position, state) in pairs(states))
end

function mpo_transition_coordinate(state::Integer, left_position, right_position, left_link, right_link)
  coordinate = Int[]
  !isnothing(left_link) && push!(coordinate, left_position)
  push!(coordinate, state, state)
  !isnothing(right_link) && push!(coordinate, right_position)

  return coordinate
end

function sum_constraint_projection_tensor(
  ::Type{T},
  site,
  left_link,
  right_link,
  left_states,
  right_positions,
  weight::Int,
  relation::Symbol,
  rhs::Int,
  state_limit,
) where {T}
  inds = mpo_tensor_indices(site, left_link, right_link)
  nonzeros = Tuple{Vector{Int},T}[]

  for (left_position, partial_sum) in pairs(left_states)
    for bit in 0:1
      next_sum = sum_transition_target(partial_sum, weight, bit, state_limit)
      isnothing(next_sum) && continue

      if isnothing(right_link)
        relation_holds(next_sum, relation, rhs) || continue
        right_position = nothing
      else
        right_position = right_positions[next_sum]
      end

      state = bit + 1
      coordinate = mpo_transition_coordinate(
        state,
        left_position,
        right_position,
        left_link,
        right_link,
      )
      push!(nonzeros, (coordinate, one(T)))
    end
  end

  return itensor_from_nonzeros(T, inds, nonzeros)
end

function sum_constraint_projection_mpo(::Type{T}, constraint::SumConstraint, sites) where {T}
  validate_projection_sites(sites)
  validate_constraint_site_bounds(constraint.sites, sites)

  weights_by_site, rhs = sum_constraint_projection_data(constraint)
  state_limit = sum_projection_state_limit(constraint.relation, rhs)
  states_after_site = sum_projection_reachable_states(
    weights_by_site,
    length(sites),
    constraint.relation,
    rhs,
  )
  any(isempty, states_after_site) && return tensor_to_mpo(T, SparseTensorEntry{T}[], sites)

  links = [
    ITensors.Index(length(states_after_site[site_position]), "Link,Projection,Sum,l=$site_position")
    for site_position in 1:(length(sites) - 1)
  ]

  tensors = Vector{ITensors.ITensor}(undef, length(sites))
  for site_position in eachindex(sites)
    site = sites[site_position]
    left_link = site_position == firstindex(sites) ? nothing : links[site_position - 1]
    right_link = site_position == lastindex(sites) ? nothing : links[site_position]
    left_states = site_position == firstindex(sites) ? [0] : states_after_site[site_position - 1]
    right_positions = isnothing(right_link) ? nothing : sum_state_position_map(states_after_site[site_position])
    weight = get(weights_by_site, site_position, 0)

    tensors[site_position] = sum_constraint_projection_tensor(
      T,
      site,
      left_link,
      right_link,
      left_states,
      right_positions,
      weight,
      constraint.relation,
      rhs,
      state_limit,
    )
  end

  return ITensorMPS.MPO(tensors)
end

"""
    projection_mpo([T], constraint, sites)

Build a diagonal projection MPO that preserves basis states satisfying
`constraint` and zeros infeasible states.

`SumConstraint` projection MPOs use exact integer partial sums, so their
weights and right-hand side must be integer-valued.
For SumConstraint, MPO link dimensions grow with the number of distinct
reachable partial sums, which can be exponential in the number of weighted
sites.
For nonnegative `<=` and `==` constraints, partial sums exceeding the right-hand
side are pruned from the automaton.
"""
function projection_mpo(::Type{T}, constraint::SumConstraint, sites) where {T}
  return sum_constraint_projection_mpo(T, constraint, collect(sites))
end

function projection_mpo(::Type{T}, constraint::NotEqualsConstraint, sites) where {T}
  return projection_mpo(T, constraint, constraint_sites(constraint), sites)
end

function projection_mpo(::Type{T}, constraint::ExactlyOneConstraint, sites) where {T}
  return projection_mpo(T, constraint, constraint_sites(constraint), sites)
end

function projection_mpo(::Type{T}, constraint::RelationConstraint, sites) where {T}
  return projection_mpo(T, constraint, constraint_sites(constraint), sites)
end

function projection_mpo(::Type{T}, constraint::AbstractConstraint, constraint_sites, sites) where {T}
  site_vec = collect(sites)
  validate_constraint_site_bounds(constraint_sites, site_vec)

  return tensor_to_mpo(T, projection_entries(T, constraint), site_vec)
end

projection_mpo(constraint::SumConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

projection_mpo(constraint::NotEqualsConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

projection_mpo(constraint::ExactlyOneConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

projection_mpo(constraint::RelationConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

"""
    projection_mpos([T], constraints, sites)

Build one projection MPO per constraint.
"""
function projection_mpos(::Type{T}, constraints::AbstractVector{<:AbstractConstraint}, sites) where {T}
  site_vec = collect(sites)

  return [projection_mpo(T, constraint, site_vec) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, sites) =
  projection_mpos(Float64, constraints, sites)
