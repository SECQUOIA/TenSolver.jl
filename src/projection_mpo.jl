# Projection-MPO construction adapted from the CoTenN constraint projection
# design in Sharma, Ritvik, Cheng Peng, Siddharth Dangwal, and Sara Achour,
# "CoTenN: Constrained Optimization with Tensor Networks," PLDI 2026.
#
# The implementation builds exact diagonal projection MPOs by lowering each
# constraint to a step-dependent DFA and then threading the DFA through sparse
# nonzero tensor entries. The helpers below assemble the MPO directly from those
# nonzero paths.


###############################################################################
# Finite Automata to MPO utilities
###############################################################################

"""
    DFA{S, A}

Step-dependent deterministic finite automaton.

Fields:
- `states`: DFA states, used to define the MPO bond dimension.
- `alphabet`: local symbols, ordered to match the physical basis positions.
- `initial`: start state.
- `accepting`: set of accepting states.
- `transitions`: one transition table per site; each table maps `(state, symbol)` to
  the next state. Missing entries are rejected.
"""
struct DFA{S,A}
  states::Vector{S}
  alphabet::Vector{A}
  initial::S
  accepting::Set{S}
  transitions::Vector{Dict{Tuple{S,A},S}}
  function DFA{S,A}(states, alphabet, initial, accepting, transitions) where {S,A}
    state_vec      = collect(S, states)
    alphabet_vec   = collect(A, alphabet)
    transition_vec = collect(transitions)
    accepting_set  = Set{S}(accepting)
    initial_state  = convert(S, initial)

    alphabet_set   = Set(alphabet_vec)
    state_set      = Set(state_vec)

    # Validate here so all downstream code can assume these invariants.
    if isempty(state_vec)
      throw(ArgumentError("states must not be empty"))
    end
    if isempty(alphabet_vec)
      throw(ArgumentError("alphabet must not be empty"))
    end
    if isempty(transition_vec)
      throw(ArgumentError("transitions must not be empty"))
    end
    if !allunique(state_vec)
      throw(ArgumentError("states must be unique"))
    end
    if !(initial_state in state_vec)
      throw(ArgumentError("initial must be one of the DFA states"))
    end
    if !issubset(accepting_set, state_set)
      throw(ArgumentError("accepting must be a subset of states"))
    end

    for (i, table) in enumerate(transition_vec), ((s, a), ns) in table
      if !(s in state_set)
        throw(ArgumentError("transition table $(i): unknown source state $(repr(s))"))
      end

      if !(a in alphabet_set)
        throw(ArgumentError("transition table $(i): symbol $(repr(a)) is not in the alphabet"))
      end

      if !(ns in state_set)
        throw(ArgumentError("transition table $(i): unknown target state $(repr(ns))"))
      end
    end

    return new{S,A}(state_vec, alphabet_vec, initial_state, accepting_set, transition_vec)
  end
end

function DFA(states, alphabet, initial, accepting, transitions)
  S = eltype(states)
  A = eltype(alphabet)
  return DFA{S,A}(states, alphabet, initial, accepting, transitions)
end

DFA(; states, alphabet, initial, accepting, transitions) =
  DFA(states, alphabet, initial, accepting, transitions)


function tensor_from_nonzeros(::Type{T}, inds, nonzeros) where {T}
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

function tensor_indices(site, left_link, right_link)
  return filter(!isnothing, (left_link, site, ITensors.prime(site), right_link))
end

function tensor_coordinate(symbol_pos::Integer, left_pos, right_pos, left_link, right_link)
  coordinate = Int[]
  !isnothing(left_link) && push!(coordinate, left_pos)
  push!(coordinate, symbol_pos, symbol_pos)
  !isnothing(right_link) && push!(coordinate, right_pos)
  return coordinate
end

function dfa_site_tensor(
  ::Type{T},
  site,
  left_link,
  right_link,
  dfa::DFA,
  step::Integer,
  state_positions,
) where {T}
  inds = tensor_indices(site, left_link, right_link)
  nonzeros = Tuple{Vector{Int},T}[]
  table = dfa.transitions[step]

  source_states = isnothing(left_link) ? (dfa.initial,) : dfa.states
  for source_state in source_states
    left_pos = isnothing(left_link) ? nothing : state_positions[source_state]

    for (symbol_pos, symbol) in enumerate(dfa.alphabet)
      next_state = get(table, (source_state, symbol), nothing)
      isnothing(next_state) && continue

      if isnothing(right_link)
        next_state in dfa.accepting || continue
        push!(
          nonzeros,
          (tensor_coordinate(symbol_pos, left_pos, nothing, left_link, right_link), one(T)),
        )
      else
        right_pos = state_positions[next_state]
        push!(
          nonzeros,
          (tensor_coordinate(symbol_pos, left_pos, right_pos, left_link, right_link), one(T)),
        )
      end
    end
  end

  return tensor_from_nonzeros(T, inds, nonzeros)
end

function validate_dfa_sites(dfa::DFA, sites)
  if isempty(sites)
    throw(ArgumentError("sites must not be empty"))
  end
  if any(site -> ITensors.dim(site) != length(dfa.alphabet), sites)
    throw(DimensionMismatch("each site dimension must match the DFA alphabet size"))
  end
  if length(sites) != length(dfa.transitions)
    throw(DimensionMismatch("DFA transition tables must match the number of sites"))
  end
end

"""
    dfa_to_mpo([T], dfa, sites)

Build an exact diagonal projection MPO from a step-dependent DFA.

The physical index basis positions are matched against `dfa.alphabet` in order:
`alphabet[k]` corresponds to local basis state `k`.

The MPO bond dimension equals `length(dfa.states)`.
"""
function dfa_to_mpo(::Type{T}, dfa::DFA, sites) where {T}
  validate_dfa_sites(dfa, sites)

  state_positions = Dict(state => i for (i, state) in enumerate(dfa.states))
  links = [
    ITensors.Index(length(dfa.states), "Link,DFA,l=$i")
    for i in 1:max(length(sites) - 1, 0)
  ]

  tensors = Vector{ITensors.ITensor}(undef, length(sites))
  for site_position in eachindex(sites)
    left_link  = site_position == firstindex(sites) ? nothing : links[site_position - 1]
    right_link = site_position == lastindex(sites)  ? nothing : links[site_position]

    tensors[site_position] = dfa_site_tensor(
      T,
      sites[site_position],
      left_link,
      right_link,
      dfa,
      site_position,
      state_positions,
    )
  end

  return ITensorMPS.MPO(tensors)
end

dfa_to_mpo(dfa::DFA, sites) = dfa_to_mpo(Float64, dfa, sites)

"""
    projection_mpo([T], constraint, sites)

Build a projection MPO over the binary Qudit register `sites`.

The diagonal entry is `one(T)` for computational basis states whose bits satisfy
`constraint`, and zero otherwise. Constraint site numbers use the same 1-based
register indexing as `sites`. The generic construction is exact and
uncompressed: each feasible assignment of the constrained sites is represented
as one MPO path.

The construction is exact and uncompressed. Constraint site numbers use the same
1-based register indexing as `sites`.

# Known constraints

- [`SumConstraint`](@ref) uses a specialized exact integer partial-sum automaton.
For a constraint with rhs `k`, its maximum bond dimension is `k+2`.
- [`NotEqualsConstraint`](@ref) uses a specialized MPO with bond dimension `2`,
  independently of the rhs.
- [`ExactlyOneConstraint`](@ref) uses a specialized MPO with bond dimension `2`
  that tracks whether the target value has been seen exactly once.
- [`RelationConstraint`](@ref) uses a specialized MPO with bond dimension `2`,
  independently of the compared site positions.
"""
function projection_mpo end


function projection_mpo(::Type{T}, constraint::AbstractConstraint, sites) where {T}
  return dfa_to_mpo(T, constraint_to_dfa(constraint, sites), sites)
end

projection_mpo(constraint::AbstractConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

"""
    projection_mpos([T], constraints, sites)

Build one projection MPO per constraint over the shared binary Qudit register
`sites`.

This is a convenience wrapper around [`projection_mpo`](@ref).
`T` controls the numeric
element type of the assembled MPO tensors.
"""
function projection_mpos(::Type{T}, constraints::AbstractVector{<:AbstractConstraint}, sites) where {T}
  return [projection_mpo(T, constraint, sites) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, sites) =
  projection_mpos(Float64, constraints, sites)


##############################################
# Generic constraint construction path
##############################################

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
  if isempty(sites)
    throw(ArgumentError("sites must not be empty"))
  end
  if any(site -> ITensors.dim(site) != 2, sites)
    throw(ArgumentError("projection MPO construction only supports Qudit sites with dim=2"))
  end
end

function validate_constraint_site_bounds(constraint_sites, sites)
  if !all(site -> 1 <= site <= length(sites), constraint_sites)
    throw(BoundsError(sites, maximum(constraint_sites)))
  end
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

  states = eachindex(children)
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

##############################################
# Constraint to DFA
##############################################

"""
    constraint_to_dfa(constraint, sites)

Build a DFA for `constraint` over `sites`.
"""
function constraint_to_dfa(constraint::AbstractConstraint, sites)
  validate_projection_sites(sites)

  cs = constraint_sites(constraint)
  validate_constraint_site_bounds(cs, sites)

  return projection_entries_to_dfa(
    projection_entries(Bool, constraint),
    length(sites),
    cs,
  )
end

function constraint_to_dfa(constraint::SumConstraint{S}, sites) where {S}
  (; weights, rhs, relation) = constraint

  beyond    = rhs + one(S)
  states    = zero(S):beyond
  alphabet  = [0, 1]
  accepting = Set(q for q in states if relation_holds(q, relation, rhs))

  transitions = [
    let weight = get(weights, i, zero(S))
      Dict(
        (q, a) => min(q + weight * S(a), beyond)
        for q in states, a in alphabet
      )
    end
    for i in eachindex(sites)
  ]

  return DFA(states, alphabet, zero(S), accepting, transitions)
end

function constraint_to_dfa(constraint::NotEqualsConstraint{S}, sites) where {S}
  validate_projection_sites(sites)

  cs = constraint_sites(constraint)
  validate_constraint_site_bounds(cs, sites)

  states    = S.([0, 1])
  alphabet  = [0, 1]
  initial   = 1
  accepting = Set(0)

  transitions = [
    let target = get(constraint.values, i, nothing)
      Dict{Tuple{Int,Int},Int}(
        (q, a) => isnothing(target) || S(a) == target ? q : 0
        for q in states, a in alphabet
      )
    end
    for i in eachindex(sites)
  ]

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::ExactlyOneConstraint, sites)
  validate_projection_sites(sites)

  cs = constraint_sites(constraint)
  validate_constraint_site_bounds(cs, sites)

  target = constraint.value
  constrained_sites = Set(cs)

  not_seen = 0
  seen_once = 1
  states = [not_seen, seen_once]
  alphabet = [0, 1]

  transitions = [
    if i in constrained_sites
      Dict{Tuple{Int,Int},Int}(
        (q, a) => (a == target ? seen_once : q)
        for q in states, a in alphabet
        if !(q == seen_once && a == target)
      )
    else
      Dict{Tuple{Int,Int},Int}((q, a) => q for q in states, a in alphabet)
    end
    for i in eachindex(sites)
  ]

  return DFA(states, alphabet, not_seen, Set([seen_once]), transitions)
end

function constraint_to_dfa(constraint::RelationConstraint, sites)
  validate_projection_sites(sites)

  cs = constraint_sites(constraint)
  validate_constraint_site_bounds(cs, sites)

  (; left_site, relation, right_site) = constraint
  first_site    = min(left_site, right_site)
  second_site   = max(left_site, right_site)
  left_is_first = left_site == first_site

  states    = [0, 1]
  alphabet  = [0, 1]
  initial   = 1
  accepting = Set(states)

  transitions = [
    if i == first_site
      Dict{Tuple{Int,Int},Int}((q, a) => a for q in states, a in alphabet)
    elseif i == second_site
      Dict{Tuple{Int,Int},Int}(
        (q, a) => q
        for q in states, a in alphabet
        if left_is_first ?
          relation_holds(q, relation, a) :
          relation_holds(a, relation, q)
      )
    else
      Dict{Tuple{Int,Int},Int}((q, a) => q for q in states, a in alphabet)
    end
    for i in eachindex(sites)
  ]

  return DFA(states, alphabet, initial, accepting, transitions)
end
