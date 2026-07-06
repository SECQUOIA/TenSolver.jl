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
- `initial_state`: start state.
- `accepting_states`: set of accepting states.
- `transitions`: one transition table per site; each table maps `(state, symbol)` to
  the next state. Missing entries are rejected.
"""
struct DFA{S,A}
  states::Vector{S}
  alphabet::Vector{A}
  initial_state::S
  accepting_states::Set{S}
  transitions::Vector{Dict{Tuple{S,A},S}}

  function DFA(
    states,
    alphabet,
    initial_state,
    accepting_states,
    transitions,
  )
    state_vec = collect(states)
    alphabet_vec = collect(alphabet)
    transition_vec = collect(transitions)

    isempty(state_vec) && throw(ArgumentError("states must not be empty"))
    isempty(alphabet_vec) && throw(ArgumentError("alphabet must not be empty"))
    isempty(transition_vec) && throw(ArgumentError("transitions must not be empty"))

    length(unique(state_vec)) == length(state_vec) ||
      throw(ArgumentError("states must be unique"))

    state_set = Set(state_vec)
    initial_state in state_set ||
      throw(ArgumentError("initial_state must be one of the DFA states"))

    accepting_set = Set(collect(accepting_states))
    all(state -> state in state_set, accepting_set) ||
      throw(ArgumentError("accepting_states must be a subset of states"))

    alphabet_set = Set(alphabet_vec)
    for (table_idx, table) in enumerate(transition_vec)
      for ((state, symbol), next_state) in table
        state in state_set ||
          throw(ArgumentError("transition table $(table_idx): unknown source state $(repr(state))"))
        symbol in alphabet_set ||
          throw(ArgumentError("transition table $(table_idx): symbol $(repr(symbol)) is not in the alphabet"))
        next_state in state_set ||
          throw(ArgumentError("transition table $(table_idx): unknown target state $(repr(next_state))"))
      end
    end

    return new{eltype(state_vec),eltype(alphabet_vec)}(
      state_vec,
      alphabet_vec,
      initial_state,
      accepting_set,
      transition_vec,
    )
  end
end

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

  source_states = isnothing(left_link) ? (dfa.initial_state,) : dfa.states
  for source_state in source_states
    left_pos = isnothing(left_link) ? nothing : state_positions[source_state]

    for (symbol_pos, symbol) in enumerate(dfa.alphabet)
      next_state = get(table, (source_state, symbol), nothing)
      isnothing(next_state) && continue

      if isnothing(right_link)
        next_state in dfa.accepting_states || continue
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

function validate_dfa_sites(sites, alphabet_len::Integer)
  isempty(sites) && throw(ArgumentError("sites must not be empty"))
  all(site -> ITensors.dim(site) == alphabet_len, sites) ||
    throw(DimensionMismatch("each site dimension must match the DFA alphabet size"))
end

"""
    dfa_to_mpo([T], dfa, sites)

Build an exact diagonal projection MPO from a step-dependent DFA.

The physical index basis positions are matched against `dfa.alphabet` in order:
`alphabet[k]` corresponds to local basis state `k`.

The MPO bond dimension equals `length(dfa.states)`.
"""
function dfa_to_mpo(::Type{T}, dfa::DFA, sites) where {T}
  site_vec = collect(sites)
  length(site_vec) == length(dfa.transitions) ||
    throw(DimensionMismatch("DFA transition tables must match the number of sites"))

  validate_dfa_sites(site_vec, length(dfa.alphabet))

  state_positions = Dict(state => i for (i, state) in enumerate(dfa.states))
  links = [
    ITensors.Index(length(dfa.states), "Link,DFA,l=$i")
    for i in 1:max(length(site_vec) - 1, 0)
  ]

  tensors = Vector{ITensors.ITensor}(undef, length(site_vec))
  for site_position in eachindex(site_vec)
    left_link = site_position == firstindex(site_vec) ? nothing : links[site_position - 1]
    right_link = site_position == lastindex(site_vec) ? nothing : links[site_position]

    tensors[site_position] = dfa_site_tensor(
      T,
      site_vec[site_position],
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

- `SumConstraint` uses a specialized exact integer partial-sum automaton.
Its weights and right-hand side must be nonnegative and from an [`Integer`](@ref) type.
For a constraint with rhs `k`, its maximum bond dimension is `k`.
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

function constraint_to_dfa(constraint::SumConstraint{S}, sites) where {S<:Integer}
  (; weights, rhs, relation) = constraint

  overflow  = rhs + one(S)
  states    = collect(zero(S):overflow)
  alphabet  = [0, 1]
  accepting = Set(q for q in states if relation_holds(q, relation, rhs))

  transitions = [
    let weight = get(weights, i, zero(S))
      Dict(
        (q, a) => min(q + weight * S(a), overflow)
        for q in states, a in alphabet
      )
    end
    for i in eachindex(sites)
  ]

  return DFA(states, alphabet, zero(S), accepting, transitions)
end
