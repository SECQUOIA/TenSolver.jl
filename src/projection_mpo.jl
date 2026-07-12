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


function projection_mpo(
  ::Type{T},
  constraint::AbstractConstraint,
  sites,
  permutation = 1:length(sites),
) where {T}
  return dfa_to_mpo(T, constraint_to_dfa(constraint, invperm(permutation)), sites)
end

projection_mpo(constraint::AbstractConstraint, args...) =
  projection_mpo(Float64, constraint, args...)

"""
    projection_mpos([T], constraints, sites)

Build one projection MPO per constraint over the shared binary Qudit register
`sites`.

This is a convenience wrapper around [`projection_mpo`](@ref).
`T` controls the numeric
element type of the assembled MPO tensors.
"""
function projection_mpos(::Type{T}, constraints::AbstractVector{<:AbstractConstraint}, args...) where {T}
  return [projection_mpo(T, constraint, args...) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, args...) =
  projection_mpos(Float64, constraints, args...)

"""
    project_hamiltonian(H, projections; cutoff=1e-8, kwargs...)

Apply one or more diagonal projection MPOs to a Hamiltonian MPO.

For projection MPOs `P`, this builds the CoTenN-style effective Hamiltonian
`P' * H * P`. The resulting bond dimension can grow with the product of
`H`'s links and the projection links.
"""
function project_hamiltonian(H::ITensorMPS.MPO, projections; cutoff=1e-8, kwargs...)
  projection_tuple = projection_sequence(projections)
  target_sites = projection_target_sites(H)
  validate_projection_sequence(target_sites, projection_tuple)

  H_eff = H
  for P in projection_tuple
    H_eff = ITensors.apply(ITensors.dag(P), H_eff; cutoff, kwargs...)
    H_eff = ITensors.apply(H_eff, P; cutoff, kwargs...)
  end

  return H_eff
end

"""
    project_state(psi, projections; cutoff=1e-8, kwargs...)

Apply one or more diagonal projection MPOs to an MPS.

The result has zero amplitude on basis states rejected by any projection,
while keeping the original unprimed site indices
so it can be used as a DMRG input state.
"""
function project_state(psi::ITensorMPS.MPS, projections; cutoff=1e-8, kwargs...)
  projection_tuple = projection_sequence(projections)
  target_sites = projection_target_sites(psi)
  validate_projection_sequence(target_sites, projection_tuple)

  projected = psi
  for P in projection_tuple
    projected = ITensors.apply(P, projected; cutoff, kwargs...)
  end

  return projected
end

projection_sequence(projection::ITensorMPS.MPO) = (projection,)
projection_sequence(projections::Tuple{Vararg{ITensorMPS.MPO}}) = projections
projection_sequence(projections::AbstractVector{<:ITensorMPS.MPO}) = Tuple(projections)

projection_target_sites(H::ITensorMPS.MPO) = ITensorMPS.siteinds(first, H; plev=0)
projection_target_sites(psi::ITensorMPS.MPS) = ITensorMPS.siteinds(psi)

function validate_projection_sequence(target_sites, projections)
  for (i, P) in enumerate(projections)
    if length(P) != length(target_sites)
      msg = "projection MPO $(i) has length $(length(P)); expected $(length(target_sites))"
      throw(DimensionMismatch(msg))
    end

    projection_sites = ITensorMPS.siteinds(first, P; plev=0)
    if projection_sites != target_sites
      msg = "projection MPO $(i) must share the target's unprimed site indices"
      throw(DimensionMismatch(msg))
    end
  end
end


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

function default_constraint_to_dfa(constraint::AbstractConstraint, sites)
  validate_projection_sites(sites)

  cs = constraint_sites(constraint)
  validate_constraint_site_bounds(cs, sites)

  return projection_entries_to_dfa(
    projection_entries(Bool, constraint),
    length(sites),
    cs,
  )
end

##############################################
# Constraint to DFA
##############################################

"""
    constraint_to_dfa(constraint, indices)

Build a DFA for `constraint` over `sites`.

The argument `indices` is a vector describing the mapping
variable indices, as specified in the constraints, to tensor site indices.
"""
function constraint_to_dfa end

function constraint_to_dfa(constraint::AbstractConstraint, nsites::Integer)
  return constraint_to_dfa(constraint, collect(1:nsites))
end

function constraint_to_dfa(constraint::SumConstraint{S}, indices::Vector{Int}) where {S}
  (; weights, rhs, relation) = constraint

  beyond    = rhs + one(S)
  states    = zero(S):beyond
  alphabet  = S[0, 1]
  initial   = zero(S)
  accepting = Set(q for q in states if relation_holds(q, relation, rhs))

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, length(indices))

  for site in constraint_sites(constraint)
    tensor_site = indices[site]
    weight = weights[site]

    transitions[tensor_site] = Dict(
      (q, a) => min(q + weight * a, beyond)
      for q in states, a in alphabet
    )
  end

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::NotEqualsConstraint{S}, indices::Vector{Int}) where {S}
  states    = S[0, 1]
  alphabet  = S[0, 1]
  initial   = one(S)
  accepting = Set([zero(S)])

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, length(indices))

  for site in constraint_sites(constraint)
    tensor_site = indices[site]
    target = constraint.values[site]

    transitions[tensor_site] = Dict(
      (q, a) => S(a) == target ? q : zero(S)
      for q in states, a in alphabet
    )
  end

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::ExactlyOneConstraint{S}, indices::Vector{Int}) where {S}
  target = constraint.value
  not_seen  = zero(S)
  seen_once = one(S)

  states    = S[not_seen, seen_once]
  alphabet  = S[0, 1]
  initial   = not_seen
  accepting = Set([seen_once])

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, length(indices))

  for site in constraint_sites(constraint)
    tensor_site = indices[site]
    transitions[tensor_site] = Dict(
      (q, a) => (a == target ? seen_once : q)
      for q in states, a in alphabet
      if !(q == seen_once && a == target)
    )
  end

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::RelationConstraint, indices::Vector{Int})
  left  = indices[constraint.left_site]
  right = indices[constraint.right_site]
  first_site    = min(left, right)
  second_site   = max(left, right)
  left_is_first = left == first_site

  states    = [0, 1]
  alphabet  = [0, 1]
  initial   = 1
  accepting = Set(states)

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, length(indices))

  transitions[first_site] = Dict(
    (q, a) => a for q in states, a in alphabet
  )

  transitions[second_site] = Dict(
    (q, a) => q
    for q in states, a in alphabet
    if left_is_first ? relation_holds(q, constraint.relation, a) :
                       relation_holds(a, constraint.relation, q)
  )

  return DFA(states, alphabet, initial, accepting, transitions)
end
