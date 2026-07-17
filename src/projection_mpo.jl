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

function permute_dfa!(dfa::DFA, permutation::AbstractVector{<:Integer})
  length(permutation) == length(dfa.transitions) ||
    throw(DimensionMismatch("DFA permutation length must match the number of transition tables"))

  reordered = dfa.transitions[permutation]
  for i in eachindex(reordered)
    dfa.transitions[i] = reordered[i]
  end

  return dfa
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

Build a projection MPO representing a `constraint` applicable to any MPS over `sites`.

The diagonal entry is `one(T)` for computational basis states
satisfying `constraint`, and `zero(T)` otherwise.
The current construction is exact and uncompressed.
Constraint site numbers use the same 1-based register indexing as `sites`.

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


function projection_mpo(::Type{T}
                       , constraint::AbstractConstraint
                       , sites
                       ; permutation = 1:length(sites)) where {T}
  d = ITensorMPS.dim(sites[1])
  domain = 0:(d - 1)
  dfa = constraint_to_dfa(constraint, length(sites), domain)
  dfa_perm = permute_dfa!(dfa, permutation)
  return dfa_to_mpo(T, dfa_perm, sites)
end

projection_mpo(constraint::AbstractConstraint, sites; kws...) =
  projection_mpo(Float64, constraint, sites; kws...)

"""
    projection_mpos([T], constraints, sites)

Build a list of projection MPOs representing  `constraints` applicable to any MPS over `sites`.

This is a convenience wrapper around [`projection_mpo`](@ref).
`T` controls the numeric element type of the assembled MPO tensors.
"""
function projection_mpos(::Type{T}, constraints::AbstractVector{<:AbstractConstraint}, sites; kws...) where {T}
  return [projection_mpo(T, constraint, sites; kws...) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, sites; kws...) =
  projection_mpos(Float64, constraints, sites; kws...)

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
# Constraint to DFA
##############################################

"""
    constraint_to_dfa(constraint, n, alphabet)

Build a [`DFA`](@ref) recognizing `constraint` with transitions for `n` steps.
The `alphabet` parameter represents the domain for a constraint's variables.
"""
function constraint_to_dfa end

function constraint_to_dfa(constraint::SumConstraint{S}, nsites::Integer, alphabet) where {S}
  if minimum(alphabet) < 0
    throw(ArgumentError("SumConstraint only supports nonnegative domains."))
  end

  (; weights, rhs, relation) = constraint
  beyond    = rhs + one(S)

  states    = zero(S):beyond
  initial   = zero(S)
  accepting = Set(q for q in states if relation_holds(q, relation, rhs))

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, nsites)

  for site in constraint_sites(constraint)
    transitions[site] = Dict(
      (q, a) => min(q + weights[site] * a, beyond)
      for q in states, a in alphabet
    )
  end

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::NotEqualsConstraint{S}, nsites::Integer, alphabet) where {S}
  states    = [:mismatch, :all_matched]
  initial   = :all_matched
  accepting = Set([:mismatch])

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, nsites)

  for site in constraint_sites(constraint)
    target = constraint.values[site]

    transitions[site] = Dict(
      (q, a) => S(a) == target ? q : :mismatch
      for q in states, a in alphabet
    )
  end

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::ExactlyOneConstraint{S}, nsites::Integer, alphabet) where {S}
  target = constraint.value

  states    = [:not_seen, :seen_once]
  initial   = :not_seen
  accepting = Set([:seen_once])

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, nsites)

  for site in constraint_sites(constraint)
    transitions[site] = Dict(
      (q, a) => (a == target ? :seen_once : q)
      for q in states, a in alphabet
      if !(q == :seen_once && a == target)
    )
  end

  return DFA(states, alphabet, initial, accepting, transitions)
end

function constraint_to_dfa(constraint::RelationConstraint, nsites::Integer, alphabet)
  left  = constraint.left_site
  right = constraint.right_site

  first_site    = min(left, right)
  second_site   = max(left, right)
  left_is_first = left == first_site

  states    = alphabet
  initial   = last(states)
  accepting = Set(states)

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, nsites)

  transitions[first_site] = Dict((q, a) => a for q in states, a in alphabet)

  transitions[second_site] = Dict(
    (q, a) => q
    for q in states, a in alphabet
    if left_is_first ? relation_holds(q, constraint.relation, a) :
                       relation_holds(a, constraint.relation, q)
  )

  return DFA(states, alphabet, initial, accepting, transitions)
end
