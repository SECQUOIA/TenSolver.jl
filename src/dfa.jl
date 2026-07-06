import ITensors
import ITensorMPS

"""
    DFA

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
    for table in transition_vec
      for ((state, symbol), next_state) in table
        state in state_set ||
          throw(ArgumentError("transition table contains an unknown source state"))
        symbol in alphabet_set ||
          throw(ArgumentError("transition table contains a symbol outside the alphabet"))
        next_state in state_set ||
          throw(ArgumentError("transition table contains an unknown target state"))
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

function dfa_nonzero_tensor(::Type{T}, inds, nonzeros) where {T}
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

function dfa_tensor_indices(site, left_link, right_link)
  return filter(!isnothing, (left_link, site, ITensors.prime(site), right_link))
end

function dfa_coordinate(symbol_pos::Integer, left_pos, right_pos, left_link, right_link)
  coordinate = Int[]
  !isnothing(left_link) && push!(coordinate, left_pos)
  push!(coordinate, symbol_pos, symbol_pos)
  !isnothing(right_link) && push!(coordinate, right_pos)
  return coordinate
end

function validate_dfa_sites(sites, alphabet_len::Integer)
  isempty(sites) && throw(ArgumentError("sites must not be empty"))
  all(site -> ITensors.dim(site) == alphabet_len, sites) ||
    throw(DimensionMismatch("each site dimension must match the DFA alphabet size"))
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
  inds = dfa_tensor_indices(site, left_link, right_link)
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
          (dfa_coordinate(symbol_pos, left_pos, nothing, left_link, right_link), one(T)),
        )
      else
        right_pos = state_positions[next_state]
        push!(
          nonzeros,
          (dfa_coordinate(symbol_pos, left_pos, right_pos, left_link, right_link), one(T)),
        )
      end
    end
  end

  return dfa_nonzero_tensor(T, inds, nonzeros)
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
