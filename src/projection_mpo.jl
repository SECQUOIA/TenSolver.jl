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

function DFA(; states, alphabet, initial, accepting, transitions)
  return DFA(states, alphabet, initial, accepting, transitions)
end

function permute_dfa!(dfa::DFA, permutation::AbstractVector{<:Integer})
  if length(permutation) != length(dfa.transitions)
    throw(DimensionMismatch("DFA permutation length must match the number of transition tables"))
  end

  permute!(dfa.transitions, permutation)
  return dfa
end

function validate_dfa_sites(dfa::DFA, sites)
  if any(site -> ITensors.dim(site) != length(dfa.alphabet), sites)
    throw(DimensionMismatch("each site dimension must match the DFA alphabet size"))
  end
end

"""
    dfa_to_mpo([T], dfa, sites)

Build an exact diagonal projection MPO from a step-dependent DFA.

The physical index basis positions are matched against `dfa.alphabet` in order:
`alphabet[k]` corresponds to local basis state `k`.

The MPO bond dimension equals `length(dfa.states)`.
"""
function dfa_to_mpo(::Type{T}, dfa::DFA, sites) where T
  validate_dfa_sites(dfa, sites)
  tensors = transition_tensors(T, dfa)
  return arrays_to_itensor_mpo( tensors, sites)
end


# Turn a stepwise DFA into a sequence of 3-tensors or 4-tensors
# representing its transition matrices.
function transition_tensors(::Type{T}, dfa::DFA) where T
  (; states, alphabet, transitions, initial, accepting) = dfa

  # initial -> states -> states -> ... -> states -> accepting
  sources(i) = i == firstindex(transitions) ? (initial,)   : states
  targets(i) = i == lastindex(transitions)  ? (accepting,) : tuple.(states)

  # Turn a 1xkxnxn or kx1xnxn tensor into a kxnxn tensor (used on the boundaries)
  proper_shape(A) = dropdims(A; dims = Tuple(filter(d -> size(A, d) == 1, (1, 2))))

  return [
    proper_shape(T[
      a == b && haskey(transitions[i], (s, a)) && transitions[i][(s, a)] in ts
      for s  in sources(i),
          ts in targets(i),
          a  in alphabet,
          b  in alphabet
    ])
    for i in eachindex(transitions)
  ]
end

# Turn a homebrew MPO into an appropriate ITensor.
# This is the only bridge between ITensor and this module.
function arrays_to_itensor_mpo(arrays, sites)
  links = [
    ITensors.Index(size(A, 1), "Link,l=$i")
    for (i, A) in pairs(arrays) if i != lastindex(arrays)
  ]
  wires(i) = filter(!isnothing, (get(links, i-1, nothing), get(links, i, nothing), sites[i]', sites[i]))
  itensors = [ ITensors.itensor(A, wires(i)...) for (i, A) in pairs(arrays) ]

  return ITensorMPS.truncate!(ITensorMPS.MPO(itensors); cutoff = eps(real(eltype(first(arrays)))))
end

"""
    projection_mpo([T], constraint, sites; domain)

Build a projection MPO representing a `constraint` applicable to any MPS over `sites`.
Constraint site numbers must use the same 1-based register indexing as `sites`.

# Known constraints

- [`SumConstraint`](@ref) uses a exact integer partial-sum automaton.
  For a constraint with rhs `k`, its maximum bond dimension is `k+2`.
- [`SumModConstraint`](@ref) uses a modular partial-sum automaton.
  Its `m` residue states give it bond dimension `m`.
- [`NotEqualsConstraint`](@ref) uses a MPO with bond dimension `2`,
  independently of the rhs.
- [`AssignmentConstraint`](@ref) uses a membership counting automaton.
  For rhs `k`, the maximum bond dimension is `k+2`.
- [`RelationConstraint`](@ref) uses a MPO with bond dimension `2`,
  independently of the compared site positions.
"""
function projection_mpo end


function projection_mpo(::Type{T}
                       , constraint::AbstractConstraint
                       , sites
                       ; permutation = 1:length(sites)
                       , domain) where {T}
  dfa = constraint_to_dfa(constraint, length(sites), domain)
  dfa_perm = permute_dfa!(dfa, permutation)
  return dfa_to_mpo(T, dfa_perm, sites)
end

projection_mpo(constraint::AbstractConstraint, sites; kws...) =
  projection_mpo(Float64, constraint, sites; kws...)

"""
    projection_mpos([T], constraints, sites; domain)

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
    project_hamiltonian(H, projections; formulation=:commuting, cutoff=1e-8, kwargs...)

Project a Hamiltonian MPO with one or more projection MPOs.

If `Q = P₁ * ⋯ * Pₙ` is the combined projector, the effective Hamiltonian has
the semantics `Q' * H * Q`.

With the default `formulation=:commuting`, `H` and all `Pᵢ` must be mutually
commuting, while each `Pᵢ` must an orthogonal projection (Hermitian and idempotent).
The construction then simplifies to `H * Q`, with bond dimension bounded by the product of `H`'s
links and each projection link. TenSolver's objective and constraint MPOs
satisfy these assumptions because they are diagonal.

Use `formulation=:sandwich` for general, potentially noncommuting MPOs. It
constructs `Q' * H * Q` directly, so each projection link contributes twice to
the bond-dimension bound.
"""
function project_hamiltonian(
  H::ITensorMPS.MPO,
  projections;
  formulation = :commuting,
  kwargs...,
)
  projection_tuple = projection_sequence(projections)
  target_sites     = projection_target_sites(H)
  validate_projection_sequence(target_sites, projection_tuple)

  op = (x, y) -> ITensors.apply(x, y; kwargs...)
  if formulation === :commuting
    # TODO: We should profile and check that this simplification actually speeds up the code.
    return reduce(op, projection_tuple; init = H)
  elseif formulation === :sandwich
    op2(h, p) = op(ITensors.dag(p), op(h, p))
    return reduce(op2, projection_tuple; init = H)
  else
    msg = "formulation must be :commuting or :sandwich; got $(repr(formulation))"
    throw(ArgumentError(msg))
  end
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
  if !all(a -> isinteger(a) && a >= 0, alphabet)
    throw(ArgumentError("SumConstraint only supports nonnegative integer domains."))
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

function constraint_to_dfa(constraint::SumModConstraint{S}, nsites::Integer, alphabet) where {S}
  if !all(isinteger, alphabet)
    throw(ArgumentError("SumModConstraint only supports integer domains."))
  end

  (; weights, rhs) = constraint
  modulus = constraint.mod

  states    = zero(S):(modulus-one(S))
  initial   = zero(S)
  accepting = Set(rhs)

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, nsites)

  for site in constraint_sites(constraint)
    transitions[site] = Dict(
      (q, a) => mod(q + weights[site] * a, modulus)
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

function constraint_to_dfa(constraint::AssignmentConstraint{S}, nsites::Integer, alphabet) where {S}
  (; values, rhs, relation) = constraint
  beyond    = rhs + one(S)

  states    = zero(S):beyond
  initial   = zero(S)
  accepting = Set(q for q in states if relation_holds(q, relation, rhs))

  id_dict = Dict((q, a) => q for q in states for a in alphabet)
  transitions = fill(id_dict, nsites)

  f(_, a) = S(a in values)
  for site in constraint_sites(constraint)
    transitions[site] = Dict(
      (q, a) => min(q + f(site, a), beyond)
      for q in states, a in alphabet
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
