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

Deterministic finite automaton with step-dependent and partial transitions.

Fields:
- `states`: DFA states, used to define the MPO bond dimension.
- `alphabet`: Per-stage DFA alphabet.
- `initial`: start state.
- `accepting`: set of accepting states.
- `transitions`: one transition table per step; each table maps `(state, symbol)` to
  the next state. Missing entries are rejected.
"""
struct DFA{S,A}
  states::Vector{S}
  alphabets::Vector{Vector{A}}
  initial::S
  accepting::Set{S}
  transitions::Vector{Dict{Tuple{S,A},S}}
  function DFA{S,A}(states, alphabets, initial, accepting, transitions) where {S,A}
    accepting = Set{S}(accepting)

    @argcheck (!isempty)(states)
    @argcheck allunique(states)
    @argcheck initial in states
    @argcheck issubset(accepting, states)

    @argcheck (!isempty)(alphabets)
    @argcheck all(!isempty, alphabets)
    @argcheck length(alphabets) == length(transitions)

    @argcheck (!isempty)(transitions)
    let states_set = Set(states)
      for (i, table) in enumerate(transitions), ((s, a), ns) in table
        @argcheck s  in states_set   "transitions[$(i)]: unknown source state"
        @argcheck a  in alphabets[i] "transitions[$(i)]: symbol not in alphabet"
        @argcheck ns in states_set   "transitions[$(i)]: unknown target state"
      end
    end

    return new{S,A}(states, alphabets, initial, accepting, transitions)
  end
end

function DFA(states, alphabets, initial, accepting, transitions)
  S = eltype(states)
  A = eltype(first(alphabets))
  return DFA{S,A}(states, alphabets, initial, accepting, transitions)
end

function DFA(; states, alphabets, initial, accepting, transitions)
  return DFA(states, alphabets, initial, accepting, transitions)
end

alphabet(dfa::DFA, i) = dfa.alphabets[i]
states(dfa::DFA, i)   = dfa.states

"""
    dfa_to_mpo([T], dfa, sites)

Build an exact diagonal projection MPO from a step-dependent DFA.

The MPO bond dimension is at most the number of states.
"""
function dfa_to_mpo(::Type{T}, dfa::DFA, sites) where T
  for (k, site) in pairs(sites)
    @argcheck ITensors.dim(site) == length(alphabet(dfa, k))
  end
  tensors = transition_tensors(T, dfa)
  return arrays_to_itensor_mpo( tensors, sites)
end

# Turn a stepwise DFA into a sequence of 3-tensors or 4-tensors
# representing its transition matrices.
function transition_tensors(::Type{T}, dfa::DFA) where T
  (; transitions, initial, accepting) = dfa

  # initial -> states -> states -> ... -> states -> accepting
  sources(i) = i == firstindex(transitions) ? (initial,)   : states(dfa, i)
  targets(i) = i == lastindex(transitions)  ? (accepting,) : tuple.(states(dfa, i))

  # Turn a 1xkxnxn or kx1xnxn tensor into a kxnxn tensor (used on the boundaries)
  proper_shape(A) = dropdims(A; dims = Tuple(filter(d -> size(A, d) == 1, (1, 2))))

  return [
    proper_shape(T[
      a == b && haskey(transitions[i], (s, a)) && transitions[i][(s, a)] in ts
      for s  in sources(i),
          ts in targets(i),
          a  in alphabet(dfa, i),
          b  in alphabet(dfa, i)
    ])
    for (i, t) in pairs(transitions)
  ]
end

# Turn a homebrew MPO into an appropriate ITensor.
# This is the only bridge between ITensor and this module.
function arrays_to_itensor_mpo(arrays, sites) :: MPO
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
- [`RelationConstraint`](@ref) uses a MPO with bond dimension equal to the first variable's domain size.
"""
function projection_mpo end


function projection_mpo(::Type{T}
                       , constraint::AbstractConstraint
                       , sites
                       ; domain) where {T}
  dfa = constraint_to_dfa(constraint, length(sites), domain)
  return dfa_to_mpo(T, dfa, sites)
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
  return MPO[projection_mpo(T, constraint, sites; kws...) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, sites; kws...) =
  projection_mpos(Float64, constraints, sites; kws...)

"""
    project_hamiltonian(H, projections; formulation=:commuting, cutoff, kwargs...)

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
    project_state(psi, projections; kwargs...)

Apply one or more diagonal projection MPOs to an MPS.

The result has zero amplitude on basis states rejected by any projection,
while keeping the original unprimed site indices
so it can be used as a DMRG input state.
"""
function project_state(psi::ITensorMPS.MPS, projections; kwargs...)
  projection_tuple = projection_sequence(projections)
  target_sites     = projection_target_sites(psi)
  validate_projection_sequence(target_sites, projection_tuple)

  op = (x, y) -> ITensors.apply(x, y; kwargs...)
  return foldr(op, projection_tuple; init = psi)
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
    mapreduce_dfa(f, op, constraint, nsites, alphabet; initial, predicate, states)

Build a DFA by mapping each constrained site symbol through `f` and combining
the result in a state accumulator with `op`.

The function `f` is assumed to take the `states` to a set where `op`
acts as a monoid operation, i.e., its associative and `initial` is the identity element.
The predicate must be a Boolean-valued function deciding whether a state is accepting or not.

This is an internal method encapsulating a common pattern for constraint representation.
"""
function mapreduce_dfa(f, op, constraint, nsites, domains; initial, predicate, states)
  accepting = Set(q for q in states if predicate(q))

  transitions = [Dict((q, a) => q for q in states for a in domains[i]) for i in 1:nsites]

  for i in constraint_sites(constraint)
    transitions[i] = Dict((q, a) => op(i)(q, f(i)(a)) for q in states, a in domains[i])
  end

  S = eltype(states)
  A = eltype(first(domains))
  return DFA{S,A}(states, [domains...], initial, accepting, transitions)
end

"""
    constraint_to_dfa(constraint, n, domain)

Build a [`DFA`](@ref) recognizing `constraint` with transitions for `n` steps.
The `domain` parameter represents the (finite) domain for each of the `n` variables.
"""
function constraint_to_dfa end

function constraint_to_dfa(constraint::SumConstraint{S}, nsites::Integer, domains::Domains) where {S}
  for domain in domains
    @argcheck all(isinteger, domain)
    @argcheck all(>=(0), domain)
  end

  (; weights, rhs, relation) = constraint
  beyond = rhs + one(S)

  return mapreduce_dfa(
    i -> a -> weights[i] * S(a),
    i -> (x, y) -> min(x + y, beyond),
    constraint,
    nsites,
    domains,
    ;
    states    = zero(S):beyond,
    initial   = zero(S),
    predicate = q -> relation_holds(q, relation, rhs),
  )
end

function constraint_to_dfa(constraint::SumModConstraint{S}, nsites::Integer, domains) where {S}
  for domain in domains
    @argcheck all(isinteger, domain)
  end

  (; weights, rhs) = constraint
  modulus = constraint.mod

  return mapreduce_dfa(
    i -> a -> mod(weights[i] * a, modulus),
    i -> (x, y) -> mod(x + y, modulus),
    constraint,
    nsites,
    domains,
    ;
    states    = zero(S):(modulus-one(S)),
    initial   = zero(S),
    predicate = ==(rhs),
  )
end

function constraint_to_dfa(constraint::NotEqualsConstraint{S}, nsites::Integer, domains) where {S}
  (; values) = constraint

  return mapreduce_dfa(
    i -> a -> S(a) != values[i],
    i -> (|),
    constraint,
    nsites,
    domains,
    ;
    states    = Bool[0, 1],
    initial   = false,
    predicate = identity,
  )
end

function constraint_to_dfa(constraint::AssignmentConstraint{S}, nsites::Integer, domains) where {S}
  (; values, rhs, relation) = constraint
  beyond = rhs + 1

  return mapreduce_dfa(
    i -> in(values),
    i -> (x, y) -> min(x + y, beyond),
    constraint,
    nsites,
    domains,
    ;
    states    = 0:beyond,
    initial   = 0,
    predicate = q -> relation_holds(q, relation, rhs),
  )
end

function constraint_to_dfa(constraint::RelationConstraint, nsites::Integer, domains)
  # Assumes left_site < right_site, as enforced by RelationConstraint
  (; left_site, right_site, relation) = constraint

  states    = domains[left_site]
  initial   = last(states)
  accepting = Set(states)

  transitions = [Dict((q, a) => q for q in states for a in domains[k]) for k in 1:nsites]

  transitions[left_site] = Dict((q, a) => a for q in states, a in domains[left_site])

  transitions[right_site] = Dict(
    (q, a) => q
    for q in states, a in domains[right_site]
    if relation_holds(q, constraint.relation, a)
  )

  return DFA{eltype(states), eltype(states)}(states, [domains...], initial, accepting, transitions)
end
