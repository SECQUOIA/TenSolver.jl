# Adapted with permission from Filip B. Maciejewski's QAOA warm-start preparation.
# Copyright 2025 USRA. Source: fbm2718-usra/quantum-approximate-optimization,
# revision 2d6296206ec7955f3a24392d05e7225860043c89, jl/ws_qaoa.jl.

"""
    TenSolver.QAOAInitialState(gamma, beta)

Prepare a depth-one QAOA state before DMRG for an unconstrained matrix or linear
objective with `domain = [-1, 1]`. Starting from the uniform superposition, apply
`exp(-im * gamma * H)` followed by `exp(-im * beta * sum(X))`, where `H` is the
objective Hamiltonian. Both angles must be finite real numbers, in radians.
The constant objective offset contributes only a global phase and is omitted
from preparation; it remains included in reported objective values.

Pass this configuration as `initial_state` to [`minimize`](@ref) or
[`maximize`](@ref). For maximization, the cost Hamiltonian is the *negated*
objective. `inidim` caps the preparation's bond dimension; `cutoff` controls
coefficient filtering and gate truncation. The state is normalized after
preparation. DMRG's `maxdim` applies afterward.
Angles are supplied by the caller; this configuration does not optimize them.
"""
struct QAOAInitialState
  gamma::Float64
  beta::Float64

  function QAOAInitialState(gamma::Real, beta::Real)
    angles = (Float64(gamma), Float64(beta))
    if !(all(isfinite, angles))
      throw(ArgumentError("QAOA angles must be finite."))
    end
    return new(angles...)
  end
end

dmrg_initializer(::Nothing, Q, l; domain, constraints) = constrained_initial_state

function dmrg_initializer(initial_state, Q, l; domain, constraints)
  return throw(ArgumentError("initial_state must be nothing or a QAOAInitialState."))
end

function dmrg_initializer(initial_state::QAOAInitialState, Q, l; domain, constraints)
  if !(all(d -> d == [-1, 1], domain))
    throw(ArgumentError("QAOA initialization requires domain = [-1, 1]."))
  end
  if !(isempty(constraints))
    throw(ArgumentError("QAOA initialization requires an unconstrained objective."))
  end
  if !(all(isfinite, Q) && all(isfinite, l))
    throw(ArgumentError("QAOA initialization requires finite coefficients."))
  end
  return (T, sites, projections; cutoff, inidim) ->
    qaoa_initial_state(initial_state, Q, l, sites; cutoff, maxdim = inidim)
end

function qaoa_phase_gate(gamma, weight, spin)
  angle = gamma * weight
  if !(isfinite(angle))
    throw(ArgumentError("QAOA cost angles must be finite."))
  end
  return exp(-im * angle * spin)
end

function qaoa_initial_state(initial_state::QAOAInitialState, Q, l, sites; cutoff, maxdim)
  @argcheck maxdim isa Integer && maxdim >= 1
  @argcheck isfinite(cutoff) && cutoff >= 0
  gamma, beta = initial_state.gamma, initial_state.beta
  # TenSolver orders spin values as [-1, +1], so its D operator is -PauliZ.
  spin = [ITensors.op("D", s; domain = [-1.0, 1.0]) for s in sites]
  gates = ITensors.ITensor[]
  for i in eachindex(sites)
    abs(l[i]) > cutoff && push!(gates, qaoa_phase_gate(gamma, l[i], spin[i]))
  end
  for j in eachindex(sites), i in 1:(j - 1)
    weight = Q[i, j] + Q[j, i]
    abs(weight) > cutoff && push!(gates, qaoa_phase_gate(gamma, weight, spin[i] * spin[j]))
  end
  mixer = ComplexF64[cos(beta) -im*sin(beta); -im*sin(beta) cos(beta)]
  for s in sites
    push!(gates, ITensors.ITensor(mixer, ITensors.prime(s), s))
  end
  psi = MPS([ITensors.ITensor(ComplexF64[1, 1] / sqrt(2), s) for s in sites])
  psi = ITensorMPS.apply(gates, psi; cutoff, maxdim)
  normalize!(psi)
  return psi
end
