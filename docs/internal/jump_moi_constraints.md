# JuMP/MOI Constrained Optimization Architecture

This note defines how a future JuMP/MathOptInterface (MOI) front end should
lower constrained binary models into TenSolver's native projection-constraint
pipeline. It is an implementation design, not a description of behavior in the
current release. In particular, `TenSolver.Optimizer` still accepts
QUBO models and fixed binary variables, but not the general hard constraints
described here.

The central decision is to keep two layers with different responsibilities:

- MOI `function-in-set` constraints are the high-level modeling interface used
  by JuMP.
- TenSolver's native [`AbstractConstraint`](@ref) subtypes remain the small,
  projection-aware intermediate representation (IR) consumed by the solver.

The MOI layer must lower into the native IR. It must not teach the projection
code to dispatch on arbitrary MOI sets, and it must not turn supported hard
constraints into penalty terms.

## Current Boundary

The direct Julia API already accepts native constraints through the
`constraints` keyword of [`minimize`](@ref) and [`maximize`](@ref). The DMRG
backend lowers each native constraint with `constraint_to_dfa`, constructs its
projection MPO, and solves in the projected space. The four v1 native types are:

- [`SumConstraint`](@ref);
- [`NotEqualsConstraint`](@ref);
- [`ExactlyOneConstraint`](@ref); and
- [`RelationConstraint`](@ref).

The JuMP path stops earlier. `TenSolver.Optimizer` is generated with
`QUBODrivers.@setup`, and QUBODrivers defines an `AbstractSampler` as an
unconstrained QUBO/Ising optimizer, apart from variable-domain and fixed-value
constraints. Its generic `MOI.supports_constraint` fallback rejects other
constraint types. `QUBODrivers.sample` in TenSolver then extracts only QUBO
objective data with `QUBOTools.qubo` and calls `minimize` without a
`constraints` argument. Consequently, JuMP rejects an affine hard constraint
before `optimize!` reaches TenSolver.

This split is useful: QUBODrivers should continue owning QUBO objective storage,
sampler attributes, samples, and common MOI result plumbing. TenSolver should
own the additional hard-constraint store, MOI-to-native normalization, and
constraint-aware status overrides.

## Public API Decision

`TenSolver.Optimizer` remains the only JuMP optimizer type. A separate
`ConstrainedOptimizer` would fragment model code and force users to select an
optimizer based on whether a model happens to contain constraints.

The native constraint constructors remain exported as the low-level direct
Julia API. They are not a second general-purpose modeling language: the native
taxonomy stays deliberately small and represents projection strategies with
known feasibility and cost behavior. JuMP/MOI is the preferred high-level
modeling surface, while direct users may continue to pass the native types to
`minimize` and `maximize`.

Both layers therefore remain public, but only the MOI layer should grow
modeling conveniences and bridges. New native types should be added only when
they define a useful projection target, normally with a specialized DFA.

## Model-Ingestion Flow

The constrained optimizer should preserve QUBODrivers' non-incremental
`MOI.copy_to` architecture and specialize it for `TenSolver.Optimizer`:

1. JuMP stores the source model in its normal MOI caching optimizer.
2. TenSolver's concrete `MOI.copy_to(::Optimizer, src)` inventories domain,
   fixed-variable, objective, and supported hard-constraint data.
3. A `MOI.Utilities.ModelFilter` hides only the hard constraints that TenSolver
   will lower. It must retain `VariableIndex`-in-`ZeroOne`, fixed-variable
   `VariableIndex`-in-`EqualTo`, the objective, and relevant model and variable
   attributes.
4. The filtered view is materialized into a fresh `MOI.Utilities.Model{T}`.
   This extra copy is required because `ModelFilter` filters query results but
   does not itself advertise `supports_constraint`; QUBODrivers uses that query
   to identify the binary domain. TenSolver records the source-to-filtered
   `MOI.IndexMap`.
5. The materialized model is passed to the QUBODrivers `AbstractSampler` copy
   path, which keeps ownership of QUBO construction and fixed-variable
   objective substitution. TenSolver composes this second index map with the
   first.
6. TenSolver substitutes those same fixed values into each hard constraint,
   simplifies the result, lowers it to native constraints, and maps every
   remaining source `MOI.VariableIndex` through the composed map to the
   free-variable site returned by `QUBOTools.index`.
7. The native constraints are stored in an unexported, typed optimizer
   attribute. `MOI.empty!` resets that store along with the QUBODrivers model.
8. `QUBODrivers.sample(::Optimizer)` reads the stored native constraints and
   passes them to `minimize(...; constraints)`.

This plan uses public MOI filtering and QUBODrivers dispatch rather than calling
QUBODrivers' private objective-copy helpers. Access to the internal store should
be isolated behind TenSolver helpers and covered by a QUBODrivers conformance
test. If the generated optimizer cannot host an unexported typed attribute
without relying on undocumented fields, the prerequisite is a small
QUBODrivers extension hook; the fallback is not a global side table keyed by
optimizer identity.

The optimizer should also retain the original MOI functions and sets needed to
answer `NumberOfConstraints`, `ListOfConstraintIndices`, `ConstraintFunction`,
`ConstraintSet`, and `is_valid`. The native constraints are execution data, not
a substitute for the MOI model's queryable representation. Its returned
`MOI.IndexMap` must include the separately stored hard-constraint indices as
well as the objective/domain indices copied by QUBODrivers.

### Fixed variables and static simplification

QUBODrivers can remove fixed variables from the sampled QUBO. Constraint
lowering must therefore substitute fixed values before assigning native site
numbers. For example, fixing `x[1] = 1` turns `x[1] + x[2] <= 1` into
`x[2] <= 0`; referring to the old first site after QUBODrivers removes it would
constrain the wrong variable.

After substitution, lowering produces one of three results:

- a tautology, which is dropped;
- one or more native constraints over free sites; or
- a static contradiction, which marks the solve infeasible without running
  DMRG.

A static contradiction is a solver outcome, not a model-construction error. It
must use the same infeasibility status path as an all-zero projection.

## Canonicalization Rules

For a scalar affine function

```math
f(x) = c + \sum_i a_i x_i,
```

the lowerer first combines duplicate terms, removes zero coefficients, and
moves `c` into the set bound. For example,
`f(x) in MOI.LessThan(u)` becomes `sum(a[i] * x[i]) <= u - c`.
`MOI.GreaterThan`, `MOI.EqualTo`, and both sides of `MOI.Interval` are handled
analogously.

The v1 native sum projection accepts nonnegative integer weights and a
nonnegative integer right-hand side. The MOI bridge must preserve that
contract:

- coefficients and normalized bounds must be exactly integer-valued; the
  bridge must not round approximately integral floating-point data;
- an all-negative expression may be multiplied by `-1`, reversing `<=` and
  `>=`;
- mixed-sign expressions are accepted only when they match a specialized
  pairwise-relation pattern; and
- strict `<` and `>` constraints are not accepted. A caller that relies on an
  integer domain must express the equivalent shifted non-strict bound
  explicitly.

Specialized compact patterns are selected before the general sum fallback.
This is part of correctness for the performance contract, not a cosmetic
optimization.

## Initial Mapping Table

All rows below require `ZeroOne` variables. “Bond” is the expected projection
MPO bond bound after lowering, based on the current native DFA methods.

| MOI/JuMP form | Native lowering | Bond | Conditions and notes |
|:--------------|:----------------|:-----|:---------------------|
| `ScalarAffineFunction`-in-`EqualTo(1)`, all unit coefficients | `ExactlyOneConstraint(sites, 1)` | 2 | Prefer this over a general sum. |
| `sum(x) == length(x) - 1` | `ExactlyOneConstraint(sites, 0)` | 2 | Exactly one selected site is zero. |
| `x[i] + x[j] <= 1` | `NotEqualsConstraint([i, j], [1, 1])` | 2 | Compact pairwise exclusion. |
| Nonnegative integer affine function in `EqualTo`, `LessThan`, or `GreaterThan` | `SumConstraint` with `==`, `<=`, or `>=` | `rhs + 2` | Constants are moved into `rhs`; reject invalid normalized data. |
| Nonnegative integer affine function in `Interval` | Two `SumConstraint`s | Product of the two projection costs | Lower the lower and upper bounds separately; drop a redundant infinite side. |
| `x[i] - x[j]` in `EqualTo(0)`, `LessThan(0)`, or `GreaterThan(0)` | `RelationConstraint(i, relation, j)` | 2 | Covers equality, implication-style `<=`, and `>=`. |
| `VectorOfVariables([x[i], x[j]])` in `MOI.AllDifferent(2)` | `RelationConstraint(i, :(!=), j)` | 2 | `AllDifferent` over more than two binary variables is statically infeasible. |
| `VectorOfVariables(x)` in `MOI.SOS1(weights)` | `SumConstraint(sites, ones, 1; relation = :(<=))` | 3 | `SOS1` means *at most* one nonzero, not exactly one; ordering weights do not change feasibility. |
| `ScalarAffineFunction` in a TenSolver `NotEqualTo(value)` set | `SumConstraint(...; relation = :(!=))`, or `RelationConstraint` for a pairwise pattern | Native target's bound | MOI has no standard scalar not-equal set. |
| `VectorOfVariables(x)` in a TenSolver `ForbiddenAssignment(values)` set | `NotEqualsConstraint(sites, values)` | 2 | Compact custom set for excluding one bit pattern. |
| `VariableIndex` in `EqualTo(0 or 1)` | Existing QUBODrivers fixed-variable path | No projector | Substitute before hard-constraint lowering. |

Exactly-one should normally be modeled as `sum(x) == 1`. Combining that
equality with `SOS1` is redundant and would create a second projector. A bare
`SOS1` cannot lower to `ExactlyOneConstraint` because the all-zero assignment
is feasible for `SOS1`.

The custom `NotEqualTo` and `ForbiddenAssignment` sets belong to the MOI front
end. They do not replace `SumConstraint` or `NotEqualsConstraint`; they provide
standard function-in-set syntax for native targets that have no lossless
off-the-shelf MOI set.

## Point-on-Set Versus Projection MPOs

MOI's function-in-set vocabulary is the right abstraction for declaring the
feasible set. It should not be confused with the operator that TenSolver builds.

`MOI.Utilities.distance_to_set` and the separate
[MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl)
package operate on numeric points and MOI sets. They can support validation or
diagnostics, but they do not construct TenSolver's projection MPO. TenSolver's
`P` is a Hilbert-space operator that is diagonal in the binary computational
basis, with value one on feasible bitstrings and zero on infeasible bitstrings.

Therefore the first bridge needs no new set-distance dependency. The useful
connection is conceptual: the MOI set defines membership, the native constraint
defines a compact automaton for that membership predicate, and the automaton
defines `P`.

This separation also leaves room for future non-diagonal projections, such as
the eigenvalue constraints discussed in the CoTenN work, without pretending
that every projection MPO is a Euclidean point projection.

## Unsupported Constraints and Fallback Policy

The default policy is an explicit hard error. The v1 bridge must not silently:

- convert a hard constraint into a penalty QUBO;
- enumerate every feasible assignment through the legacy generic projection
  path; or
- accept a bridge that introduces continuous, general-integer, or otherwise
  non-binary auxiliary variables.

Penalty conversion changes the optimization problem unless a safe penalty
scale is known. Generic feasible-path enumeration preserves semantics but can
have exponentially larger bonds and is slated for removal in favor of explicit
automata. Neither is an acceptable implicit fallback.

For unsupported function/set *types*, `MOI.supports_constraint` returns false,
allowing MOI to try a registered bridge and otherwise raise
`MOI.UnsupportedConstraint`. For a supported type with unsupported data, such
as mixed-sign general affine coefficients, TenSolver raises
`MOI.AddConstraintNotAllowed{F,S}` with the exact violated restriction and a
suggested supported formulation.

The native layer should gain catchable `InvalidConstraintError` and
`ConstraintUnsupportedByBackend` errors in the implementation epic. The MOI
adapter translates validation failures during model copy into MOI's typed
construction errors. Backend incompatibility remains a typed solve-time error;
infeasibility never uses either exception.

If an opt-in penalty or generic-exact fallback is added later, it requires a
separate public optimizer attribute, explicit cost warnings, and tests proving
that the default remains hard-error behavior.

## Bridge Policy

TenSolver should advertise direct support for each row in the mapping table so
MOI does not replace a compact pattern with a generic MILP bridge. Other MOI
bridges may be used only when their output consists entirely of supported
binary functions and sets. Every introduced variable must have a `ZeroOne`
domain before the model reaches native lowering.

Bridge output is canonicalized exactly like user-authored constraints. The
bridge boundary cannot bypass integer, nonnegativity, fixed-variable, or
backend checks.

The projection layer remains MOI-agnostic. No method in `projection_mpo.jl`
should accept an MOI function or set.

## Result and Status Contract

The native status behavior is already defined:

- infeasible `minimize` returns `(+Inf, infeasible_solution)`;
- infeasible `maximize` returns `(-Inf, infeasible_solution)`;
- `is_feasible(solution)` is false; and
- sampling an infeasible solution throws.

The MOI layer maps that outcome to:

- `MOI.TerminationStatus() == MOI.INFEASIBLE`;
- `MOI.ResultCount() == 0`;
- `MOI.PrimalStatus() == MOI.NO_SOLUTION`; and
- an empty QUBOTools `SampleSet` with the TenSolver status in metadata.

TenSolver already computes `MOI.INFEASIBLE` in `tensolver_status`, but the
generic QUBODrivers termination-status getter treats a nonempty metadata object
with zero samples as `MOI.OTHER_ERROR`. The constrained implementation must add
a concrete `MOI.get(::Optimizer, ::MOI.TerminationStatus)` override that reads
TenSolver's recorded status. Metadata alone does not complete the MOI contract.

Invalid model data and unsupported backends do not map to `INFEASIBLE`: they
remain typed construction or solve errors. Constraint duals are unsupported.

## Coordination With Related Work

- #37 and #62 are complete. The bridge targets the landed backend-aware
  `minimize`/`maximize` path and native `constraints` keyword.
- #64 is complete. Its public constraints page defines the native vocabulary
  and measured projection costs used by this design.
- #68 should reuse the same MOI ingestion layer for polynomial objectives. A
  future objective adapter may accept PolyJuMP data or a polynomial subset of
  `MOI.ScalarNonlinearFunction`, but it must not create a second optimizer or a
  separate model cache.
- #95 is a future native `SumModConstraint`. Because MOI has no standard scalar
  modulo-equality set, it should add a custom MOI set that lowers to that native
  type, following the same extension pattern as `NotEqualTo` and
  `ForbiddenAssignment`.
- #55 remains the coordination epic. This design completes its JuMP/MOI design
  child but does not implement the bridge.

## Follow-Up Implementation Epic

The implementation should be split into reviewable stages:

1. **Lowering core and typed errors**: add canonical affine normalization,
   fixed-value substitution, specialized-pattern recognition, a static
   infeasibility sentinel, and unit tests that compare each native result with
   the source MOI predicate.
2. **Optimizer storage and copy path**: add the internal MOI/native constraint
   store, the concrete `supports_constraint`, `copy_to`, `empty!`, and query
   methods, and the filtered-model materialization plus composed-index-map
   handoff to QUBODrivers.
3. **Standard and custom sets**: implement the supported standard mapping rows,
   then add `NotEqualTo` and `ForbiddenAssignment` with JuMP syntax and MOI
   conformance tests.
4. **Solve and status integration**: pass native constraints through
   `QUBODrivers.sample`, add concrete status getters, and verify empty-result
   infeasibility behavior.
5. **End-to-end documentation and tests**: add JuMP examples for weighted sums,
   exactly-one, pairwise exclusion/relation, infeasibility, fixed variables,
   and every rejection path. Compare small solutions against brute force and
   the direct native API.

Each stage must preserve unconstrained QUBODrivers behavior. The test matrix
should include direct MOI use, JuMP's caching/bridge optimizer, model reuse after
`empty!`, fixed variables, maximization, unsupported sets, and a constraint that
becomes contradictory only after fixed-variable substitution.

## Non-Goals

This design does not:

- implement functional JuMP constraint support;
- add a runtime dependency on MathOptSetDistances.jl;
- stabilize every native constraint type against future SemVer changes;
- support continuous, spin-domain, or general-integer constrained models;
- define penalty strengths for unsupported constraints;
- design polynomial objective lowering beyond the shared ingestion boundary;
  or
- add CoTenN's quantum/eigenvalue constraints.

## Design Acceptance Checklist

A follow-up implementation is complete only when:

- JuMP uses the existing `TenSolver.Optimizer` for constrained binary models;
- the native direct API and MOI front end remain clearly separated;
- every accepted MOI constraint has a documented native target and projection
  cost;
- unsupported data fails with a typed, actionable error and no implicit
  penalty or exponential fallback;
- fixed variables cannot corrupt MOI-variable-to-site mapping;
- infeasibility is observable through standard MOI status and result
  attributes; and
- unconstrained QUBODrivers conformance and current direct constraint behavior
  remain unchanged.

The projection strategy is inspired by Sharma, Peng, Dangwal, and Achour,
*“CoTenN: Constrained Optimization with Tensor Networks,”* PLDI 2026. TenSolver
uses that projection-MPO framework as the motivation for keeping the native IR
small and cost-aware.
