# JuMP/MOI Constrained Optimization Architecture

This note defines how a future JuMP/MathOptInterface (MOI) front end should
lower constrained finite-domain models into TenSolver's native
projection-constraint pipeline. It is an implementation design, not a
description of behavior in the current release. The direct Julia API already
solves quadratic and polynomial objectives over a uniform finite domain, while
`TenSolver.Optimizer` still accepts only Boolean QUBO models and fixed Boolean
variables and does not accept the general hard constraints described here.

The central decision is to keep two layers with different responsibilities:

- MOI `function-in-set` constraints are the high-level modeling interface used
  by JuMP.
- TenSolver's native `AbstractConstraint` subtypes remain the small,
  projection-aware intermediate representation (IR) consumed by the solver.

The MOI layer must lower into the native IR. It must not teach the projection
code to dispatch on arbitrary MOI sets, and it must not turn supported hard
constraints into penalty terms.

## Current Boundary

The direct Julia API already accepts native constraints through the
`constraints` keyword of `minimize` and `maximize`. The DMRG
backend lowers each native constraint with `constraint_to_dfa`, constructs its
projection MPO, and solves in the projected space. The four v1 native types are:

- `SumConstraint`;
- `NotEqualsConstraint`;
- `ExactlyOneConstraint`; and
- `RelationConstraint`.

The JuMP path stops earlier. `TenSolver.Optimizer` is generated with
`QUBODrivers.@setup`, and QUBODrivers defines an `AbstractSampler` around a
Boolean QUBO/Ising model, apart from fixed-value constraints. Its generic
`MOI.supports_constraint` fallback rejects other constraint types.
`QUBODrivers.sample` in TenSolver then extracts only QUBO objective data with
`QUBOTools.qubo` and calls `minimize` without a `domain`, `constraints`, or
`backend` argument. Consequently, the current JuMP path cannot represent the
finite-domain work from #105/#67, the hard constraints from #62, the backend
selection planned in #44, or the polynomial objectives requested in #68.

That generated sampler remains a useful compatibility adapter for Boolean
quadratic models, solver attributes, `SampleSet` construction, and common MOI
result behavior. It must not become the authoritative model store for the
expanded interface. TenSolver should own the full MOI model, domain and
objective normalization, native-constraint lowering, backend selection, and
status mapping. If that implementation exposes a generally useful optimizer
storage or result hook, contribute the narrow hook to QUBODrivers; keep
TenSolver-specific domains, constraints, and backend rules in TenSolver.

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

MOI's function-in-set vocabulary supplies the modeling abstraction required by
the bridge. Numerical point-projection helpers such as
`MOI.Utilities.distance_to_set` do not construct a Hilbert-space projection MPO
and are not part of this implementation; no set-distance dependency is needed.

## Model-Ingestion Flow

The constrained optimizer should use a TenSolver-owned, non-incremental
`MOI.copy_to` path:

1. JuMP stores the source model in its normal MOI caching optimizer.
2. TenSolver's concrete `MOI.copy_to(::Optimizer, src)` copies the complete
   queryable model into a TenSolver-owned `MOI.Utilities` model store and
   returns its source-to-local `MOI.IndexMap`. `MOI.empty!` resets that store,
   cached execution data, and result state together.
3. Before solving, TenSolver inventories each variable's domain without
   assuming `ZeroOne`. `ZeroOne` normalizes to `[0, 1]`; other supported finite
   sets normalize to their explicit values. The current DMRG API requires one
   uniform finite real domain, so heterogeneous domains remain stored correctly
   but produce a typed backend-capability error until #67 adds per-variable
   domains.
4. An objective adapter normalizes the objective independently of constraints.
   Boolean quadratic data may reuse QUBOTools/QUBODrivers conversion, while the
   shared store leaves a direct path for #68 to lower a polynomial JuMP
   objective to the existing `AbstractPolynomial` API without creating another
   optimizer or model cache.
5. Fixed values are validated against their declared domains and substituted
   into both the objective and hard constraints. TenSolver records the mapping
   from surviving MOI variables to native solver sites.
6. Hard constraints are canonicalized only after domains and fixed values are
   known. Domain-independent patterns lower directly; target-specific
   restrictions, such as `SumConstraint` requiring a nonnegative integer
   domain, are checked by the native lowering layer.
7. The selected backend is normalized through the existing backend interface.
   TenSolver validates its objective, domain, constraint, and topology
   capabilities before calling `minimize`; unsupported combinations raise
   `ConstraintUnsupportedByBackend` or the corresponding typed capability
   error rather than silently changing backend.
8. The solve result is translated into standard MOI status and primal
   attributes. The Boolean quadratic compatibility path may still construct a
   QUBOTools `SampleSet`, but that adapter does not define the model's accepted
   domains or objective classes.

This ownership avoids a constraint side table attached to generated sampler
internals and removes the need to hide constraints from a QUBODrivers copy.
TenSolver may reuse public QUBODrivers attributes and result helpers or request
a small generic upstream hook, but it must not call private objective-copy
helpers or move projection-specific semantics into QUBODrivers.

The stored MOI model answers `NumberOfConstraints`,
`ListOfConstraintIndices`, `ConstraintFunction`, `ConstraintSet`, and
`is_valid`. Native constraints are execution data, not a substitute for the
MOI model's queryable representation.

### Variable domains

The first front end should recognize domains independently from objectives and
hard constraints:

- `VariableIndex`-in-`ZeroOne` normalizes to `[0, 1]`;
- `VariableIndex`-in-`Integer` combined with finite integral bounds normalizes
  to the enumerated integer values;
- a TenSolver `FiniteDomain(values)` set represents spin, sparse integer,
  fractional, or other explicit finite real domains; and
- `VariableIndex`-in-`EqualTo(value)` fixes a variable but does not replace its
  declared domain.

Multiple domain constraints are intersected. An empty intersection is a static
infeasibility result; an unbounded `Integer` or continuous interval is an
unsupported domain. The current DMRG handoff requires all surviving variables
to have the same normalized domain. This restriction is checked after the MOI
model is stored so #67 can later add heterogeneous domains without changing the
front-end representation.

### Fixed variables and static simplification

Objective normalization can remove fixed variables from the sampled problem.
Constraint lowering must therefore substitute fixed values before assigning
native site numbers. For example, fixing `x[1] = 1` turns
`x[1] + x[2] <= 1` into `x[2] <= 0`; referring to the old first site after
removal would constrain the wrong variable.

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

Canonicalization itself is domain-neutral. It must not recognize Boolean-only
identities until the variables' domains are known. The current native sum
projection accepts nonnegative integer weights and bounds and only lowers over
a nonnegative integer domain. The native adapter must preserve that narrower
target contract:

- coefficients and normalized bounds must be exactly integer-valued; the
  bridge must not round approximately integral floating-point data;
- an all-negative expression may be multiplied by `-1`, reversing `<=` and
  `>=`;
- mixed-sign expressions are accepted only when they match a specialized
  pairwise-relation pattern; and
- strict `<` and `>` constraints are not accepted. A caller that relies on an
  integer domain must express the equivalent shifted non-strict bound
  explicitly.

After domain normalization, specialized compact patterns are selected before
the general sum fallback. For example, `sum(x) == 1` is exactly-one only for a
domain where that predicate is equivalent to exactly one site taking the
target value. These simplifications belong to TenSolver's native lowering, not
the MOI ingestion layer. This is part of correctness for the performance
contract, not a cosmetic optimization.

A genuine two-sided bound `l <= sum(a[i] * x[i]) <= u` should likewise use one
specialized projection rather than compose separate lower- and upper-bound
projectors. Extend `SumConstraint` with a two-sided relation variant that
stores both closed bounds. Its `constraint_to_dfa` method uses partial-sum
states `0:u+1`, sends every sum above `u` to the overflow state `u+1`, and
accepts states `l:u`. The projection MPO therefore has bond at most `u + 2`,
instead of the product of the two one-sided bonds. A separate
`SumIntervalConstraint` would duplicate the same weighted-sum abstraction and
is not needed. Normalization should still reduce an interval with a redundant
side to the existing one-sided relation, or report static infeasibility when
its bounds cannot contain a reachable sum.

## Initial Mapping Table

The MOI store and the table make no blanket `ZeroOne` assumption. Each row
states the domain required by its current native target. “Bond” is the expected
projection-MPO bond bound per native constraint after lowering; `|U|` is the
size of the uniform finite domain.

| MOI/JuMP form | Native lowering | Bond | Conditions and notes |
|:--------------|:----------------|:-----|:---------------------|
| Nonnegative integer affine function in `EqualTo`, `LessThan`, or `GreaterThan` | `SumConstraint` with `==`, `<=`, or `>=` | `rhs + 2` | Requires a uniform nonnegative integer domain and exactly integral normalized coefficients/bounds. |
| Nonnegative integer affine function in `Interval` | `SumConstraint` with a two-sided relation | `upper + 2` | Same domain contract; use one capped DFA and simplify a redundant side to a one-sided relation. |
| `x[i] - x[j]` in `EqualTo(0)`, `LessThan(0)`, or `GreaterThan(0)` | `RelationConstraint(i, relation, j)` | `|U|` | Domain-independent for a common ordered finite real domain. |
| `VectorOfVariables(x)` in `MOI.AllDifferent(length(x))` | One `RelationConstraint(i, :(!=), j)` per pair | `|U|` each | Domain-independent; statically infeasible when `length(x) > |U|`. |
| `VectorOfVariables(x)` in a TenSolver `ExactlyOneValue(value)` set | `ExactlyOneConstraint(sites, value)` | 2 | Any uniform finite domain; a target outside the domain is statically infeasible. |
| `VectorOfVariables(x)` in `MOI.SOS1(weights)` | Domain-aware lowering | 3 for `U = {0,1}` | Boolean domains lower to `SumConstraint(..., <=, 1)`; other domains are unsupported until a compact at-most-one-nonzero target exists. |
| `ScalarAffineFunction` in a TenSolver `NotEqualTo(value)` set | `SumConstraint(...; relation = :(!=))`, or `RelationConstraint` for a pairwise pattern | Native target's bound | General sums retain the sum target's domain restrictions. |
| `VectorOfVariables(x)` in a TenSolver `ForbiddenAssignment(values)` set | `NotEqualsConstraint(sites, values)` | 2 | Any uniform finite domain; values outside the domain make the constraint a tautology. |
| `VariableIndex` in `EqualTo(value)` | Fixed-value substitution | No projector | Validate `value` against the declared finite domain before lowering. |

Once the domain is known, the native simplifier may recover compact Boolean
identities: `sum(x) == 1` and `sum(x) == length(x)-1` become
`ExactlyOneConstraint` targets, and `x[i] + x[j] <= 1` becomes a forbidden
assignment. They are not valid domain-independent MOI rewrites. A bare `SOS1`
cannot lower to `ExactlyOneConstraint` because the all-zero assignment is
feasible for `SOS1`.

The custom `ExactlyOneValue`, `NotEqualTo`, and `ForbiddenAssignment` sets
belong to the MOI front end. They do not replace native constraint types; they
provide function-in-set syntax for projection targets that have no lossless
off-the-shelf MOI set.

## Unsupported Constraints and Fallback Policy

The default policy is an explicit hard error. The v1 bridge must not silently:

- convert a hard constraint into a penalty QUBO;
- introduce exhaustive feasible-assignment enumeration as a fallback; or
- accept a bridge that introduces a continuous, infinite, or unspecified
  domain that the selected backend cannot represent.

Penalty conversion changes the optimization problem unless a safe penalty
scale is known. Exhaustive feasible-path enumeration can preserve semantics but
can have exponentially larger bonds; no such fallback exists in the current
code, and this design does not reintroduce it. Neither behavior is acceptable
implicitly.

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
functions and explicit finite-domain sets. A bridge must not silently replace a
declared domain with `ZeroOne`; every introduced variable carries a domain that
the selected backend validates.

Bridge output is canonicalized exactly like user-authored constraints. The
bridge boundary cannot bypass domain, integer, nonnegativity, fixed-variable,
objective, or backend checks.

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
- no variable primal values.

TenSolver already computes `MOI.INFEASIBLE` in `tensolver_status`, but the
generic QUBODrivers termination-status getter treats a nonempty metadata object
with zero samples as `MOI.OTHER_ERROR`. The constrained implementation must add
a concrete `MOI.get(::Optimizer, ::MOI.TerminationStatus)` override that reads
TenSolver's recorded status. The Boolean quadratic compatibility adapter also
returns an empty QUBOTools `SampleSet` with that status in metadata. Metadata
alone does not complete the MOI contract.

Invalid model data and unsupported backends do not map to `INFEASIBLE`: they
remain typed construction or solve errors. Constraint duals are unsupported.

## Coordination With Related Work

- #37 and #62 are complete. The bridge targets the landed backend-aware
  `minimize`/`maximize` path and native `constraints` keyword.
- #64 is complete. Its public constraints page defines the native vocabulary
  and measured projection costs used by this design.
- #105 and the subsequent #106/#109 domain work are on `main`. The MOI store is
  therefore finite-domain-neutral, while the first backend adapter targets the
  uniform-domain contract currently implemented by DMRG. Per-variable domains
  remain coordinated through #67.
- #44 is the pending optional-backend integration. Backend selection is part of
  model normalization, and every backend must explicitly advertise supported
  objective, domain, constraint, and topology combinations. The MOI front end
  must not assume DMRG or silently fall back to it.
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

1. **Optimizer storage and domain inventory**: add the TenSolver-owned MOI model
   store, `supports_constraint`, `copy_to`, `empty!`, query methods, explicit
   finite-domain normalization, and compatibility tests for existing
   QUBODrivers attributes and Boolean QUBO behavior.
2. **Objective and backend handoff**: normalize quadratic objectives through
   the existing adapter, preserve one extension point for #68 polynomial
   objectives, select the backend explicitly, and map surviving MOI variables
   to native sites after fixed-value substitution.
3. **Lowering core and typed errors**: add canonical affine normalization,
   domain-aware specialized-pattern recognition, the two-sided
   `SumConstraint` relation, a static infeasibility sentinel, and unit tests
   that compare each native result with the source MOI predicate.
4. **Standard and custom sets**: implement the supported standard mapping rows,
   then add `ExactlyOneValue`, `NotEqualTo`, and `ForbiddenAssignment` with JuMP
   syntax and MOI conformance tests.
5. **Solve, status, and end-to-end coverage**: pass the normalized domain and
   native constraints to the selected backend, add concrete status/primal
   getters, and add JuMP examples for weighted sums, exactly-one, pairwise
   exclusion/relation, infeasibility, fixed variables, and every rejection
   path.

Each stage must preserve existing unconstrained QUBODrivers behavior. The test
matrix should include direct MOI use, JuMP's caching/bridge optimizer, model
reuse after `empty!`, Boolean, spin, and nonnegative-integer uniform domains,
fixed variables, maximization, unsupported sets/backends, and a constraint that
becomes contradictory only after fixed-variable substitution. Small accepted
models must be compared with brute force and the direct native API.

## Non-Goals

This design does not:

- implement functional JuMP constraint support;
- add a runtime dependency on MathOptSetDistances.jl;
- stabilize every native constraint type against future SemVer changes;
- support continuous or infinite-domain models;
- implement heterogeneous per-variable domains before #67 provides the direct
  solver contract;
- make every current native constraint work on every finite domain;
- define penalty strengths for unsupported constraints;
- design polynomial objective lowering beyond the shared ingestion boundary;
- implement PEPS-specific constraint projections; or
- add CoTenN's quantum/eigenvalue constraints.

## Design Acceptance Checklist

A follow-up implementation is complete only when:

- JuMP uses the existing `TenSolver.Optimizer` for constrained finite-domain
  models supported by the selected backend;
- the native direct API and MOI front end remain clearly separated;
- the MOI model store does not assume `ZeroOne`, quadratic objectives, or DMRG;
- every accepted MOI constraint has a documented native target and projection
  cost;
- unsupported data fails with a typed, actionable error and no implicit
  penalty or exponential fallback;
- fixed variables cannot corrupt MOI-variable-to-site mapping;
- infeasibility is observable through standard MOI status and result
  attributes; and
- Boolean QUBODrivers conformance and current direct constraint behavior remain
  unchanged.

The projection strategy is inspired by Sharma, Peng, Dangwal, and Achour,
*“CoTenN: Constrained Optimization with Tensor Networks,”* PLDI 2026. TenSolver
uses that projection-MPO framework as the motivation for keeping the native IR
small and cost-aware.
