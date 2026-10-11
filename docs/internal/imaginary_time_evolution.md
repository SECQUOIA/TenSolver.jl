# Imaginary-Time Evolution Feasibility Assessment

This assessment records a local prototype for
[issue #93](https://github.com/SECQUOIA/TenSolver.jl/issues/93), tested on
2026-10-11 at `main` commit
`610203f974af2f6aceee19e7bda34de57a1f3570`.

Existing `ITensorMPS.tdvp` and normalization compose with TenSolver's current
tensorization, spectral shifting, and feasible-state projection. The results
support documenting that composition in a runnable example. They do not
establish a need for an additional helper or settle a public API.

## Existing Composition

ITensorMPS defines `tdvp(H, t, psi)` as an approximation to `exp(t * H) * psi`.
Imaginary-time evolution therefore uses a negative real time:

```julia
psi = ITensorMPS.tdvp(H, -dt, psi; nsteps=1, normalize=true, kwargs...)
psi = normalize(psi)
```

For constrained models, the prototype prepares a feasible initial state,
evolves under the shifted projected Hamiltonian, and reprojects after each step.
Model preparation relies on qualified, unexported TenSolver APIs:
`Domains`, `tensorize`, `projection_mpos`, `objective_box_bound`,
`project_hamiltonian`, and `project_feasible_state`. These remain internal APIs
subject to change.

The July comment on #93 predates
[the spectral-shift fix in #133](https://github.com/SECQUOIA/TenSolver.jl/pull/133).
At the tested commit, constrained solves construct `P * (Hobj - lambda * I) * P`,
placing the feasible spectrum below the zero-energy projection kernel. The
constant objective term is excluded from `Hobj` and the bound. The prototype
measures the original objective through `Hobj` plus that constant.

## Fixtures and Independent References

All objectives are minimized. Four deterministic fixtures have complete state
spaces of at most 16 assignments:

- **Binary:** `x` in `{0,1}^4`, with objective
  `0.7 + 2x1 - x2 + 0.5x3 + 3x4 - 3x1x2 + 1.5x2x3 - 2x3x4 + 0.75x1x4`.
  The unique minimum is `-1.3` at `(1,1,0,0)`.
- **Degenerate:** `x` in `{0,1}^3`, with objective `(x1 - x2)^2`.
  The minimum is zero at four assignments: `x1 = x2`, with `x3` free.
- **Heterogeneous domains:** supplied domains `[2,0]`, `[3,-1,1]`, and
  `[1.5,-0.5]`, with objective
  `1.1 + 0.7x1 - 1.2x2 + 0.3x3 + 0.5x1x2 - 0.8x2x3`.
  The unique minimum is `-5.65` at `(0,3,1.5)`.
- **Positive-energy constrained:** `x` in `{0,1}^3`, with `x1 + x2 + x3 = 1`
  and objective `5 + 2x1 + 3x2 + 4x3 + 1.5x1x2 - 0.5x2x3`.
  Feasible energies are `7`, `8`, and `9`; the minimum is at `(1,0,0)`.
  Here `lambda = 12`, giving feasible solve energies `-10`, `-9`, and `-8`.

Dense reference Hamiltonians were constructed directly from these mathematical
objectives and constraint masks, independently of the tensorized MPOs and
TenSolver's feasibility predicates. Domains were independently sorted and
deduplicated. Explicit tensor indices and Cartesian enumeration both used
site 1 as the fastest-varying index.

Initial amplitudes were proportional to
`1 + 0.15sin(0.7k) + 0.08cos(0.31k^2)`, with `k` the basis index. The constrained
initial state was projected and normalized. Initial ground-space probabilities
were `0.067226`, `0.527569`, `0.109975`, and `0.404898`, respectively; the
degenerate case sums over all four ground states.

## Protocol and Results

Parameters and tolerances were fixed before execution. Each main fixture used
total imaginary time `tau = 1`, with both four steps of `0.25` and eight steps of
`0.125`. Two-site TDVP used order 2, `maxdim = 16`, `cutoff = 1e-14`, and a
Hermitian Krylov updater with tolerance `1e-12`, dimension 32, and maximum
iterations 30.

Acceptance tolerances were `2e-10` for objective/solve MPO entries and
normalization, `2e-7` for final fine-step state and objective errors, and `2e-9`
for feasible-mass and projection errors. Ground-space overlap had to exceed
`1e-10`. All gates passed.

Fine-step results at `tau = 1`:

| Fixture | States | Objective expectation | Gap above minimum | State error |
|---|---:|---:|---:|---:|
| Binary | 16 | -1.101883518 | 0.198116482 | 2.62e-15 |
| Degenerate | 8 | 0.108091074 | 0.108091074 | 9.61e-16 |
| Heterogeneous domains | 12 | -5.648501334 | 0.001498666 | 3.38e-16 |
| Positive-energy constrained | 8 | 7.119036247 | 0.119036247 | 2.77e-15 |

State error is the norm of the difference from normalized dense finite-time
evolution after global phase alignment. Feasible probability mass remained one
within roundoff. Across all main steps, the largest objective error was
`7.99e-15`, norm error `6.66e-16`, feasible-mass discrepancy `1.44e-15`, and
projection implementation error `5.66e-16`.

A separate closed-form audit checked all 56 reference expectation and
ground-mass rows, plus the saved final vectors, objective values, and feasible
masses. An independent critic checked the source, protocol, script, and raw
results without running further evolutions. It found no blocking scientific
or acceptance defect. Its replay-artifact retention and counter finding was
corrected and checked without another TDVP run.

The runtime was Julia 1.10.11, TenSolver 0.3.0, ITensorMPS 0.4.1, and ITensors
0.9.30, with one Julia and BLAS thread. The campaign completed 56 TDVP steps in
151.1 seconds, including startup and compilation, within the predeclared limits
of 64 steps and 600 seconds. Execution used an OS memory limit and a process
timeout. There were no retries.

## Approximation and Constraint Controls

The main fixtures used enough bond capacity to represent the full state.
Separate binary controls started from an exact uniform product state:

- Four one-site rank-one steps with zero cutoff produced state error `0.316`.
  This illustrates restricted-manifold error without initial compression.
- Four two-site rank-one steps produced state error `0.261`, combining bond
  restriction and repeated SVD truncation.
- Direct rank-one compression of the exact finite-time state produced error
  `0.138`, measuring compression alone.

These quantities do not form an additive error decomposition. Agreement between
the main coarse and fine runs also does not establish an asymptotic error order.

A dense-only constrained control started with 10% infeasible probability mass.
At `tau = 1`, the unshifted projected operator increased it to 93.0%; the current
shifted operator reduced it to `5.04e-10`. Shifting suppresses leakage but does
not eliminate it. The initially feasible main runs did not stress structural
leakage, and the prototype retained feasible-state projection.

## Recommendation and Remaining Limits

Document a runnable example using the existing composition. Consider a helper
only if a specific usability gap emerges. This assessment introduces no backend,
exports, default changes, dependencies, TEBD implementation, or hybrid scheduling.

The finite-time objective gaps remain nonzero, and these tiny fixtures provide
little evidence about scalable ground-state search. The results do not establish
general TDVP convergence or resolve #19 or #134. Single-site evolution and
numerical compatibility with ITensorMPS 0.3 were not tested; the upstream 0.3
TDVP sign convention and call shape were inspected only. The experimental
campaigns associated with #19 and #134 were not reopened.

The runnable script, fixed protocol, dependency lock, raw results, and critic
report are retained in the separate local evidence archive
`issue93-20261011`. This repository document records the assessment; the full
evidence bundle remains a local artifact.
