# [PEPS Backend](@id peps-backend-guide)

TenSolver provides two tensor-network backends:

- DMRG is the default and supports the general QUBO/PUBO interface.
- PEPS is optional and experimental. It targets quadratic Ising models whose
  variables follow a supported two-dimensional layout.

PEPS is not a replacement for DMRG. Select it when the problem already has a
square- or king-grid structure and you want to explore the contraction and
branch-and-bound controls described below. Keep DMRG for arbitrary dense models,
higher-degree objectives, or models without a known layout.

## Availability

The PEPS backend requires registered `SpinGlassPEPS` 2.x and Julia 1.11 or later.
Install it in the same environment as TenSolver, then load it before solving:

```julia
using Pkg
Pkg.add(PackageSpec(name = "SpinGlassPEPS", version = "2"))
import SpinGlassPEPS
```

SpinGlassPEPS is optional and is not installed with TenSolver. Selecting PEPS
without loading it raises an error with installation requirements. DMRG remains
the default backend and supports Julia 1.10 or later.

GPU support is optional. Small examples and the repository benchmarks use CPU
execution by default.

## Model and Topology

The direct PEPS API accepts a quadratic Ising objective

```math
\min_{s \in \{-1, 1\}^N} s^\mathsf{T} J s + h^\mathsf{T}s + c.
```

Choose the constructor that matches the interaction graph and variable order:

| Layout | Constructor | Allowed interactions |
| --- | --- | --- |
| Square grid | `TenSolver.SquareGrid(m, n[, spins_per_site])` | Same cell plus horizontal and vertical neighbors |
| King grid | `TenSolver.KingGrid(m, n[, spins_per_site])` | Square-grid interactions plus diagonal neighbors |

Variables are ordered row by row. For `SquareGrid(2, 2)`, the order is:

```text
1 2
3 4
```

TenSolver checks the number of variables and every nonzero coupling against the
selected layout. It does not infer a topology from an arbitrary matrix. Pegasus
and Zephyr layouts are not currently exposed.

If the model begins as a Boolean QUBO, convert it explicitly before a direct
PEPS solve:

```julia
form = TenSolver.qubo_to_ising(Q, l, c)
_, h, J, scale, offset, _, _ = form
@assert scale == 1
```

This uses ``s = 2x - 1`` and preserves objective values. JuMP performs this
conversion automatically when `"backend"` is set to `:peps`.

## Direct API

The four-spin objective below matches a `2 × 2` square grid:

```julia
using TenSolver
import SpinGlassPEPS

J = [
    0.0  0.5  0.0  0.0
    0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.25
    0.0  0.0  0.0  0.0
]
h = [-1.0, -0.25, 0.25, -0.75]
c = 0.125

backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(2, 2))
energy, solution = minimize(
    J,
    h,
    c;
    domain = [-1, 1],
    backend,
    beta = 2.0,
    maxdim = 4,
    iterations = 1,
    max_states = 4,
    cutoff_prob = 0.0,
    contraction = :svd,
    transformations = :identity,
    verbosity = 0,
)

spins = sample(solution)
```

Direct PEPS samples contain `-1` and `1`. The solution also retains ranked
states, their original objective values including the offset, normalized
probabilities, and backend metadata. The probabilities describe the distribution
conditioned on the states retained by the search; they do not measure the total
Boltzmann mass captured. `max_states` is an upper bound, so fewer states may be
returned even with `cutoff_prob = 0.0`:

```julia
solution.states
solution.energies
solution.probabilities
solution.metadata["selected_transformation"]
get(solution.metadata, "largest_discarded_probability", missing)
```

The diagnostic `largest_discarded_probability` preserves the upstream log
probability, which can be `-Inf` when nothing was discarded.

## JuMP and QUBODrivers

JuMP models remain Boolean. Select PEPS explicitly and supply the layout:

```julia
using JuMP, TenSolver
import SpinGlassPEPS

m, n = 2, 2
model = Model(TenSolver.Optimizer)
set_silent(model)
set_attribute(model, "backend", :peps)
set_attribute(model, "peps_layout", :square)
set_attribute(model, "peps_topology", (m, n))
set_attribute(model, "peps_beta", 2.0)
set_attribute(model, "peps_bond_dim", 4)
set_attribute(model, "peps_max_states", 4)
set_attribute(model, "peps_cutoff_prob", 0.0)
set_attribute(model, "peps_strategy", :svd)
set_attribute(model, "peps_transformations", :identity)

@variable(model, x[1:(m * n)], Bin)
@objective(model, Min, -sum(x))
optimize!(model)

objective_value(model)
value.(x)
```

The optimizer converts the QUBO to Ising spins for the solve and returns Boolean
QUBOTools samples. PEPS details are namespaced under
`metadata["tensolver"]["peps"]`:

```julia
import QUBOTools

samples = QUBOTools.solution(JuMP.unsafe_backend(model))
metadata = QUBOTools.metadata(samples)
peps = metadata["tensolver"]["peps"]

peps["candidate_states"]
peps["states"]
peps["probabilities"]
peps["parameters"]
peps["effective_time"]
```

These metadata states are Boolean vectors, with normalized probabilities over
the retained set. They include candidates whose allocated read count rounds to
zero. SampleSet read counts sum to the requested final number of reads (or
`num_reads` when no final number is supplied); integer allocation can differ
slightly from probability times reads. Both `Min` and `Max` objectives retain
their original sense and constant offset in the returned sample values.

## Parameters

The direct solve keywords and JuMP attributes configure the same settings:

| Purpose | Direct solve keyword | JuMP attribute | Default |
| --- | --- | --- | --- |
| Inverse temperature | `beta` | `"peps_beta"` | `2.0` |
| Boundary-MPS bond dimension | `maxdim` | `"peps_bond_dim"` | `16` |
| Contraction sweeps | `iterations` | `"peps_num_sweeps"` | `1` |
| Accelerator function | `device` | `"peps_device"` | CPU |
| Retained search width | `max_states` | `"peps_max_states"` | `256` |
| Branch pruning threshold | `cutoff_prob` | `"peps_cutoff_prob"` | `1e-4` |
| Contraction implementation | `contraction` | `"peps_strategy"` | `:auto` |
| Gradual truncation | `graduate_truncation` | `"peps_graduate_truncation"` | `true` |
| Lattice transformations | `transformations` | `"peps_transformations"` | `:all` |
| Optional local dimension | `local_dimension` | `"peps_local_dimension"` | `nothing` |
| Disable contraction cache | `no_cache` | `"peps_no_cache"` | `false` |

Higher `beta` sharpens low-energy resolution but can make approximate
contraction less stable. Larger `maxdim` and `max_states` can retain more
information at higher memory and runtime cost. Increasing `cutoff_prob` prunes
more aggressively and may discard useful states.

Use `transformations = :identity` for the smallest run or `:all` to try the
available rotations and reflections. `:auto`, `:svd`, and `:svd_truncate`
currently select the same SVD-truncation path; `:zipper` selects zipper
contraction. `local_dimension` can lower cost but may remove the globally best
configuration if chosen too aggressively.

## Restrictions and Cost

- PEPS accepts quadratic objectives only and requires `domain = [-1, 1]` in the
  direct API.
- `preprocess = true` is unsupported because preprocessing would change the
  declared variable layout.
- Native constraints are unsupported by PEPS; use DMRG for constrained models.
- Couplings outside the selected square/king graph are rejected.
- Approximate contraction can fail or return poor probability estimates.
- Runtime and memory grow with layout size, boundary bond dimension, retained
  search width, and the number of transformations.
- Truncated or randomized numerical routines may introduce run-to-run variation.

## Local Benchmarks

Small deterministic comparison scripts live outside normal CI:

```bash
julia --project=benchmarks benchmarks/peps_square.jl
julia --project=benchmarks benchmarks/peps_king.jl
```

Each script compares brute force, DMRG, and PEPS on a tiny structured instance.
It reports objective value, exact gap, runtime, retained-state count, selected
transformation, and the upstream largest-discarded log probability. When
SpinGlassPEPS is unavailable, only the PEPS row is skipped. Load or solve failures
appear as error rows. See `benchmarks/README.md` for environment setup and scope.
