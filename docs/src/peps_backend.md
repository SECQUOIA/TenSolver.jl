# PEPS Backend

TenSolver has two backend families:

- `DMRG` is the default backend. It accepts general QUBO/PUBO inputs and does
  not require any graph layout metadata.
- `PEPS` is an optional experimental backend for structured quasi-two-
  dimensional QUBO/Ising problems. It is useful only when variables have an
  explicit supported layout, such as a square or king grid.

The PEPS backend is inspired by the SpinGlassPEPS work described in
[arXiv:2411.16431](https://arxiv.org/abs/2411.16431). It should not be treated
as a replacement for the default TenSolver DMRG path. The paper reports useful
behavior on structured lattice and planted/tile-style instances, while also
showing stability and scaling limitations for large random Pegasus and Zephyr
instances. TenSolver therefore keeps PEPS opt-in and topology-aware.

## Installation

The default TenSolver installation does not install the SpinGlassPEPS component
stack. The optional bridge loads only when the component packages are available:

```julia
using Pkg

Pkg.add([
    "SpinGlassNetworks",
    "SpinGlassEngine",
    "SpinGlassTensors",
])
```

The registered SpinGlassPEPS umbrella package has historically required newer
Julia versions than TenSolver's core support window. TenSolver's bridge uses the
component packages as weak dependencies, but if the package resolver cannot find
a compatible environment, try a Julia 1.11+ project dedicated to PEPS
experiments. GPU packages are optional; the examples and small benchmarks use
CPU execution by default.

If the optional packages are not installed, constructing a `PEPSBackend` is
still allowed, but solving with it fails early with an error explaining which
packages are needed. DMRG remains available.

## Backend Selection

Use the default path for arbitrary QUBO/PUBO models:

```julia
using TenSolver

energy, solution = minimize(Q; verbosity = 0)
```

or select it explicitly:

```julia
energy, solution = minimize(Q; backend = :dmrg, verbosity = 0)
energy, solution = minimize(Q; backend = DMRGBackend(), verbosity = 0)
```

Use PEPS only when the variable order matches a supported structured topology:

```julia
backend = PEPSBackend(
    SquareGrid(2, 2);
    beta = 2.0,
    bond_dim = 4,
    max_states = 16,
    cutoff_prob = 0.0,
    contraction = :svd,
    transformations = :identity,
)

energy, solution = minimize(Q, l, c; backend, verbosity = 0)
```

`SquareGrid(m, n)` assumes variables are ordered row by row across the `m` by
`n` grid. `KingGrid(m, n)` uses the same variable order but permits diagonal
neighbor interactions in addition to horizontal and vertical ones.

## Supported Topologies

| Layout | TenSolver API | Status |
| --- | --- | --- |
| Square grid | `SquareGrid(m, n[, spins_per_site])` | Implemented |
| King grid | `KingGrid(m, n[, spins_per_site])` | Implemented |
| Pegasus | Not exposed yet | Planned only |
| Zephyr | Not exposed yet | Planned only |

TenSolver does not infer topology from an arbitrary dense QUBO. Passing PEPS a
problem whose couplings do not fit the selected topology raises an error; use
the DMRG backend unless the model has known layout structure.

## Direct API Example

This example is small enough to compare with brute force by inspection. It uses
a square grid, so the four binary variables are ordered as:

```text
1 2
3 4
```

```julia
using TenSolver

Q = [
    -1.0  0.5  0.0  0.0
     0.0 -0.5  0.0  0.0
     0.0  0.0 -0.25 0.25
     0.0  0.0  0.0 -0.75
]
l = [0.0, 0.25, -0.25, 0.0]
c = 0.125

backend = PEPSBackend(
    SquareGrid(2, 2);
    beta = 2.0,
    bond_dim = 4,
    max_states = 4,
    cutoff_prob = 0.0,
    contraction = :svd,
    transformations = :identity,
)

energy, solution = minimize(Q, l, c; backend, verbosity = 0)
state = sample(solution)
```

For PEPS runs, `solution` is a [`PEPSSolution`](@ref). It stores retained Boolean
states, objective values, probabilities, and backend metadata:

```julia
solution.states
solution.energies
solution.probabilities

metadata = solution.metadata
metadata["topology"]
metadata["selected_transformation"]
get(metadata, "largest_discarded_probability", missing)
```

## JuMP and QUBODrivers Example

The JuMP optimizer keeps DMRG as the default. Select PEPS with raw optimizer
attributes and provide topology metadata explicitly:

```julia
using JuMP, TenSolver

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
```

When PEPS is used through QUBODrivers, the returned QUBOTools `SampleSet`
contains PEPS metadata under the `"peps"` key, including topology, search
parameters, selected transformation, candidate-state count, and effective time.
Standard JuMP calls such as `objective_value(model)` and `value.(x)` remain the
portable interface for ordinary modeling workflows.

## Parameter Guide

- `beta`: inverse temperature for the PEPS representation. Higher values
  sharpen low-energy resolution but can make approximate contraction less
  stable.
- `bond_dim`: boundary MPS bond dimension. Larger values can improve
  contraction quality and increase memory/time cost.
- `max_states`: branch-and-bound state width. Increasing it retains more
  candidates but raises runtime and memory use.
- `cutoff_prob`: branch pruning threshold. Higher values prune more aggressively
  and may discard useful low-energy candidates.
- `transformations`: lattice transformations to try. `:identity` is fastest;
  `:all` can improve robustness by trying rotations/reflections.
- `contraction`: contraction strategy, currently `:auto`, `:svd`,
  `:svd_truncate`, or `:zipper`.
- `local_dimension`: optional local dimension reduction. It can reduce cost but
  may remove globally optimal configurations if set too aggressively.
- `onGPU`: requests SpinGlassPEPS GPU execution. GPU memory can limit feasible
  bond dimensions, especially on hardware-style topologies.

## Limitations

- PEPS is not appropriate for arbitrary dense QUBOs.
- The current TenSolver bridge supports square and king grids; Pegasus and
  Zephyr are not exposed yet.
- Approximate contraction can fail or produce poor probability estimates.
- Randomized or truncated contraction strategies can introduce run-to-run
  variability.
- Large hardware-style instances can be limited by memory, especially on GPU.
- The default DMRG backend remains the general-purpose TenSolver backend.

## Benchmarks

Small reproducible benchmark scripts live outside CI under `benchmarks/`:

```bash
julia --project=. benchmarks/peps_square.jl
julia --project=. benchmarks/peps_king.jl
```

Each script sets a random seed, uses CPU defaults, compares DMRG and PEPS when
the optional PEPS packages are available, and includes brute force for the tiny
instances used by the script. If the optional PEPS stack is not installed, the
PEPS row is reported as skipped while the DMRG and brute-force rows still run.
The default instances should finish in seconds to a few minutes on a laptop,
depending on Julia precompilation and whether PEPS is installed.

The scripts report best objective value, gap from brute force when available,
runtime, retained-state count, selected transformation, and largest discarded
probability. They are intended for local experimentation and regression checks,
not for reproducing the full arXiv benchmark suite.
