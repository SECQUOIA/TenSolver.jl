# Knapsack penalty vs projection benchmark

This benchmark compares two encodings of the same binary knapsack instances:

- a conventional unconstrained QUBO with bounded-binary slack variables and a
  squared capacity penalty;
- TenSolver's native `SumConstraint`, lowered to a hard projection MPO.

The comparison is motivated by CoTenN's direct-constraint tensor-network
framing ([Sharma et al., PLDI 2026](https://doi.org/10.1145/3808272)). It is a
TenSolver benchmark, not a reproduction of CoTenN's published experiments,
which evaluate two physics problems rather than this knapsack penalty baseline.

## Run

The benchmark has its own environment and treats TenSolver as an external
dependency. From the repository root, develop the current checkout into that
environment once:

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

Then run the solver comparison:

```bash
julia --project=benchmarks benchmarks/knapsack/run.jl
```

Pass one path to also save the CSV output:

```bash
julia --project=benchmarks benchmarks/knapsack/run.jl results.csv
```

The solver workload is configurable without editing the harness:

```bash
julia --project=benchmarks benchmarks/knapsack/run.jl \
  --iterations=6 --reads=64 --timing-samples=3 --time-limit=120 \
  --benchmark-id=<commit-or-run-id> results.csv
```

The time limit is soft and is checked after each DMRG sweep. Use `--help` for
the complete option list. Add `--pilot` to run the same penalty sweep only on
the four-item reference instance before committing to the complete workload.

The fixed workload contains a four-item hand-checkable instance plus
uncorrelated, weakly correlated, strongly correlated, and subset-sum instance
classes from Martello, Pisinger, and Toth's standard 0-1 knapsack generator
([paper](https://doi.org/10.1287/mnsc.45.3.414),
[generator archive](https://hjemmesider.diku.dk/~pisinger/codes.html)). The
instances use 8, 12, or 16 items, coefficient range 10, and the half-total-weight
capacity slice. These small parameters keep the projection's capacity register
tractable and allow every instance to be checked against a brute-force optimum.
Integer weights and capacity are required by the projection formulation.

Each generated instance resets Julia's standard RNG at its start. Every solver
case likewise resets to seed 66 immediately before execution, so formulations
receive comparable initial randomness without derived or incremented seeds.
The benchmark remains entirely outside the package test and documentation
environments.

Before collecting rows, the runner performs one unreported two-item solve with
each formulation to reduce one-sided Julia compilation bias in the timing.
Reported wall time, GC time, allocations, and allocated bytes come from
`BenchmarkTools.jl`. Each expensive solver case uses one evaluation per sample
and, by default, the median of three fixed-seed samples. Formulation construction
is measured separately from the public solver call. Callback-Hamiltonian setup
and final sampling are also reported separately, while `end_to_end_wall_seconds`
includes formulation construction and the complete measured solver/sample case.
`solver_excluding_observer_setup_seconds` subtracts only the independently
measured callback-Hamiltonian construction; it still includes the public solver's
own tensorization/projection work and per-sweep variance calculations.

For each instance the runner executes the projection formulation once and the
penalty formulation at factors `0.001`, `0.01`, `0.1`, and `1.1` times the sum
of item values. The last coefficient is deliberately above the simple
sufficient bound `penalty > sum(values)` for a one-unit violation. Smaller
coefficients reveal when the penalty solver prefers an infeasible selection.
That bound characterizes the encoded global optimum; it does not guarantee that
a finite DMRG run reaches that optimum.

The projection-only resource table is fast to generate:

```bash
julia --project=benchmarks benchmarks/knapsack/run.jl --resources
```

It controls capacity, item count, and weight magnitude in separate sweeps. This
makes the projection's `capacity + 2` state bound visible: its bond dimension
tracks capacity but does not grow merely because there are more items or larger
integer weights. The table reports the projected Hamiltonian bond separately,
because that—not the projection alone—is the network used by constrained DMRG.
Use the main solver CSV's `solution_max_bond` column to compare this controlled
projection scaling with the observed penalty-QUBO DMRG bond growth.

## Output

Rows report:

- original knapsack value, feasibility, and gap from the exact feasible optimum;
- penalty factor, coefficient, and encoded objective for penalty-QUBO rows;
- formulation, solve, callback-Hamiltonian setup, sampling, and end-to-end wall
  times, with timing sample count and per-solve limit;
- solver-reported elapsed time, requested/actual sweep count, time-limit status,
  maximum solution MPS bond dimension, and final-state variance;
- GC time and allocations for the formulation and measured solve case;
- objective, projection, and effective projected-Hamiltonian MPO bond dimensions.

Every solver row also records the Julia/package versions, thread count, system,
architecture, seed, cutoff, reads, and caller-supplied benchmark identifier.
Runtime comparisons are meaningful only alongside feasibility, original-objective
gap, variance, and whether the time limit was reached; equal sweep counts do not
imply equal convergence quality.

The projection and effective-Hamiltonian bonds are separate because the latter
drives constrained DMRG cost. Final variance is calculated independently from
the per-sweep MPS supplied by `on_iteration`. Truncation error remains empty:
the callback runs after the DMRG sweep has discarded singular values, so that
error cannot be reconstructed from the retained MPS.
