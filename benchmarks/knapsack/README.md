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

From the repository root:

```bash
julia --project=. benchmarks/knapsack/run.jl
```

Pass one path to also save the CSV output:

```bash
julia --project=. benchmarks/knapsack/run.jl results.csv
```

The fixed workload contains a four-item hand-checkable instance and random
families of 8, 12, and 16 items generated from seed 66. Random weights are
integers from 1 through 10, values are integers from 1 through 15, and capacity
is 40% of total weight rounded to the nearest integer. Integer weights and
capacity are required by the projection formulation. Every instance is checked
against a brute-force constrained optimum. Heavy benchmark execution is
intentionally outside normal CI; CI only checks the encoding, exact reference,
seed reproducibility, and small MPO resource construction.

Before collecting rows, the runner performs one unreported two-item solve with
each formulation to reduce one-sided Julia compilation bias in the timing.

For each instance the runner executes the projection formulation once and the
penalty formulation at factors `0.001`, `0.01`, `0.1`, and `1.1` times the sum
of item values. The last coefficient is deliberately above the simple
sufficient bound `penalty > sum(values)` for a one-unit violation. Smaller
coefficients reveal when the penalty solver prefers an infeasible selection.
That bound characterizes the encoded global optimum; it does not guarantee that
a finite DMRG run reaches that optimum.

The projection-only resource table is fast to generate:

```bash
julia --project=. benchmarks/knapsack/run.jl --resources
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
- wall time, solver-reported elapsed time, sweep count, and maximum solution MPS
  bond dimension;
- objective, projection, and effective projected-Hamiltonian MPO bond dimensions.

The projection and effective-Hamiltonian bonds are separate because the latter
drives constrained DMRG cost. Variance and truncation error are not reported:
the current public `Solution` records energies, bond dimensions, and elapsed
times, but does not expose those diagnostics.
