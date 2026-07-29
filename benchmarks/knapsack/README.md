# Knapsack penalty vs projection benchmark

This benchmark compares two encodings of the same binary knapsack instances:

- a conventional unconstrained QUBO with bounded-binary slack variables and a
  squared capacity penalty;
- TenSolver's native `SumConstraint`, lowered to a hard projection MPO.

The comparison follows CoTenN's direct-constraint tensor-network framing
([Sharma et al., PLDI 2026](https://doi.org/10.1145/3808272)).

## Run

From the repository root, instantiate the environment once:

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'
```

Run the default benchmark:

```bash
julia --project=benchmarks benchmarks/knapsack/run.jl
```

The runner writes a CSV report to standard output. To change the workload or
save the report, call `run_benchmarks` from the REPL:

```julia
include("benchmarks/knapsack/KnapsackBenchmark.jl")
using .KnapsackBenchmark

report = run_benchmarks(
  [KnapsackBenchmark.reference_instance()];
  penalty_factors = (0.1, 1.1),
  iterations = 3,
  reads = 8,
  timing_samples = 1,
  time_limit = 30,
  benchmark_id = "pilot",
)

open("results.csv", "w") do io
  write_csv(io, report)
end
```

`run_benchmarks` builds a `BenchmarkTools.BenchmarkGroup` with one entry for
each instance and formulation. Each entry measures the complete formulation,
solve, and sampling path with one evaluation per sample.

## Instances and methods

The workload contains a four-item hand-checkable instance plus uncorrelated,
weakly correlated, strongly correlated, and subset-sum classes from Martello,
Pisinger, and Toth's standard 0-1 knapsack generator
([paper](https://doi.org/10.1287/mnsc.45.3.414),
[generator archive](https://hjemmesider.diku.dk/~pisinger/codes.html)).

The generated instances use one shared RNG and contain 8, 12, or 16 items.
Integer weights and capacity make the instances compatible with the projection
formulation, and their small size permits exact brute-force reference values.

For each instance, the suite benchmarks the projection formulation and penalty
factors `0.001`, `0.01`, `0.1`, and `1.1` times the sum of item values. Every
result is scored against the original knapsack objective and exact feasible
optimum.

## Resource scaling

Generate the projection resource table from the REPL:

```julia
include("benchmarks/knapsack/KnapsackBenchmark.jl")
using .KnapsackBenchmark

write_csv(stdout, projection_scaling_rows())
```

It controls capacity, item count, and weight magnitude in separate sweeps. The
table reports the projected Hamiltonian bond separately because it is the
network used by constrained DMRG. Use the main solver CSV's
`solution_max_bond` column to compare this controlled projection scaling with
the observed penalty-QUBO DMRG bond growth.

## Output

The report header records run-wide provenance and settings once: runtime and
package versions, thread count, system and architecture, RNG seed, sweep/read
limits, tensor cutoff, timing samples, and benchmark identifier.

Each result row records only case-specific information:

- instance, formulation, capacity, item count, and encoded variable count;
- penalty factor and coefficient for penalty-QUBO rows;
- original value, feasibility, and exact optimality gap;
- BenchmarkTools end-to-end time and solver-reported elapsed time;
- completed sweeps and soft-time-limit status;
- solution, objective, projection, and projected-Hamiltonian bond dimensions.
