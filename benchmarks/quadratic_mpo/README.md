# Quadratic objectives and native constraints

Run from an environment containing this checkout:

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=benchmarks benchmarks/quadratic_mpo/run.jl > results.csv
```

The script uses the public `minimize` API to compare the same exactly-one
assignment problem represented by a quadratic penalty or native constraints.
Both formulations have the independently known optimum printed in the output.
It warms both paths, uses three fixed seeds, records three sweeps per solve, and
evaluates 32 final samples against the original native constraints and cost.

Times cover the complete public solve call, including objective construction,
projection and DMRG. Allocations are total bytes allocated, not peak memory.
Bond dimensions and feasible-sample costs accompany every timing; finite-sweep
solutions are not assumed optimal. These fixtures use contiguous disjoint groups,
so they do not measure the bond growth possible with overlapping constraints.
Run the same script and dependencies on the base and candidate to measure a
package-level change. Keep generated CSV output outside version control.

Constructor-only routing measurements in the PR are separate from this public
API benchmark. They isolate objective construction and do not imply the same
speedup for complete constrained solves.
