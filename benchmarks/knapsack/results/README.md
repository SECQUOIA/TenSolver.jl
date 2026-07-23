# Knapsack benchmark result: 2026-07-22

This directory records a complete bounded run of the penalty-versus-projection
benchmark. The raw measurements are in
[`2026-07-22-656491d.csv`](2026-07-22-656491d.csv).

## Reproduction

- Benchmark code: `656491d6daeaaa17ec6cd22b32014a7b5d376374`
- Machine: Intel Xeon w5-2565X, Linux x86-64
- Julia: 1.12.6, one Julia thread
- Packages: TenSolver 0.2.3, ITensors 0.9.30, ITensorMPS 0.4.1,
  BenchmarkTools 1.8.0
- Workload: six DMRG sweeps, 64 reads, three fixed-seed timing samples, and a
  120-second soft limit per solve

From the repository root, after instantiating the benchmark environment, the
recorded command was:

```bash
julia +1.12 --project=benchmarks benchmarks/knapsack/run.jl \
  --iterations=6 --reads=64 --timing-samples=3 --time-limit=120 \
  --benchmark-id=656491d6daeaaa17ec6cd22b32014a7b5d376374 \
  benchmarks/knapsack/results/2026-07-22-656491d.csv
```

All 25 rows completed all six requested sweeps without reaching the time limit.

## Results

"Best feasible penalty" selects the penalty row with the highest original
knapsack value. The final column compares projection time with the median time
over all four penalty factors; it is not an equal-quality speedup.

| Instance | Items | Exact / projection value | Projection (s) | Best feasible penalty: factor, value, gap, time (s) | Projection / median penalty time |
| --- | ---: | ---: | ---: | --- | ---: |
| Reference | 4 | 13 / 13 | 0.0579 | 0.1, 13, 0, 0.0514 | 1.18x |
| Uncorrelated | 8 | 41 / 41 | 0.1031 | 0.01, 37, 4, 0.1403 | 0.57x |
| Weakly correlated | 12 | 36 / 36 | 0.2439 | 0.01, 31, 5, 0.2615 | 0.87x |
| Strongly correlated | 16 | 51 / 51 | 1.3467 | 0.01, 44, 7, 0.4891 | 2.59x |
| Subset sum | 16 | 44 / 44 | 1.4662 | 0.01, 35, 9, 0.5752 | 2.78x |

Projection returned the exact feasible optimum on all five instances. Of the
20 penalty rows, 14 were feasible and only the reference instance at factor
0.1 was exact. Every factor-0.001 row was infeasible; larger factors restored
feasibility but did not guarantee that the six-sweep DMRG run reached the
encoded global optimum.

The warm, repeated measurements do not reproduce the earlier compilation-heavy
slowdown. Projection and penalty are comparable on the reference instance,
projection is faster than the median penalty row at 8 and 12 items, and it
becomes slower on both 16-item instances. The projected Hamiltonian bond grows
to 54 and 37 on those two cases, versus 3 for the penalty Hamiltonians. Observer
setup accounts for only 0.120 and 0.066 seconds respectively; even after
subtracting it, the projection solver paths take 1.214 and 1.390 seconds. This
points to the larger effective projected Hamiltonian and its contractions, not
callback construction or first-use compilation, as the dominant measured cost.

These runtime comparisons must remain paired with solution quality: none of the
faster 16-item penalty rows reaches the projection result. The data supports a
quality-versus-cost tradeoff for this bounded workload, not a general claim
that either formulation is faster.

## Truncation-error limitation

`truncation_error` is intentionally empty. TenSolver's public iteration callback
runs after a sweep has discarded singular values, so the discarded weight
cannot be reconstructed from the retained MPS. This limitation is acceptable
for this benchmark because feasibility, exact optimality gap, independently
calculated final variance, bond dimensions, allocations, and phase timings are
all recorded. It does prevent using this dataset to attribute a poor result
specifically to truncation, so the empty field and that narrower conclusion
must remain explicit.

This is a single-machine, single-thread, small-instance run with three timing
samples and a fixed sweep budget. It is reproducible evidence for this PR, not
a broad performance characterization.
