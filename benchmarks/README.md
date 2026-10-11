# PEPS Benchmarks

These scripts provide small, reproducible local comparisons between TenSolver's
default DMRG backend and the optional structured PEPS backend. The PEPS rows run
when registered SpinGlassPEPS 2.x is installed in the active environment on Julia
1.11 or later. An absent package produces a skipped row; loading or solving
failures produce error rows.

Set up the benchmark environment from the repository root:

```julia
using Pkg
Pkg.activate("benchmarks")
Pkg.develop(PackageSpec(path = "."))
Pkg.instantiate()
# Optional; requires Julia 1.11 or later:
Pkg.add(PackageSpec(name = "SpinGlassPEPS", version = "2"))
```

Run from the repository root in that environment:

```bash
julia --project=benchmarks benchmarks/peps_square.jl
julia --project=benchmarks benchmarks/peps_king.jl
```

The instances are intentionally tiny:

- brute force is used as a reference objective value;
- random seeds are fixed;
- CPU execution is the default;
- each script should finish in seconds to a few minutes on a laptop, depending
  on Julia precompilation and whether PEPS is installed;
- the scripts load `SpinGlassPEPS` to activate TenSolver's optional extension;
- `discarded_log` preserves the upstream largest-discarded log probability,
  including `-Inf` when no branch was discarded;
- times include compilation on the first solve and are exploratory diagnostics,
  not a performance ranking.

The scripts are not part of normal CI and are not intended to reproduce the full
SpinGlassPEPS arXiv benchmark suite.
