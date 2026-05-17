# PEPS Benchmarks

These scripts provide small, reproducible local comparisons between TenSolver's
default DMRG backend and the optional structured PEPS backend.

Run from the repository root:

```bash
julia --project=. benchmarks/peps_square.jl
julia --project=. benchmarks/peps_king.jl
```

The instances are intentionally tiny:

- brute force is used as a reference objective value;
- random seeds are fixed;
- CPU execution is the default;
- each script should finish in seconds to a few minutes on a laptop, depending
  on Julia precompilation and whether PEPS is installed;
- PEPS is skipped unless `SpinGlassNetworks`, `SpinGlassEngine`, and
  `SpinGlassTensors` are available in the active Julia environment.

The scripts are not part of normal CI and are not intended to reproduce the full
SpinGlassPEPS arXiv benchmark suite.
