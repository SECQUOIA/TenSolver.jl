# PEPS Benchmarks

These scripts provide small, reproducible local comparisons between TenSolver's
default DMRG backend and the optional structured PEPS backend. The PEPS rows are
currently scaffolding for source/development environments where the SpinGlass
component stack resolves; in ordinary registered-package environments today,
expect the PEPS rows to be skipped.

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
  `SpinGlassTensors` are available and importable in the active Julia
  environment.

The scripts are not part of normal CI and are not intended to reproduce the full
SpinGlassPEPS arXiv benchmark suite.
