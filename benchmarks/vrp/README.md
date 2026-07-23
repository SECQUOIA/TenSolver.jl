# VRP benchmark data

The eight benchmark instances used by the QInnovision World Challenge 2025
experiments remain in their canonical data repository instead of being copied
into the TenSolver package tree. Download the exact, pinned files with:

```sh
julia benchmarks/vrp/download_data.jl
```

The script writes to `benchmarks/vrp/data/` by default. It accepts another
destination as its sole argument, verifies every file against the SHA-256
checksum recorded in the script, and leaves an existing file untouched only
when its checksum is correct.

## Provenance and license

The files are byte-for-byte copies of the `TestSet/` files at commit
[`78c1e390`](https://github.com/SECQUOIA/vrp-qinnovision/tree/78c1e390309127a6be09b11b599f25e435d9f324/TestSet)
of [`SECQUOIA/vrp-qinnovision`](https://github.com/SECQUOIA/vrp-qinnovision).
That repository identifies them as instances generated with
[`smharwood/vrp-as-qubo`](https://github.com/smharwood/vrp-as-qubo) for time
horizons `16, 21, 26, 31, 36, 41, 46, 51`.

The canonical data repository is distributed under the
[MIT License](https://github.com/SECQUOIA/vrp-qinnovision/blob/78c1e390309127a6be09b11b599f25e435d9f324/LICENSE),
Copyright (c) 2024 SECQUOIA. The downloaded files remain subject to that
license and notice, which is compatible with TenSolver's MIT license.

The formulation and generator are described by:

> Stuart Harwood, Claudio Gambella, Dimitar Trenev, Andrea Simonetto,
> David Bernal, and Donny Greenberg. "Formulating and Solving Routing Problems
> on Quantum Computers." *IEEE Transactions on Quantum Engineering* 2 (2021),
> 1–17.

These instances were not produced for the NeurIPS ScaleOPT 2025 TenSolver
paper, whose numerical study used QPLib instances.

## File names

The generator names files `test_<formulation>_<variables>_<class>.rudy`.
For the files selected here:

- `test` marks a generated test-set instance;
- `pb` means the path-based vehicle-routing formulation;
- the integer (`10` through `794`) is the number of spin variables; and
- `o` means the objective-bearing optimization instance, as opposed to an `f`
  feasibility-only instance whose optimum is zero.

## Rudy-style format

Although the source repository informally calls these Q matrices, the files
were exported by `QUBOContainer.export(..., as_ising=True)` and encode Ising
objectives over spins `s_i ∈ {-1, +1}`:

```text
E(s) = constant + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j
```

Each file contains:

1. a generated-at comment;
2. a `# Constant term of objective = ...` comment;
3. `# Diagonal terms`, followed by `i i h_i`; and
4. `# Off-Diagonal terms`, followed by `i j J_ij`.

Indices are zero-based. The interaction matrix is stored sparsely in
upper-triangular form, so an off-diagonal pair appears once rather than in both
orders. Coefficients are written to two decimal places.
