# TenSolver.jl Conventions

Conventions for contributors and automated agents working on this repository,
distilled from the maintainers' review history. When in doubt, follow the
existing code and ask in an issue before diverging.

## Code style

- Run JuliaFormatter before committing; `.JuliaFormatter.toml` is authoritative
  (2-space indent, 92-column margin, explicit `return`, `if` blocks instead of
  `||`/`&&` guards, aligned fields, trailing commas, semicolon-separated kwargs).
- No leading underscores on function or variable names. Privacy is expressed by
  not exporting, never by naming.
- Export only user-facing entry points. Do not export internal helpers or
  abstract backend types; do not pollute the caller's namespace.
- Prefer multiple dispatch over parallel per-type functions: when two methods
  differ only in a type, write one method on the abstract supertype.
- Encode invariants in the type system (for example `SumConstraint{<:Integer}`)
  instead of runtime validation. Validate at public boundaries and constructors
  only; internal code assumes the invariants.
- Keyword defaults belong only on the public API (`minimize`). Internal
  functions take every argument explicitly — internal code needs no ergonomy.
- Prefer neutral-element representations (identity permutation, empty vector)
  over `nothing` sentinels threaded through `isnothing` branches.
- Performance idioms: no `try`/`catch` inside hot loops, no `Any`-typed struct
  fields, do not `collect` lazy iterators except at boundaries, keep functions
  type-stable (assert stability claims with `Test.@inferred`).
- Report expected outcomes such as infeasibility as a status on the returned
  solution plus `@warn`/`@info` logs, matching solver-ecosystem conventions —
  not as exceptions. When throwing, use the semantically correct exception type.

## Module layout

- One concern per file in `src/` (`constraints.jl`, `solver.jl`,
  `projection_mpo.jl`, `solution.jl`, ...).
- Backend-specific code, defaults, and documentation live in
  `src/backends/<name>.jl`. The generic solver layer references only the
  abstract backend interface; backends do not know about each other.

## Documentation

- Docstrings: signature line first, then what/why only. Never describe how the
  current implementation works, restate default Julia behavior, or explain what
  the code does *not* do. Keep them concise.
- Use LaTeX or pseudo-math notation for mathematical content, index-set
  notation for site/index arguments, and `See also [`X`](@ref)` cross-links.
  Keep parallel entries (constraint tables, bond-dimension notes) synchronized.
- Internal design notes go to `docs/internal/`, not the rendered site or README.

## Tests

- One topic per file, registered in the commented `include` list in
  `test/runtests.jl`; shared helpers in `test/utils.jl`; nested `@testset`s
  with descriptive names.
- Hard-code known optima in assertions; do not recompute them with brute force.
- Pass `verbosity = 0` to solves inside tests.
- No trivial tests (for example, checking that a constructor sets fields).
  Backend tests exercise backend infrastructure only and run no solves.
- Rely on `@testset`'s RNG reproducibility; never set per-instance seeds.
- Keep semantically distinct dual assertions: `x0 in psi` (all optima found)
  and `sample(psi) in optima` (only optima returned) test different failures.
- Doctests assert objective values or samples deterministically; no random
  doctests.
- Test-only dependencies go in `test/Project.toml`, docs dependencies in
  `docs/Project.toml` — never in the package `Project.toml`.

## Scope and features

- A feature users can compose from the public API (calling `minimize` in a
  loop, a callback hook) stays in user land. Features must align with the
  package's core value proposition: GPU-capable tensor-network solving.
- One issue per PR; no opportunistic additions. A cleanup or trim PR should
  shrink its target area.
- Never commit generated files (benchmark CSVs, outputs). Benchmarks live in
  `benchmarks/` with their own `Project.toml`, depend on TenSolver as if it
  were an external package, and use only the public API.
- While the package is pre-1.0, delete superseded or unreleased APIs in the
  same PR instead of deprecating them.

## Commits and PRs

- Commit messages: lowercase `feat:`, `fix:`, `refactor(scope):`, `tests:`,
  `docs:` prefixes, imperative mood, one logical change per commit; suffix
  `(WIP)` when incomplete. Release commits read `Release version vX.Y.Z`.
- PR titles are plain imperative sentences. Reference related issues and PRs,
  and honor decisions already agreed in sibling threads.

## CI and releases

- CI runs the `lts`/`1`/`pre` × Linux/macOS/Windows matrix. Optimize test
  wall-clock per job (testset redundancy), not matrix breadth.
- Release flow: a release PR bumping the `Project.toml` version and refreshing
  stale TenSolver compat pins in `docs/Project.toml` and
  `benchmarks/Project.toml`; merge on green CI; comment
  `@JuliaRegistrator register` on the merge commit with a `Release notes:`
  bullet list; TagBot creates the tag and GitHub release.
