# API Reference

This page documents the public API of TenSolver.jl.

## Optimization Functions

```@docs
TenSolver.minimize
TenSolver.maximize
TenSolver.solve_ising
```

## Solver Backends

```@docs
TenSolver.AbstractTenSolverBackend
TenSolver.DMRGBackend
TenSolver.normalize_backend
TenSolver.PEPSBackend
```

## Solution

```@docs
TenSolver.Solution
TenSolver.PEPSSolution
```

## Sampling Functions

```@docs
TenSolver.sample
```

## Boolean/Spin Conversions

```@docs
TenSolver.bool_to_spin
TenSolver.spin_to_bool
TenSolver.qubo_to_ising
TenSolver.ising_to_qubo
```

## Constraints

```@docs
TenSolver.AbstractConstraint
TenSolver.SumConstraint
TenSolver.NotEqualsConstraint
TenSolver.ExactlyOneConstraint
TenSolver.RelationConstraint
TenSolver.is_feasible
TenSolver.constraint_sites
```

## Utility Functions

```@docs
Base.in(::AbstractVector, ::TenSolver.Solution)
Base.in(::AbstractVector, ::TenSolver.PEPSSolution)
```

## Internal Functions

These functions are part of the internal implementation and are not exported.
They are documented here for advanced users who may need to understand the internals.
Notice: As unexported method and types, they are subject to change without warning.

### Objective Construction

```@docs
TenSolver.tensorize
TenSolver.qmatrix_permutation
TenSolver.preprocess_qubo
```

### MPO Construction

```@docs
TenSolver.DFA
TenSolver.constraint_to_dfa
TenSolver.dfa_to_mpo
TenSolver.projection_mpo
TenSolver.projection_mpos
TenSolver.SparseTensorEntry
TenSolver.projection_entries
```

### Projected Hamiltonian Construction

```@docs
TenSolver.project_hamiltonian
TenSolver.project_state
```

### PEPS Backend


```@docs
TenSolver.SquareGrid
TenSolver.KingGrid
```

## Index

```@index
Pages = ["api.md"]
```
