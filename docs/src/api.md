# API Reference

This page documents the public API of TenSolver.jl.

## Optimization Functions

```@docs
TenSolver.minimize
TenSolver.maximize
```

## Solver Backends

```@docs
TenSolver.AbstractTenSolverBackend
TenSolver.DMRGBackend
TenSolver.PEPSBackend
TenSolver.normalize_backend
```

## Solution

```@docs
TenSolver.AbstractSolution
TenSolver.Solution
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

Hard constraints are enforced by lowering each one to an exact projection MPO,
following CoTenN (Sharma, Peng, Dangwal, and Achour, *"CoTenN: Constrained
Optimization with Tensor Networks,"* PLDI 2026). See [Constrained Optimization](@ref)
for a worked example.

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
TenSolver.solve_ising
TenSolver.peps_options
```

## Index

```@index
Pages = ["api.md"]
```
