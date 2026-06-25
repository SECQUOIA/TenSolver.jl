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
TenSolver.normalize_backend
```

## Solution

```@docs
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

```@docs
TenSolver.AbstractConstraint
TenSolver.SumConstraint
TenSolver.NotEqualsConstraint
TenSolver.ExactlyOneConstraint
TenSolver.RelationConstraint
TenSolver.is_feasible
```

## Utility Functions

```@docs
Base.in(::AbstractVector, ::TenSolver.Solution)
```

## Internal Functions

These functions are part of the internal implementation and are not exported. 
They are documented here for advanced users who may need to understand the internals.

```@docs
TenSolver.tensorize
TenSolver.itensor_from_nonzeros
TenSolver.projection_mpo
TenSolver.projection_mpos
TenSolver.qmatrix_permutation
TenSolver.preprocess_qubo
```

## Index

```@index
Pages = ["api.md"]
```
