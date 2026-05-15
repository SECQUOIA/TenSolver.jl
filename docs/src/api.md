# API Reference

This page documents the public API of TenSolver.jl.

## Optimization Functions

```@docs
TenSolver.minimize
TenSolver.maximize
```

## Solution

```@docs
TenSolver.Solution
```

## Sampling Functions

```@docs
TenSolver.sample
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
TenSolver.qmatrix_permutation
```

## Index

```@index
Pages = ["api.md"]
```
