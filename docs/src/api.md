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
TenSolver.GTNSolution
```

## Sampling Functions

```@docs
TenSolver.sample
```

## Backends

```@docs
TenSolver.AbstractBackend
TenSolver.DMRGBackend
TenSolver.GTNBackend
TenSolver.solution_space
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
```

## Index

```@index
Pages = ["api.md"]
```
