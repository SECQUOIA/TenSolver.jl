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
TenSolver.PEPSBackend
TenSolver.SquareGrid
TenSolver.KingGrid
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
TenSolver.IsingModel
TenSolver.bool_to_spin
TenSolver.spin_to_bool
TenSolver.qubo_to_ising
TenSolver.ising_to_qubo
TenSolver.ising_energy
```

## Utility Functions

```@docs
Base.in(::AbstractVector, ::TenSolver.Solution)
Base.in(::AbstractVector, ::TenSolver.PEPSSolution)
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
