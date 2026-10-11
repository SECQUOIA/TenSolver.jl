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
TenSolver.Solution
TenSolver.DMRGSolution
TenSolver.SolverStatistics
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
TenSolver.SumModConstraint
TenSolver.NotEqualsConstraint
TenSolver.AssignmentConstraint
TenSolver.RelationConstraint
TenSolver.is_feasible
TenSolver.constraint_sites
```

## Utility Functions

```@docs
Base.in(::AbstractVector, ::TenSolver.Solution)
TenSolver.permute
```

## Internal Functions

These functions are part of the internal implementation and are not exported.
They are documented here for advanced users who may need to understand the internals.
Notice: As unexported method and types, they are subject to change without warning.

### Objective Construction

```@docs
TenSolver.tensorize
TenSolver.qmatrix_permutation
TenSolver.preprocess_model
```


### MPO Construction

```@docs
TenSolver.DFA
TenSolver.constraint_to_dfa
TenSolver.mapreduce_dfa
TenSolver.dfa_to_mpo
TenSolver.projection_mpo
TenSolver.projection_mpos
```

### Projected Hamiltonian Construction

```@docs
TenSolver.project_hamiltonian
TenSolver.project_state
```

### Variable Domains

```@docs
TenSolver.Domains
TenSolver.domain_residue
```

### PEPS Backend

The optional structured backend requires Julia 1.11 or later and registered
`SpinGlassPEPS` 2.x. Install it with `Pkg.add("SpinGlassPEPS")`, then load it
alongside TenSolver to activate the extension. DMRG remains the default backend.
The PEPS types are unexported and accessed through the `TenSolver` namespace.

```julia
using TenSolver
import SpinGlassPEPS

backend = TenSolver.PEPSBackend(TenSolver.SquareGrid(2, 2))
J = [0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.0;
     0.0 0.0 0.0 0.25; 0.0 0.0 0.0 0.0]
h = [-1.0, -0.25, 0.25, -0.75]
energy, solution = minimize(J, h, 0.125; domain = [-1, 1], backend,
                            device = TenSolver.cpu, transformations = :identity,
                            verbosity = 0)
# energy ≈ -2.375; sample(solution) returns native spins in {-1, 1}.
```

Retained states use the shared solution interface (`sample`, `prob`, membership,
and `is_feasible`). Their probabilities sum to one over the retained unique
states. Raw upstream log probabilities and search diagnostics remain in result
metadata. PEPS inputs must fit the specified topology and use the spin domain
`[-1, 1]`; native constraints are unsupported.

```@docs
TenSolver.SquareGrid
TenSolver.KingGrid
```

## Index

```@index
Pages = ["api.md"]
```
