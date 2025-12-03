# Examples

This page provides practical examples of using TenSolver.jl for various optimization problems.

## Basic QUBO Problem

The most basic usage is solving a quadratic unconstrained binary optimization problem:

```julia
using TenSolver

# Define a random QUBO problem
Q = randn(40, 40)

# Solve for minimum
E, psi = TenSolver.minimize(Q)

# Sample a solution
x = TenSolver.sample(psi)

# Verify the solution
energy = x' * Q * x
```

## QUBO with Linear and Constant Terms

You can also specify linear and constant terms:

```julia
using TenSolver

n = 40
Q = randn(n, n)
l = randn(n)
c = 5.0

# Solve: min b'Qb + l'b + c
E, psi = TenSolver.minimize(Q, l, c)

# Sample multiple solutions
samples = [TenSolver.sample(psi) for _ in 1:10]
```

## Using with JuMP

TenSolver.jl provides an optimizer interface for JuMP models:

```julia
using JuMP, TenSolver

dim = 40
Q   = randn(dim, dim)

# Create a JuMP model with TenSolver
model = Model(TenSolver.Optimizer)
@variable(model, x[1:dim], Bin)
@objective(model, Min, x' * Q * x)

# Solve the optimization problem
optimize!(model)

# Get the solution
solution = value.(x)
objective_value = objective_value(model)
```

## Controlling Solver Parameters

You can control various solver parameters for better performance:

```julia
using TenSolver

Q = randn(40, 40)

# Solve with custom parameters
E, psi = TenSolver.minimize(
    Q;
    iterations = 20,        # More iterations for better convergence
    cutoff = 1e-10,         # Higher accuracy
    maxdim = [10, 20, 50, 100, 200],  # Bond dimension schedule
    noise = [1e-5, 1e-6, 1e-7, 0.0],  # Noise schedule for convergence
    time_limit = 60.0       # Stop after 60 seconds
)
```

## Running on GPU

TenSolver.jl supports GPU acceleration for faster computation:

```julia
using TenSolver
import CUDA

Q = randn(100, 100)

# Solve on GPU using CUDA
E, psi = TenSolver.minimize(Q; device = CUDA.cu)

# Sample solutions
x = TenSolver.sample(psi)
```

For Apple Silicon GPUs:

```julia
using TenSolver
import Metal

Q = randn(100, 100)

# Solve on GPU using Metal
E, psi = TenSolver.minimize(Q; device = Metal.mtl)
```

## Maximization Problems

To solve maximization problems instead of minimization:

```julia
using TenSolver

Q = randn(40, 40)

# Solve for maximum instead of minimum
E, psi = TenSolver.maximize(Q)

x = TenSolver.sample(psi)
```

## Checking Solution Probability

You can check if a specific solution is in the support of the distribution:

```julia
using TenSolver

Q = randn(40, 40)
E, psi = TenSolver.minimize(Q)

# Sample a solution
x = TenSolver.sample(psi)

# Check if x is a likely solution (probability > cutoff)
is_valid = x in psi  # Uses default cutoff of 1e-8

# Or with custom cutoff
is_valid = x in (psi, cutoff=1e-6)
```

## Multiple Samples

Generate multiple independent samples from the solution distribution:

```julia
using TenSolver

Q = randn(40, 40)
E, psi = TenSolver.minimize(Q)

# Generate 1000 samples
num_samples = 1000
samples = [TenSolver.sample(psi) for _ in 1:num_samples]

# Find the best sample
energies = [s' * Q * s for s in samples]
best_idx = argmin(energies)
best_solution = samples[best_idx]
best_energy = energies[best_idx]
```
