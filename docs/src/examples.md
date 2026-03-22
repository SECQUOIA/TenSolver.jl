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

### Passing Solver Parameters to JuMP

You can pass solver-specific parameters to the optimizer using `set_attribute`:

```julia
using JuMP, TenSolver

dim = 40
Q   = randn(dim, dim)

model = Model(TenSolver.Optimizer)
@variable(model, x[1:dim], Bin)
@objective(model, Min, x' * Q * x)

# Set solver parameters
set_attribute(model, "iterations", 20)
set_attribute(model, "cutoff", 1e-10)
set_attribute(model, "maxdim", [10, 20, 50, 100, 200])
set_attribute(model, "noise", [1e-5, 1e-6, 1e-7, 0.0])
set_attribute(model, "time_limit", 60.0)
set_attribute(model, "verbosity", 1)

# Solve with custom parameters
optimize!(model)

solution = value.(x)
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

## Tracking Optimization Progress

Pass a pre-allocated `IterationSnapshot[]` vector to collect the MPS at each iteration.
This is useful for stochastic post-processing — e.g. analyzing how the energy distribution evolves:

```julia
using TenSolver, Serialization, Statistics

Q = randn(40, 40)

hist = IterationSnapshot[]
E, psi = TenSolver.minimize(Q; iterations=50, history=hist, save_every=5)

# Inspect convergence
for snap in hist
    xs = TenSolver.sample(snap.distribution, 200)
    energies = [x' * Q * x for x in xs]
    println("iter=$(snap.iteration)  mean=$(mean(energies))  std=$(std(energies))")
end

# Persist for later post-processing
serialize("history.jls", hist)
```

To load in a separate session:

```julia
using TenSolver, Serialization

hist = deserialize("history.jls")
xs   = TenSolver.sample(hist[end].distribution, 1000)
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

## Using MultivariatePolynomials.jl Interface (Experimental)

!!! warning "Experimental Feature"
    The MultivariatePolynomials.jl interface is experimental and may change in future versions.

TenSolver.jl can directly optimize polynomial objective functions defined using the MultivariatePolynomials.jl interface. This is particularly useful for problems that naturally express themselves as polynomial optimization.

```julia
using TenSolver
using DynamicPolynomials

# Define polynomial variables
@polyvar x[1:4]

# Create a polynomial objective function
# For example: x₁² + 2x₁x₂ + x₂² - x₃ + 3
polynomial = x[1]^2 + 2*x[1]*x[2] + x[2]^2 - x[3] + 3

# Minimize the polynomial
E, psi = TenSolver.minimize(polynomial)

# Sample a solution
solution = TenSolver.sample(psi)

# Evaluate the polynomial at the solution
# Note: solutions are in {0, 1}, indexed from 1 in Julia
objective_value = polynomial(x => solution)
```

This interface automatically handles the conversion of polynomial expressions into the tensor network representation used internally by the solver. The variables in the polynomial are treated as binary variables taking values in {0, 1}.

### Example: Graph Coloring

Here's an example of using the polynomial interface for a graph coloring problem:

```julia
using TenSolver
using DynamicPolynomials

# Define variables for a graph with 4 nodes
@polyvar x[1:4]

# Minimize the number of conflicts (edges with same color)
# For a simple graph: edges (1,2), (2,3), (3,4), (4,1)
polynomial = x[1]*x[2] + x[2]*x[3] + x[3]*x[4] + x[4]*x[1]

# Solve
E, psi = TenSolver.minimize(polynomial)
coloring = TenSolver.sample(psi)

println("Graph coloring: ", coloring)
println("Number of conflicts: ", E)
```
