# Examples

This page provides practical examples of using TenSolver.jl for various optimization problems.

## Basic QUBO Problem

The most basic usage is solving a quadratic unconstrained binary optimization problem:

```jldoctest basic
using TenSolver

assets = ["Wind", "Solar", "Battery"]

# Wind and solar overlap, while storage pairs well with either source.
Q = [0.0 1.0 -1.5;
     1.0 0.0 -0.5;
     -1.5 -0.5 0.0]

# Solve for minimum
E, psi = TenSolver.minimize(Q; verbosity=0)

# Sample a solution
x = TenSolver.sample(psi)

# Verify the solution achieves the minimum energy and inspect the chosen bundle
(E ≈ -3.0, join(assets[findall(==(1), x)], ", "))

# output

(true, "Wind, Battery")
```

## Ising Model

Use `domain = [-1, 1]` when the variables are Ising spins:

```jldoctest ising-domain
using TenSolver

J = [0.0 -1.0;
     -1.0 0.0]
h = [0.25, -0.5]

E, psi = TenSolver.minimize(J, h; domain = [-1, 1], verbosity = 0)
s = TenSolver.sample(psi)

(E ≈ -2.25, s, all(in((-1, 1)), s))

# output

(true, [1.0, 1.0], true)
```

## Non-consecutive Integer Domains

Use any finite integer domain:

```jldoctest sparse-domain
using TenSolver

l = [1.0, -4.0, 2.0]

E, psi = TenSolver.minimize(l; domain = [-2, 0, 3], verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ -18.0, x, psi.domain)

# output

(true, [-2.0, 3.0, -2.0], [-2.0, 0.0, 3.0])
```

## Fractional Domains

Unconstrained DMRG optimization accepts any finite domain of real values,
including fractional values:

```jldoctest fractional-domain
using TenSolver

l = [-2.0, 3.0]

E, psi = TenSolver.minimize(l; domain = [0.0, 0.5, 1.0], verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ -2.0, x, psi.domain)

# output

(true, [1.0, 0.0], [0.0, 0.5, 1.0])
```

Hard constraints can impose narrower domain requirements. In particular,
[`SumConstraint`](@ref) currently requires a nonnegative integer domain.

## QUBO with Linear and Constant Terms

You can also specify linear and constant terms:

```jldoctest linear
using TenSolver

assets = ["Wind", "Solar", "Battery"]

# Linear terms encode stand-alone value, quadratic terms encode overlap penalties.
Q = [0.0 2.0 0.0;
     2.0 0.0 0.5;
     0.0 0.5 0.0]
l = [-3.0, -2.0, -2.0]
c = 5.0

# Solve: min b'Qb + l'b + c
E, psi = TenSolver.minimize(Q, l, c; verbosity=0)
x = TenSolver.sample(psi)

# The best plan keeps wind and storage, but skips solar because of overlap penalties.
(E ≈ 0.0, join(assets[findall(==(1), x)], ", "))

# output

(true, "Wind, Battery")
```

## Using with JuMP

TenSolver.jl provides an optimizer interface for JuMP models:

```jldoctest jump
using JuMP, TenSolver

assets = ["Wind", "Solar", "Battery"]

Q = [0.0 2.0 0.0;
     2.0 0.0 0.5;
     0.0 0.5 0.0]
l = [-3.0, -2.0, -2.0]
c = 5.0

# Create a JuMP model with TenSolver
model = Model(TenSolver.Optimizer)
set_silent(model)
@variable(model, x[1:3], Bin)
@objective(
    model,
    Min,
    sum(Q[i, j] * x[i] * x[j] for i in 1:3, j in 1:3) +
    sum(l[i] * x[i] for i in 1:3) + c
)

# Solve the optimization problem
optimize!(model)
selected = assets[findall(>(0.5), value.(x))]

# Verify the solution is optimal
(objective_value(model) ≈ 0.0, join(selected, ", "))

# output

(true, "Wind, Battery")
```

### Passing Solver Parameters to JuMP

You can pass solver-specific parameters to the optimizer using `set_attribute`:

```jldoctest jump_params
using JuMP, TenSolver

assets = ["Wind", "Solar", "Battery"]

Q = [0.0 2.0 0.0;
     2.0 0.0 0.5;
     0.0 0.5 0.0]
l = [-3.0, -2.0, -2.0]
c = 5.0

model = Model(TenSolver.Optimizer)
set_silent(model)
@variable(model, x[1:3], Bin)
@objective(
    model,
    Min,
    sum(Q[i, j] * x[i] * x[j] for i in 1:3, j in 1:3) +
    sum(l[i] * x[i] for i in 1:3) + c
)

# Set solver parameters
set_attribute(model, "iterations", 20)
set_attribute(model, "cutoff", 1e-10)
set_attribute(model, "maxdim", [10, 20, 50, 100, 200])
set_attribute(model, "noise", [1e-5, 1e-6, 1e-7, 0.0])
set_attribute(model, "time_limit", 60.0)
set_attribute(model, "verbosity", 0)

# Solve with custom parameters
optimize!(model)
selected = assets[findall(>(0.5), value.(x))]

# Verify solution
(objective_value(model) ≈ 0.0, join(selected, ", "))

# output

(true, "Wind, Battery")
```

## Controlling Solver Parameters

You can control various solver parameters for better performance:

```jldoctest params
using TenSolver

assets = ["Wind", "Solar", "Battery"]

Q = [0.0 1.0 -1.5;
     1.0 0.0 -0.5;
     -1.5 -0.5 0.0]

# Solve with custom parameters
E, psi = TenSolver.minimize(
    Q;
    iterations = 20,        # More iterations for better convergence
    cutoff = 1e-10,         # Higher accuracy
    maxdim = [10, 20, 50, 100, 200],  # Bond dimension schedule
    noise = [1e-5, 1e-6, 1e-7, 0.0],  # Noise schedule for convergence
    time_limit = 60.0,      # Stop after 60 seconds
    verbosity = 0
)
x = TenSolver.sample(psi)

# Verify we get the expected minimum
(E ≈ -3.0, join(assets[findall(==(1), x)], ", "))

# output

(true, "Wind, Battery")
```

## Tracking Optimization Progress

The returned `Solution` always carries lightweight per-iteration stats:

```julia
using TenSolver

Q = randn(40, 40)
E, psi = TenSolver.minimize(Q; iterations=50)

psi.energies       # objective value at each iteration
psi.bond_dims      # MPS bond dimension at each iteration
psi.elapsed_times  # wall-clock time at each iteration
```

For per-iteration sampling, pass an `on_iteration` callback.
The callback receives the MPS for that iteration alongside metadata as keyword arguments.
In this example, 200 bitstrings are sampled at each recorded iteration and their objective
values are stored in a dictionary, which is then serialized to disk:

```julia
using TenSolver, ITensorMPS, Serialization, Statistics

Q = randn(40, 40)

results = Dict{Int, Vector{Float64}}()
function cb(mps; iteration, kw...)  # kw... absorbs unused kwargs (objective, bond_dim, elapsed_time)
    xs = ITensorMPS.sample!(mps) .- 1
    results[iteration] = [x' * Q * x for x in xs]
end

E, psi = TenSolver.minimize(Q; iterations=50, on_iteration=cb, callback_every=5)

# Persist derived statistics (not the MPS) for later post-processing
serialize("results.jls", results)
```

To save the full MPS at each recorded iteration, use HDF5 (ITensors' native format, more
stable across versions than `Serialization`). Each iteration is stored as a named group
inside a single file:

```julia
using TenSolver, HDF5

Q = randn(40, 40)

function cb(mps; iteration, kw...)
    h5open("snapshots.h5", "cw") do f
        g = create_group(f, "iter_$iteration")
        write(g, "mps", mps)
    end
end

E, psi = TenSolver.minimize(Q; iterations=50, on_iteration=cb, callback_every=5)
```

To load a snapshot in a later session:

```julia
using TenSolver, HDF5, ITensorMPS

mps = h5open("snapshots.h5", "r") do f
    read(f["iter_25"], "mps", MPS)
end

xs = [ITensorMPS.sample!(mps) .- 1 for _ in 1:1000]
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

```jldoctest maximize
using TenSolver

campaigns = ["Email", "Webinar", "Follow-up"]

Q = [0.0 0.5 0.0;
     0.5 0.0 0.5;
     0.0 0.5 0.0]

# Solve for maximum instead of minimum
E, psi = TenSolver.maximize(Q; verbosity=0)
x = TenSolver.sample(psi)

# Verify we get the expected maximum
(E ≈ 2.0, join(campaigns[findall(==(1), x)], ", "))

# output

(true, "Email, Webinar, Follow-up")
```

## Multiple Samples

Generate multiple independent samples from the solution distribution:

```jldoctest samples
using TenSolver

# Two single-route plans are equally good, but taking both routes together is wasteful.
Q = [0.0 1.0; 1.0 0.0]
l = [-1.0, -1.0]
E, psi = TenSolver.minimize(Q, l, 0.0; verbosity=0)

# Generate multiple samples
num_samples = 10
samples = [TenSolver.sample(psi) for _ in 1:num_samples]
optimal_plans = [[1, 0], [0, 1]]

# All samples should be one of the optimal plans, and both plans are represented in psi
([1, 0] in psi, [0, 1] in psi,
 all(s -> s in optimal_plans && (s' * Q * s + l' * s) ≈ E, samples))

# output

(true, true, true)
```

## Using MultivariatePolynomials.jl Interface (Experimental)

!!! warning "Experimental Feature"
    The MultivariatePolynomials.jl interface is experimental and may change in future versions.

TenSolver.jl can directly optimize polynomial objective functions defined using the MultivariatePolynomials.jl interface. This is particularly useful for problems that naturally express themselves as polynomial optimization.

```jldoctest polynomial
using TenSolver
using DynamicPolynomials

# Define polynomial variables
@polyvar x[1:4]

# Create a polynomial objective function
# For example: x₁² + 2x₁x₂ + x₂² - x₃ + x₄
polynomial = 1.0*x[1]^2 + 2.0*x[1]*x[2] + 1.0*x[2]^2 - 1.0*x[3] + 1.0*x[4]

# Minimize the polynomial
E, psi = TenSolver.minimize(polynomial; verbosity=0)

# Sample a solution
solution = TenSolver.sample(psi)

# Evaluate the polynomial at the solution
# Note: solutions are in {0, 1}, indexed from 1 in Julia
objective_value = polynomial(x => solution)

# Verify the sampled solution achieves the reported minimum
objective_value ≈ E

# output

true
```

This interface automatically handles the conversion of polynomial expressions into the tensor network representation used internally by the solver. The variables in the polynomial are treated as binary variables taking values in {0, 1}.

## Optional GenericTensorNetworks Backend

TenSolver can use [GenericTensorNetworks.jl](https://github.com/QuEraComputing/GenericTensorNetworks.jl)
as an optional exact backend for QUBO/PUBO objectives already expressible through TenSolver.
This does not expose GenericTensorNetworks' native structured problem types through TenSolver.

```julia
using TenSolver
using GenericTensorNetworks, ProblemReductions

Q = [0.0 2.0; 0.0 0.0]
l = [-1.0, -1.0]
c = 3.0

E, sol = TenSolver.minimize(Q, l, c; backend=:gtn)
TenSolver.sample(sol)  # one exact optimum bitstring

E_count, count_sol = TenSolver.minimize(Q, l, c; backend=:gtn, property=:count)
count_sol.metadata["count"]  # exact optimum degeneracy for this instance
```

GTN settings are passed as solve keywords, for example
`minimize(Q; backend=:gtn, property=:single, k=3)`. The first return value is the
true optimum for every supported property. The backend object itself carries no tunable
state. The default backend remains `TenSolver.DMRGBackend()`. GTN-backed solves return a
[`TenSolver.GTNSolution`](@ref), which stores exact solution-space data when the selected
property produces configurations. MPS-specific APIs such as coefficient/probability queries
remain specific to the default DMRG [`TenSolver.Solution`](@ref).

### Example: Graph Coloring

Here's an example of using the polynomial interface for a graph coloring problem:

```jldoctest graphcolor
using TenSolver
using DynamicPolynomials

# Define variables for a graph with 4 nodes
@polyvar x[1:4]

# Minimize the number of conflicts (edges with same color)
# For a simple graph: edges (1,2), (2,3), (3,4), (4,1)
polynomial = 1.0*x[1]*x[2] + 1.0*x[2]*x[3] + 1.0*x[3]*x[4] + 1.0*x[4]*x[1]

# Solve
E, psi = TenSolver.minimize(polynomial; verbosity=0)

# Pick a deterministic representative from the two equivalent 2-colorings
coloring = [0, 1, 0, 1] in psi ? [0, 1, 0, 1] : [1, 0, 1, 0]
conflicts = polynomial(x => coloring)

println("Graph coloring: ", coloring)
println("Number of conflicts: ", conflicts)

# output

Graph coloring: [0, 1, 0, 1]
Number of conflicts: 0.0
```

## Constraints

TenSolver can also enforce hard constraints on the binary variables, so every
sampled solution is guaranteed feasible. This is a larger topic with its own
page — see [Constrained Optimization](@ref) for the projection method behind it,
the available constraint types, and worked examples.
