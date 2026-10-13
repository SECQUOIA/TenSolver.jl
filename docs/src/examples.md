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
E, psi = TenSolver.minimize(Q; domain = 0:1, verbosity=0)

# Sample a solution
x = TenSolver.sample(psi)

# Verify the solution achieves the minimum energy and inspect the chosen bundle
(E ≈ -3.0, join(assets[findall(==(1), x)], ", "))

# output

(true, "Wind, Battery")
```

## Choosing Native Domains

For native DMRG solves, pass `domain` explicitly to [`minimize`](@ref) or
[`maximize`](@ref). A single nonempty vector or range of real values is a
**uniform domain**, shared by every variable. A vector of nonempty vectors or
ranges gives **one domain per variable**; its length must equal the number of
variables.

| Variable values | Uniform `domain` | Worked example |
|:----------------|:-----------------|:---------------|
| Binary | `0:1` | [Basic QUBO Problem](@ref) |
| Ising spins | `[-1, 1]` | [Ising Model](@ref) |
| Multi-valued integers | `0:2` or `[-2, 0, 3]` | [Non-consecutive Integer Domains](@ref) |
| Fractional values | `[0.0, 0.5, 1.0]` | [Fractional Domains](@ref) |

These domains select discrete values, including gaps; `[0.0, 0.5, 1.0]` is a
three-value choice. Each domain is converted to the objective's floating-point
type, sorted and deduplicated before solving. [`sample`](@ref) returns the
selected values, rather than their positions in the sorted domain; integer
choices can therefore appear as floating-point values in a sample.

For a matrix/linear objective `x'Q*x + l'x + c`, domain `i` and sample entry `i`
refer to row/column `i` of `Q` and entry `i` of `l`. Matrix preprocessing returns
samples in that same input order. Polynomial inputs use the input's
`effective_variables` order, as explained in [Polynomial Variable Order](@ref).
Constraint site indices follow the same variable order. For constraint-specific
requirements, see [Domains with Constraints](@ref).

### Per-variable Domains

Pass one domain for each variable when their allowed values differ. This
example minimizes `2x₁ - 3x₂ + 4`: the first coefficient selects `x₁ = -1`,
the second selects `x₂ = 1.5`, giving `-2 - 4.5 + 4 = -2.5`:

```jldoctest heterogeneous-domain
using TenSolver

domains = [[-1, 2], [0.0, 0.5, 1.5]]
E, psi = TenSolver.minimize([2.0, -3.0], 4.0;
    domain = domains, verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ -2.5, x, 2x[1] - 3x[2] + 4 ≈ E,
 all(x[i] in domains[i] for i in eachindex(x)))

# output

(true, [-1.0, 1.5], true, true)
```

### Ising Model

Use `domain = [-1, 1]` when the variables are Ising spins:

```jldoctest ising-domain
using TenSolver

J = [0.0 -1.0;
     -1.0 0.0]
h = [0.25, -0.5]

E, psi = TenSolver.minimize(J, h; domain = [-1, 1], verbosity = 0)
s = TenSolver.sample(psi)

(E ≈ -2.25, s, s' * J * s + h' * s ≈ E, all(in((-1, 1)), s))

# output

(true, [1.0, 1.0], true, true)
```

### Non-consecutive Integer Domains

Use a finite set with more than two integer values. Positive coefficients select
`-2`, while the negative coefficient selects `3`, giving `-2 - 12 - 4 = -18`:

```jldoctest sparse-domain
using TenSolver

l = [1.0, -4.0, 2.0]

E, psi = TenSolver.minimize(l; domain = [-2, 0, 3], verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ -18.0, x, l' * x ≈ E, all(in((-2, 0, 3)), x))

# output

(true, [-2.0, 3.0, -2.0], true, true)
```

### Fractional Domains

Unconstrained DMRG optimization accepts any finite domain of real values,
including fractional values. Here `-2x₁ + 3x₂` is minimized by `x₁ = 1`
and `x₂ = 0`:

```jldoctest fractional-domain
using TenSolver

l = [-2.0, 3.0]

E, psi = TenSolver.minimize(l; domain = [0.0, 0.5, 1.0], verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ -2.0, x, l' * x ≈ E, all(in((0.0, 0.5, 1.0)), x))

# output

(true, [1.0, 0.0], true, true)
```

Hard constraints can impose narrower domain requirements; see
[Domains with Constraints](@ref).

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
E, psi = TenSolver.minimize(Q, l, c; domain = 0:1, verbosity=0)
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
    domain = 0:1,
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

The returned `Solution` always carries lightweight per-iteration statistics in the `stats` field.
See [`TenSolver.SolverStatistics`](@ref) for more details.

`psi.stats.energies`, `psi.stats.bond_dims`, and `psi.stats.elapsed_times` contain
one value per completed iteration. `psi.stats.variances` has the same length and
contains either the checked variance or `nothing` when that iteration did not
perform the configured variance check.

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

E, psi = TenSolver.minimize(Q; domain = 0:1, iterations=50, on_iteration=cb, callback_every=5)

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

E, psi = TenSolver.minimize(Q; domain = 0:1, iterations=50, on_iteration=cb, callback_every=5)
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
E, psi = TenSolver.minimize(Q; domain = 0:1, device = CUDA.cu)

# Sample solutions
x = TenSolver.sample(psi)
```

For Apple Silicon GPUs:

```julia
using TenSolver
import Metal

Q = randn(100, 100)

# Solve on GPU using Metal
E, psi = TenSolver.minimize(Q; domain = 0:1, device = Metal.mtl)
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
E, psi = TenSolver.maximize(Q; domain = 0:1, verbosity=0)
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
E, psi = TenSolver.minimize(Q, l, 0.0; domain = 0:1, verbosity=0)

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
E, psi = TenSolver.minimize(polynomial; domain = 0:1, verbosity=0)

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

This interface converts polynomial expressions into the tensor network representation
used by the solver. Use `domain` to select the finite values allowed for each variable.

### Polynomial Variable Order

For polynomial inputs, per-variable domains, constraint indices, and sampled
vectors use the input polynomial's `effective_variables` order. Native DMRG
retains those positions when domain reduction makes a variable independent of
the objective.

For example, `p = x[1]^2 - x[1] - 2x[2]` with binary variables has minimum
`-2`. The first two terms vanish on the binary domain, but every returned sample
still has two entries: either `[0, 1]` or `[1, 1]`. A constraint on index `1`
continues to refer to `x[1]`. With domains `[0:1, [2, 4]]` in that same order,
the minimum is `-8` and the second sample entry is `4`.

Read the order from the **input** polynomial before solving, and pair the sample
with those variables when evaluating it. Effective variables are those occurring
with nonzero degree in the input, so a declared variable absent from the input
has no sample entry. Declaration order alone should not be used to choose the
per-variable domain order.

```jldoctest polynomial-domain-order
using TenSolver
using DynamicPolynomials

@polyvar z[1:2]
p = 1.0z[1]^2 - z[1] - 2z[2]
vars = effective_variables(p)
# Assign binary values to z[1] and {2, 4} to z[2] in the input's order.
domains = [v == z[1] ? collect(0:1) : [2, 4] for v in vars]

E, psi = TenSolver.minimize(p; domain = domains, verbosity = 0)
values = TenSolver.sample(psi)

# z[1]^2 - z[1] vanishes on its domain but its sample position is retained.
(E ≈ -8.0, length(values) == length(vars) == 2,
 all(values[i] in domains[i] for i in eachindex(vars)), p(vars => values) ≈ E)

# output

(true, true, true, true)
```

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
E, psi = TenSolver.minimize(polynomial; domain = 0:1, verbosity=0)

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

TenSolver can also enforce hard constraints on the variables, so every
sampled solution is guaranteed feasible. This is a larger topic with its own
page — see [Constrained Optimization](@ref) for the projection method behind it,
the available constraint types, and worked examples.
