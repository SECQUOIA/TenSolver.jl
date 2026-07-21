# Constrained Optimization

Besides the unconstrained objective, TenSolver can enforce **hard constraints**
on the variables. Instead of adding penalty terms to the objective
(which only discourage infeasible solutions), each constraint is lowered to a
*projection MPO* that removes the infeasible subspace exactly, so every sampled
solution is guaranteed feasible.

The projection design is adapted from CoTenN (Sharma, Peng, Dangwal, and
Achour, *"CoTenN: Constrained Optimization with Tensor Networks,"* PLDI 2026).
TenSolver implements it as a native Julia lowering built on ITensors — no CoTenN
code is vendored.

!!! warning "Experimental"
    Hard constraints are experimental. They are TenSolver's native Julia
    lowering target; a future JuMP/MOI integration may change which constraint
    abstraction is considered stable public API.

## How projections work

A solve with `constraints` proceeds in three steps:

1. The objective is turned into an MPO Hamiltonian ``H``, exactly as in the
   unconstrained case.
2. Each constraint is lowered to a deterministic finite automaton (DFA) that
   accepts exactly the feasible vectors, and the automaton is threaded into
   an MPO ``P`` that is **diagonal with entries 1 on feasible basis states and
   0 on infeasible ones** — an exact projector, not a penalty.
3. DMRG minimizes the projected Hamiltonian ``P' H P``. Because numerical noise
   and truncation can leak amplitude back into the (zero-energy) infeasible
   kernel, the state is re-projected at every iteration, which is what makes
   every sampled solution feasible.

Each of the four built-in constraint types has a specialized automaton with a
compact bond dimension (see the table below).


The cost driver is the MPO bond dimension: the projected Hamiltonian satisfies
``\chi(P' H P) \le \chi(H) \cdot \prod_i \chi(P_i)^2``, so compact projections
keep constrained solves close to the unconstrained cost.

| Constraint | Enforces | Bond dimension of ``P`` |
|:-----------|:---------|:------------------------|
| [`SumConstraint`](@ref) | ``\sum_i w_i \, x_{s_i} \lessgtr b`` | ``b + 2`` (independent of the number of variables) |
| [`NotEqualsConstraint`](@ref) | ``x_S \ne v`` (one forbidden assignment) | 2 |
| [`ExactlyOneConstraint`](@ref) | ``\mathrm{count}_{i \in S}(x_i = k) = 1`` | 2 |
| [`RelationConstraint`](@ref) | ``x_i \lessgtr x_j`` | 2 |

## Penalty comparison benchmark

The repository includes a deterministic
[knapsack benchmark](https://github.com/SECQUOIA/TenSolver.jl/tree/main/benchmarks/knapsack)
that compares `SumConstraint` projection MPOs with a bounded-slack penalty QUBO
on the same instances. It reports feasibility and the original knapsack value,
alongside runtime, sweep, and tensor-network bond metrics, and sweeps several
penalty coefficients to expose their sensitivity. Heavy benchmark runs stay out
of the default test suite.

## Using constraints

Pass a vector of constraints to [`minimize`](@ref) or [`maximize`](@ref)
through the `constraints` keyword. Here we pick assets to maximize total value
under a budget that admits at most two of them:

```jldoctest budget
using TenSolver

# Choose among three assets to maximize value, but a budget admits at most two.
values = [3.0, 2.0, 4.0]
budget = SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=))

E, psi = TenSolver.maximize(zeros(3, 3), values; constraints = [budget], verbosity = 0)
x = TenSolver.sample(psi)

# The optimum keeps the two most valuable assets (1 and 3), within the budget.
(E ≈ 7.0, x, is_feasible(x, [budget]))

# output

(true, [1.0, 0.0, 1.0], true)
```

You can also check an assignment against constraints directly with
[`is_feasible`](@ref), without solving:

```jldoctest budget
# Selecting assets 2 and 3 respects the budget; selecting all three does not.
(is_feasible([0, 1, 1], [budget]), is_feasible([1, 1, 1], [budget]))

# output

(true, false)
```

## Constraint reference

### SumConstraint

`SumConstraint(sites, weights, rhs; relation)` enforces the weighted sum

```math
\sum_{i \in \texttt{sites}} \texttt{weights}[i] \cdot x[i]
\;\; \texttt{relation} \;\; \texttt{rhs},
```

where `relation` is one of `:(==)`, `:(!=)`, `:(<=)`, or `:(>=)`.
The `sites` are unique positive integers representing variable indices,
while `weights` and `rhs` must be **nonnegative integers**.
Furthermore, the solver currently only supports `SumConstraint`
when the variable domains are nonnegative integers.


Its automaton tracks a capped partial sum,
so the projection bond dimension is `rhs + 2` regardless of how many variables the sum touches.

### NotEqualsConstraint

`NotEqualsConstraint(sites, values)` forbids a single partial assignment: at
least one component of ``x[\texttt{sites}]`` must differ from `values`:
``\exists i \in \texttt{sites},\  x[i] \ne \texttt{values}[i].``

Its bond dimension is always 2.

```jldoctest noteq
using TenSolver

# [1, 1] is the unconstrained optimum, but that assignment is forbidden.
exclude = NotEqualsConstraint([1, 2], [1, 1])

E, psi = TenSolver.minimize(zeros(2, 2), [-2.0, -1.0]; constraints = [exclude], verbosity = 0)

(E ≈ -2.0, TenSolver.sample(psi))

# output

(true, [1.0, 0.0])
```

### ExactlyOneConstraint

`ExactlyOneConstraint(sites, value)` requires exactly one of the selected sites
to equal `value`:

```math
\#\{\, s \in \texttt{sites} : x_s = \texttt{value} \,\} = 1.
```

Its specialized automaton has bond dimension 2.

```jldoctest onehot
using TenSolver

# Pick exactly one of the three options; the second is the most valuable.
one_hot = ExactlyOneConstraint([1, 2, 3], 1)

E, psi = TenSolver.minimize(zeros(3, 3), [-1.0, -3.0, -2.0]; constraints = [one_hot], verbosity = 0)

(E ≈ -3.0, TenSolver.sample(psi))

# output

(true, [0.0, 1.0, 0.0])
```

### RelationConstraint

`RelationConstraint(left_site, relation, right_site)` enforces the pairwise
relation ``x_{\texttt{left}} \lessgtr x_{\texttt{right}}`` with the same four
relations as `SumConstraint`.  Bond dimension 2.

```jldoctest relation
using TenSolver

# Selecting item 1 requires also selecting item 2 (x1 <= x2).
implies = RelationConstraint(1, :(<=), 2)

E, psi = TenSolver.minimize(zeros(2, 2), [-2.0, 1.0]; constraints = [implies], verbosity = 0)

# Taking both (-2 + 1 = -1) beats the feasible alternatives [0,0] and [0,1].
(E ≈ -1.0, TenSolver.sample(psi))

# output

(true, [1.0, 1.0])
```

## Combining constraints

Passing several constraints applies their conjunction — the feasible set is the
intersection. All four types compose freely:

```jldoctest combined
using TenSolver

constraints = [
    SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=)),
    NotEqualsConstraint([1, 2], [1, 1]),
    ExactlyOneConstraint([2, 3], 1),
    RelationConstraint(1, :(>=), 3),
]

E, psi = TenSolver.minimize(zeros(3, 3), [-3.0, -2.0, -1.0]; constraints, verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ -4.0, x, is_feasible(x, constraints))

# output

(true, [1.0, 0.0, 1.0], true)
```

## Infeasible models

When the constraints admit no feasible assignment,
the solve does not throw,
but logs a warning and reports the outcome as a status:
[`minimize`](@ref) returns `+Inf` (the minimum over an empty feasible set) together with an infeasible solution,
and [`maximize`](@ref) returns `-Inf`.
Check with [`is_feasible`](@ref); sampling an infeasible solution throws.

```julia
impossible = [SumConstraint([1, 2], [1, 1], 3; relation = :(==))]
E, psi = TenSolver.minimize(zeros(2, 2); constraints = impossible)

E                 # +Inf
is_feasible(psi)  # false
```
