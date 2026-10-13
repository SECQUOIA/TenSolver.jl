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
   an MPO ``P`` that is *diagonal with entries 1 on feasible basis states and
   0 on infeasible ones*.
3. DMRG minimizes the projected Hamiltonian ``P' H P``,
   or some semantically equivalent context-dependent MPO.


Each of the four built-in constraint types has a specialized automaton with a
compact bond dimension (see the table below).
For TenSolver's current implementation,
the effective Hamiltonian has bond dimension bounded by
``\chi(P' H P) \le \chi(H) \cdot \prod_i \chi(P_i)``.

| Constraint | Enforces | Bond dimension of ``P`` |
|:-----------|:---------|:------------------------|
| [`SumConstraint`](@ref) | ``\sum_i w_i \, x_{s_i} \lessgtr b`` | ``b + 2`` (independent of the number of variables) |
| [`SumModConstraint`](@ref) | ``\sum_i w_i \, x_{s_i} \equiv b \pmod m`` | ``m`` |
| [`NotEqualsConstraint`](@ref) | ``x_S \ne v`` (one forbidden assignment) | 2 |
| [`AssignmentConstraint`](@ref) | ``\mathrm{count}_{i \in S}(x_i \in G) \lessgtr b`` | ``b + 2`` |
| [`RelationConstraint`](@ref) | ``x_i \lessgtr x_j`` | At most ``\lvert U_{\min(i,j)}\rvert`` |

## Using constraints

Pass a vector of constraints to [`minimize`](@ref) or [`maximize`](@ref)
through the `constraints` keyword. Here we pick assets to maximize total value
under a budget that admits at most two of them:

```jldoctest budget
using TenSolver

# Choose among three assets to maximize value, but a budget admits at most two.
values = [3.0, 2.0, 4.0]
budget = SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=))

E, psi = TenSolver.maximize(zeros(3, 3), values; domain = 0:1, constraints = [budget], verbosity = 0)
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

## Domains with Constraints

The native domain syntax from [Choosing Native Domains](@ref) also applies to
constrained solves. Domains select the allowed values; constraints test those
values at their variable indices, including with per-variable domains.

| Constraint | Domain requirement for participating variables |
|:-----------|:-----------------------------------------------|
| [`SumConstraint`](@ref) | Nonnegative integer-valued entries |
| [`SumModConstraint`](@ref) | Integer-valued entries, including negative values |
| [`NotEqualsConstraint`](@ref) | Real values representable in the element type of `values` |
| [`AssignmentConstraint`](@ref) | Real values, including fractional values |
| [`RelationConstraint`](@ref) | Real values, including fractional values |

For the sum constraints, the requirement applies to sites with nonzero weights;
other variables may use different real-valued domains. Integer-valued floating
values such as `2.0` satisfy the integer requirement, while `0.5` does not.
`SumConstraint` also requires nonnegative integer weights and rhs;
`SumModConstraint` allows signed integer weights and rhs and requires a positive
integer modulus. Equality and membership comparisons use exact values.

For `NotEqualsConstraint`, choose a `values` element type that represents every
participating domain value. With a fractional domain, use floating-point values,
such as `NotEqualsConstraint([1], [1.0])`; an integer `values` vector can cause an
`InexactError` when a fractional domain value is converted during lowering.

Constraint values do not enlarge domains. For example, counting occurrences of
`2` at a site with domain `0:1` always contributes zero; a requirement that it
contribute one is infeasible. Check [`is_feasible`](@ref) on the solution before
sampling when feasibility is uncertain.

### Nonbinary budget example

Reuse the asset values from [Using constraints](@ref), allowing zero, one or two
units of each asset and at most three units in total. Every unit of asset 3 is
worth `4`, compared with `3` and `2` for the others. Its limit of two units means
the best allocation takes two units of asset 3 and one of asset 1:
`4 × 2 + 3 × 1 = 11`. Any replacement lowers the value, so `[1, 0, 2]` is the
unique optimum.

```jldoctest nonbinary-budget
using TenSolver

values = [3.0, 2.0, 4.0]
domains = [0:2, 0:2, 0:2]
budget = SumConstraint([1, 2, 3], [1, 1, 1], 3; relation = :(<=))

E, psi = TenSolver.maximize(values;
    domain = domains, constraints = [budget], verbosity = 0)
x = TenSolver.sample(psi)

(E ≈ 11.0, x, values' * x ≈ E,
 all(x[i] in domains[i] for i in eachindex(x)), is_feasible(x, [budget]))

# output

(true, [1.0, 0.0, 2.0], true, true, true)
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
when domains at sites with nonzero weights contain nonnegative integer-valued entries.


Its automaton tracks a capped partial sum,
so the projection bond dimension is `rhs + 2` regardless of how many variables the sum touches.

### Modular Sum Constraint

`SumModConstraint(sites, weights, rhs; mod = m)` enforces the weighted sum

```math
\sum_{i \in \texttt{sites}} \texttt{weights}[i] \cdot x[i]
\equiv \texttt{rhs} \pmod m,
```

The `sites` are unique positive integers representing variable indices,
while `weights` and `rhs` must be **integers** and may be negative.
The modulus `m` must be a positive integer.
Weights and the rhs are normalized to their least nonnegative residues modulo `m`,
and domains at sites with nonzero weights may contain any integer-valued entries.
Other variables may have fractional domains, as described in [Domains with Constraints](@ref).

Its MPO tracks a partial sum with bond dimension `m`.

### NotEqualsConstraint

`NotEqualsConstraint(sites, values)` forbids a single partial assignment: at
least one component of ``x[\texttt{sites}]`` must differ from `values`:
``\exists i \in \texttt{sites},\  x[i] \ne \texttt{values}[i].``

Its bond dimension is always 2.

```jldoctest noteq
using TenSolver

# [1, 1] is the unconstrained optimum, but that assignment is forbidden.
exclude = NotEqualsConstraint([1, 2], [1, 1])

E, psi = TenSolver.minimize(zeros(2, 2), [-2.0, -1.0]; domain = 0:1, constraints = [exclude], verbosity = 0)

(E ≈ -2.0, TenSolver.sample(psi))

# output

(true, [1.0, 0.0])
```

### AssignmentConstraint

`AssignmentConstraint(sites, values, rel, rhs)`
restricts how many selected sites take a value in `values`:

```math
\#\{\, s \in \texttt{sites} : x_s \in \texttt{values} \,\} \;\; \texttt{relation} \;\; \texttt{rhs}.
```

For rhs `k`, its automaton has maximum bond dimension `k+2`.

```jldoctest onehot
using TenSolver

# Pick exactly one of the three options; the second is the most valuable.
one_hot = AssignmentConstraint([1, 2, 3], [1], :(==), 1)

E, psi = TenSolver.minimize(zeros(3, 3), [-1.0, -3.0, -2.0]; domain = 0:1, constraints = [one_hot], verbosity = 0)

(E ≈ -3.0, TenSolver.sample(psi))

# output

(true, [0.0, 1.0, 0.0])
```

### RelationConstraint

`RelationConstraint(left_site, relation, right_site)` enforces the pairwise
relation ``x_{\texttt{left}} \lessgtr x_{\texttt{right}}`` with the same four
relations as `SumConstraint`. Its projection bond dimension is at most the size
of the earlier-indexed variable's domain (2 for binary domains).

```jldoctest relation
using TenSolver

# Selecting item 1 requires also selecting item 2 (x1 <= x2).
implies = RelationConstraint(1, :(<=), 2)

E, psi = TenSolver.minimize(zeros(2, 2), [-2.0, 1.0]; domain = 0:1, constraints = [implies], verbosity = 0)

# Taking both (-2 + 1 = -1) beats the feasible alternatives [0,0] and [0,1].
(E ≈ -1.0, TenSolver.sample(psi))

# output

(true, [1.0, 1.0])
```

## Combining constraints

Passing several constraints applies their conjunction.
All four types compose freely:

```jldoctest combined
using TenSolver

constraints = [
    SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(<=)),
    NotEqualsConstraint([1, 2], [1, 1]),
    AssignmentConstraint([2, 3], [1], :(==), 1),
    RelationConstraint(1, :(>=), 3),
]

E, psi = TenSolver.minimize(zeros(3, 3), [-3.0, -2.0, -1.0]; domain = 0:1, constraints, verbosity = 0)
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
E, psi = TenSolver.minimize(zeros(2, 2); domain = 0:1, constraints = impossible)

E                 # +Inf
is_feasible(psi)  # false
```
