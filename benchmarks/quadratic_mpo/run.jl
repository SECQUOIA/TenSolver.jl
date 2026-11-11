using LinearAlgebra, Random, TenSolver

BLAS.set_num_threads(1)

# The same assignment problem in penalty and native-constraint formulations.
# Consecutive groups keep native projection widths bounded. The independently
# known optimum selects the first variable in every group, with total cost groups.
function assignment_problem(groups, width)
  n = groups * width
  costs = repeat(Float64.(1:width), groups)
  constraints = [
    SumConstraint(
      collect(((g - 1) * width + 1):(g * width)),
      ones(Int, width),
      1;
      relation = :(==),
    ) for g in 1:groups
  ]
  penalty = sum(costs) + 1
  Q = zeros(n, n)
  l = copy(costs)
  for g in 1:groups
    sites = ((g - 1) * width + 1):(g * width)
    Q[sites, sites] .+= penalty
    l[sites] .-= 2penalty
  end
  return Q, l, groups * penalty, costs, constraints
end

function run_case(groups, width, formulation, seed)
  Q, l, c, costs, constraints = assignment_problem(groups, width)
  if formulation == :native
    Q = zeros(length(costs), length(costs))
    l = costs
    c = 0.0
    active_constraints = constraints
  else
    active_constraints = AbstractConstraint[]
  end
  # Reproducibility is part of this benchmark: reset immediately before each solve.
  Random.seed!(seed)
  timed = @timed minimize(
    Q,
    l,
    c;
    domain = 0:1,
    constraints = active_constraints,
    verbosity = 0,
    iterations = 3,
    inidim = 8,
    maxdim = 16,
    cutoff = 1e-10,
    noise = [1e-5, 1e-6, 0.0],
    vtol = -Inf,
    check_variance_every_iteration = typemax(Int),
  )
  _, solution = timed.value
  Random.seed!(seed + 100000)
  draws = [TenSolver.sample(solution) for _ in 1:32]
  feasible = filter(x -> is_feasible(x, constraints), draws)
  best = isempty(feasible) ? Inf : minimum(x -> dot(costs, x), feasible)
  stats = solution.stats
  return (
    variables = length(costs),
    groups = groups,
    formulation = formulation,
    seed = seed,
    seconds = timed.time,
    allocated_bytes = timed.bytes,
    objective_bond = stats.max_bonds.objective,
    hamiltonian_bond = stats.max_bonds.hamiltonian,
    sweeps = length(stats.energies),
    feasible_draws = length(feasible),
    best_feasible_cost = best,
    exact_optimum = groups,
  )
end

# Warm both dispatch paths symmetrically before recording full public-call times.
warmup = run_case(2, 3, :penalty, 1)
run_case(2, 3, :native, 1)
println(join(keys(warmup), ','))
for groups in (4, 8, 16), formulation in (:penalty, :native), seed in (101, 202, 303)
  println(join(values(run_case(groups, 4, formulation, seed)), ','))
  flush(stdout)
end
