module KnapsackBenchmark

using LinearAlgebra: dot
using Random: Random, rand

import BenchmarkTools
import ITensorMPS
import ITensors
import TenSolver

export benchmark_rows, projection_scaling_rows, write_csv

const SOLVER_SEED = 66

"""
A deterministic binary knapsack instance used by the benchmark.
"""
struct KnapsackInstance
  name::String
  weights::Vector{Int}
  values::Vector{Float64}
  capacity::Int

  function KnapsackInstance(name, weights, values, capacity)
    if !(length(weights) == length(values))
      throw(DimensionMismatch("weights and values must have the same length"))
    end
    if isempty(weights)
      throw(ArgumentError("a knapsack instance must contain at least one item"))
    end
    if !(all(>(0), weights))
      throw(ArgumentError("item weights must be positive integers"))
    end
    if !(all(>(0), values))
      throw(ArgumentError("item values must be positive"))
    end
    if !(capacity >= 0)
      throw(ArgumentError("capacity must be nonnegative"))
    end

    return new(String(name), Int.(weights), Float64.(values), Int(capacity))
  end
end

"""
The small hand-checkable instance used to validate the benchmark output.
"""
reference_instance() = KnapsackInstance("reference_4", [4, 3, 2, 3], [8, 4, 5, 3], 6)

"""
Build one of the standard 0-1 knapsack instance classes from Martello,
Pisinger, and Toth's generator: uncorrelated, weakly correlated, strongly
correlated, or subset-sum. The half-total-weight capacity is one slice of the
generator's varying-capacity series.
"""
function pisinger_instance(kind, n; coefficient_range = 10, seed)
  if !(n > 0)
    throw(ArgumentError("instance size must be positive"))
  end
  if !(coefficient_range >= 10)
    throw(ArgumentError("coefficient range must be at least 10"))
  end

  Random.seed!(seed)
  weights = rand(1:coefficient_range, n)
  correlation_range = div(coefficient_range, 10)
  values = if kind == :uncorrelated
    rand(1:coefficient_range, n)
  elseif kind == :weakly_correlated
    max.(1, weights .+ rand((-correlation_range):correlation_range, n))
  elseif kind == :strongly_correlated
    weights .+ correlation_range
  elseif kind == :subset_sum
    copy(weights)
  else
    throw(ArgumentError("unsupported Pisinger instance class: $(repr(kind))"))
  end

  capacity = max(maximum(weights), div(sum(weights), 2))
  return KnapsackInstance("pisinger_$(kind)_n$(n)", weights, values, capacity)
end

function default_instances()
  specifications = (
    (kind = :uncorrelated, n = 8, seed = 6601),
    (kind = :weakly_correlated, n = 12, seed = 6602),
    (kind = :strongly_correlated, n = 16, seed = 6603),
    (kind = :subset_sum, n = 16, seed = 6604),
  )
  instances = map(specifications) do spec
    return pisinger_instance(spec.kind, spec.n; seed = spec.seed)
  end
  return [reference_instance(), instances...]
end

item_weight(instance::KnapsackInstance, items) = dot(instance.weights, items)
item_value(instance::KnapsackInstance, items) = dot(instance.values, items)
function is_capacity_feasible(instance::KnapsackInstance, items)
  return item_weight(instance, items) <= instance.capacity
end

"""
Find the exact constrained optimum by enumerating all item selections.
"""
function brute_force_optimum(instance::KnapsackInstance)
  best_value = -Inf
  best_weight = typemax(Int)
  best_items = Int[]

  for assignment in Iterators.product(fill(0:1, length(instance.weights))...)
    items = collect(assignment)
    weight = item_weight(instance, items)
    value = item_value(instance, items)

    if weight <= instance.capacity &&
       (value > best_value || (value == best_value && weight < best_weight))
      best_value = value
      best_weight = weight
      best_items = items
    end
  end

  return (value = best_value, weight = best_weight, items = best_items)
end

"""
    slack_weights(capacity)

Return a bounded-binary encoding whose subset sums cover every integer from
zero through `capacity`, without representing larger slack values.
"""
function slack_weights(capacity::Integer)
  if !(capacity >= 0)
    throw(ArgumentError("capacity must be nonnegative"))
  end
  encoded = Int[]
  remaining = Int(capacity)
  power = 1

  while remaining > 0
    weight = min(power, remaining)
    push!(encoded, weight)
    remaining -= weight
    power *= 2
  end

  return encoded
end

"""
    penalty_qubo(instance, penalty)

Encode `weight(items) <= capacity` with bounded-binary slack variables and the
objective

`-value(items) + penalty * (weight(items) + slack - capacity)^2`.

The returned `Q`, `l`, and `constant` follow TenSolver's `x'Qx + l'x + c`
convention. The first `nitems` variables are item decisions and the rest encode
slack.
"""
function penalty_qubo(instance::KnapsackInstance, penalty::Real)
  if !(penalty > 0)
    throw(ArgumentError("penalty must be positive"))
  end
  slack = slack_weights(instance.capacity)
  coefficients = Float64.([instance.weights; slack])
  values = [instance.values; zeros(length(slack))]
  lambda = Float64(penalty)

  Q = lambda .* (coefficients * coefficients')
  l = -values .- (2lambda * instance.capacity) .* coefficients
  constant = lambda * instance.capacity^2

  return (Q, l, constant, nitems = length(instance.weights), slack_weights = slack)
end

function penalty_value(model, assignment)
  return dot(assignment, model.Q, assignment) + dot(model.l, assignment) + model.constant
end

function projection_resource_metrics(instance::KnapsackInstance; cutoff = 1e-10)
  nitems = length(instance.weights)
  Q = zeros(nitems, nitems)
  l = -instance.values
  constraint = TenSolver.SumConstraint(
    collect(1:nitems),
    instance.weights,
    instance.capacity;
    relation = :(<=),
  )
  H = TenSolver.tensorize(Q, l; cutoff, domain = 0:1)
  sites = ITensorMPS.siteinds(first, H; plev = 0)
  P = TenSolver.projection_mpo(constraint, sites; domain = 0:1)
  H_effective = TenSolver.project_hamiltonian(H, P; cutoff)

  return (
    objective_mpo_bond = ITensorMPS.maxlinkdim(H),
    projection_mpo_bond = ITensorMPS.maxlinkdim(P),
    effective_hamiltonian_bond = ITensorMPS.maxlinkdim(H_effective),
  )
end

function projection_scaling_instances()
  probes = NamedTuple[]

  for capacity in (1, 2, 4, 8)
    nitems = 16
    instance = KnapsackInstance(
      "capacity_$(capacity)",
      ones(Int, nitems),
      ones(Int, nitems),
      capacity,
    )
    push!(probes, (sweep = "capacity", instance))
  end

  for nitems in (8, 16, 32)
    instance = KnapsackInstance("items_$(nitems)", ones(Int, nitems), ones(Int, nitems), 3)
    push!(probes, (sweep = "item_count", instance))
  end

  for scale in (4, 8, 32, 128)
    weights = vcat([1, 2, 3], fill(scale, 5))
    instance =
      KnapsackInstance("weight_scale_$(scale)", weights, ones(Int, length(weights)), 3)
    push!(probes, (sweep = "weight_magnitude", instance))
  end

  return probes
end

"""
Build the controlled projection-MPO bond-dimension scaling table.
"""
function projection_scaling_rows(; cutoff = 1e-10)
  return map(projection_scaling_instances()) do probe
    instance = probe.instance
    resources = projection_resource_metrics(instance; cutoff)
    return (
      sweep = probe.sweep,
      instance = instance.name,
      nitems = length(instance.weights),
      capacity = instance.capacity,
      max_weight = maximum(instance.weights),
      capacity_state_bound = instance.capacity + 2,
      objective_mpo_bond = resources.objective_mpo_bond,
      projection_mpo_bond = resources.projection_mpo_bond,
      effective_hamiltonian_bond = resources.effective_hamiltonian_bond,
    )
  end
end

function penalty_resource_metrics(model; cutoff = 1e-10)
  H = TenSolver.tensorize(model.Q, model.l; cutoff, domain = 0:1)
  bond = ITensorMPS.maxlinkdim(H)
  return (
    objective_mpo_bond = bond,
    projection_mpo_bond = missing,
    effective_hamiltonian_bond = bond,
  )
end

function qubo_hamiltonian(Q, l, sites; cutoff)
  os = ITensorMPS.OpSum{Float64}()
  nvariables = length(sites)

  for i in 1:nvariables
    linear_coefficient = Q[i, i] + l[i]
    if abs(linear_coefficient) > cutoff
      os += (linear_coefficient, "D", (domain = 0:1,), i)
    end

    for j in (i + 1):nvariables
      quadratic_coefficient = Q[i, j] + Q[j, i]
      if abs(quadratic_coefficient) > cutoff
        os += (quadratic_coefficient, "D", (domain = 0:1,), i, "D", (domain = 0:1,), j)
      end
    end
  end

  return isempty(os) ? ITensorMPS.MPO(Float64, sites) : ITensorMPS.MPO(Float64, os, sites)
end

function projection_hamiltonian(instance, sites; cutoff)
  nitems = length(instance.weights)
  H = qubo_hamiltonian(zeros(nitems, nitems), -instance.values, sites; cutoff)
  constraint = TenSolver.SumConstraint(
    collect(1:nitems),
    instance.weights,
    instance.capacity;
    relation = :(<=),
  )
  P = TenSolver.projection_mpo(constraint, sites; domain = 0:1)
  H_effective = TenSolver.project_hamiltonian(H, P; cutoff)
  for i in eachindex(H_effective)
    source = (
      only(ITensorMPS.siteinds(H_effective, i; plev = 0)),
      only(ITensorMPS.siteinds(H_effective, i; plev = 1)),
    )
    target = (sites[i], sites[i]')
    H_effective[i] = ITensors.replaceinds(H_effective[i], source, target)
  end
  return H_effective
end

function state_variance(H, mps)
  expectation = real(ITensors.inner(mps', H, mps))
  second_moment = real(ITensors.inner(H, mps, H, mps))
  return max(0.0, second_moment - expectation^2)
end

function variance_observer(hamiltonian_builder)
  hamiltonian = Ref{Any}()
  setup = Ref{Any}()
  variances = Float64[]
  callback = function (mps; kw...)
    if !isassigned(hamiltonian)
      timed = @timed hamiltonian_builder(ITensorMPS.siteinds(mps))
      hamiltonian[] = timed.value
      setup[] = (time = timed.time, gctime = timed.gctime, memory = timed.bytes)
    end
    push!(variances, state_variance(hamiltonian[], mps))
    return nothing
  end
  return variances, setup, callback
end

function benchmark_repeated(f; samples)
  outputs = Any[]
  capture() = begin
    output = f()
    push!(outputs, output)
    return output
  end
  # Leave the shorter per-solve limit to TenSolver while ensuring BenchmarkTools
  # collects every requested fixed-count sample.
  benchmark = BenchmarkTools.@benchmarkable $capture() samples=samples evals=1 seconds=3600
  trial = BenchmarkTools.run(benchmark; warmup = false)
  estimate = BenchmarkTools.median(trial)
  measured_outputs = last(outputs, min(samples, length(outputs)))
  length(measured_outputs) == length(trial.times) || error(
    "could not associate every benchmark timing with its measured output",
  )
  median_index = sortperm(trial.times)[cld(length(trial.times), 2)]
  return (
    value = measured_outputs[median_index],
    outputs = measured_outputs,
    time = estimate.time / 1e9,
    gctime = estimate.gctime / 1e9,
    memory = estimate.memory,
    allocs = estimate.allocs,
    samples = length(measured_outputs),
  )
end

function sample_median(values)
  if isempty(values)
    return 0.0
  end
  ordered = sort!(collect(values))
  return ordered[cld(length(ordered), 2)]
end

function component_median(outputs, component, field)
  return sample_median(getproperty(getproperty(output, component), field) for
                       output in outputs)
end

function assert_stable_items(outputs)
  items = getproperty.(outputs, :items)
  if !(all(isequal(first(items)), items))
    error("fixed-seed timing samples returned different selected assignments",)
  end
  return first(items)
end

function runtime_metadata()
  return (
    julia_version = string(VERSION),
    julia_threads = Threads.nthreads(),
    system = string(Sys.KERNEL),
    architecture = string(Sys.ARCH),
    tensolver_version = string(Base.pkgversion(TenSolver)),
    itensors_version = string(Base.pkgversion(ITensors)),
    itensormps_version = string(Base.pkgversion(ITensorMPS)),
    benchmarktools_version = string(Base.pkgversion(BenchmarkTools)),
  )
end

function solution_stats(solution)
  return (
    sweeps = length(solution.energies),
    solution_max_bond = isempty(solution.bond_dims) ? 0 : maximum(solution.bond_dims),
    solver_elapsed_seconds = isempty(solution.elapsed_times) ? 0.0 :
                             last(solution.elapsed_times),
  )
end

item_bits(sample, nitems) = round.(Int, sample[1:nitems])

function best_projection_sample(instance, samples)
  best = item_bits(first(samples), length(instance.weights))
  best_key = (
    !is_capacity_feasible(instance, best),
    -item_value(instance, best),
    item_weight(instance, best),
  )

  for sample in Iterators.drop(samples, 1)
    items = item_bits(sample, length(instance.weights))
    key = (
      !is_capacity_feasible(instance, items),
      -item_value(instance, items),
      item_weight(instance, items),
    )
    if key < best_key
      best = items
      best_key = key
    end
  end

  return best
end

function best_penalty_sample(instance, model, samples)
  best = round.(Int, first(samples))
  best_items = item_bits(best, model.nitems)
  best_key = (
    penalty_value(model, best),
    -item_value(instance, best_items),
    item_weight(instance, best_items),
  )

  for sample in Iterators.drop(samples, 1)
    assignment = round.(Int, sample)
    items = item_bits(assignment, model.nitems)
    key = (
      penalty_value(model, assignment),
      -item_value(instance, items),
      item_weight(instance, items),
    )
    if key < best_key
      best = assignment
      best_key = key
    end
  end

  return best
end

function solver_options(iterations, cutoff, time_limit, on_iteration)
  return (
    iterations = iterations,
    time_limit = time_limit,
    cutoff = cutoff,
    inidim = 8,
    maxdim = [10, 20, 40, 80, 120, 200],
    noise = [1e-6, 1e-8, 0.0],
    check_variance_every_iteration = iterations + 1,
    on_iteration,
    callback_every = 1,
    verbosity = 0,
  )
end

function benchmark_result_row(
  instance,
  exact,
  method,
  items,
  reported_objective,
  formulation_timed,
  case_timed,
  solution,
  resources;
  variances,
  nvariables,
  iterations,
  reads,
  cutoff,
  time_limit,
  timing_samples,
  benchmark_id,
  penalty_factor = missing,
  penalty = missing,
  penalized_objective = missing,
)
  feasible = is_capacity_feasible(instance, items)
  value = item_value(instance, items)
  stats = solution_stats(solution)
  observer_setup_seconds = component_median(case_timed.outputs, :observer_setup, :time)
  observer_setup_gc_seconds = component_median(case_timed.outputs, :observer_setup, :gctime)
  observer_setup_allocated_bytes =
    component_median(case_timed.outputs, :observer_setup, :memory)
  sampling_seconds = component_median(case_timed.outputs, :sampling, :time)
  sampling_gc_seconds = component_median(case_timed.outputs, :sampling, :gctime)
  sampling_allocated_bytes = component_median(case_timed.outputs, :sampling, :memory)
  solver_call_seconds = max(0.0, case_timed.time - sampling_seconds)
  solver_excluding_observer_setup_seconds =
    max(0.0, solver_call_seconds - observer_setup_seconds)
  metadata = runtime_metadata()

  return (;
    benchmark_id,
    metadata...,
    instance = instance.name,
    method,
    nitems = length(instance.weights),
    nvariables,
    capacity = instance.capacity,
    penalty_factor,
    penalty,
    exact_value = exact.value,
    original_value = value,
    feasible,
    optimality_gap = feasible ? exact.value - value : missing,
    penalized_objective,
    solver_reported_objective = reported_objective,
    solver_seed = SOLVER_SEED,
    requested_iterations = iterations,
    reads,
    cutoff,
    time_limit_seconds = time_limit,
    timing_samples,
    formulation_wall_seconds = formulation_timed.time,
    formulation_gc_seconds = formulation_timed.gctime,
    formulation_allocated_bytes = formulation_timed.memory,
    formulation_allocations = formulation_timed.allocs,
    solve_wall_seconds = case_timed.time,
    solve_gc_seconds = case_timed.gctime,
    solve_allocated_bytes = case_timed.memory,
    solve_allocations = case_timed.allocs,
    observer_setup_seconds,
    observer_setup_gc_seconds,
    observer_setup_allocated_bytes,
    sampling_seconds,
    sampling_gc_seconds,
    sampling_allocated_bytes,
    solver_excluding_observer_setup_seconds,
    end_to_end_wall_seconds = formulation_timed.time + case_timed.time,
    solver_elapsed_seconds = stats.solver_elapsed_seconds,
    sweeps = stats.sweeps,
    time_limit_reached = (stats.sweeps < iterations ||
                          stats.solver_elapsed_seconds > time_limit),
    solution_max_bond = stats.solution_max_bond,
    final_variance = isempty(variances) ? missing : last(variances),
    truncation_error = missing,
    objective_mpo_bond = resources.objective_mpo_bond,
    projection_mpo_bond = resources.projection_mpo_bond,
    effective_hamiltonian_bond = resources.effective_hamiltonian_bond,
  )
end

function projection_row(
  instance::KnapsackInstance,
  exact;
  iterations,
  reads,
  cutoff,
  time_limit,
  timing_samples,
  benchmark_id,
)
  nitems = length(instance.weights)
  formulation_timed = benchmark_repeated(; samples = timing_samples) do
    return TenSolver.SumConstraint(
      collect(1:nitems),
      instance.weights,
      instance.capacity;
      relation = :(<=),
    )
  end
  constraint = formulation_timed.value
  case_timed = benchmark_repeated(; samples = timing_samples) do
    variances, observer_setup, callback =
      variance_observer(sites -> projection_hamiltonian(instance, sites; cutoff),)
    Random.seed!(SOLVER_SEED)
    options = solver_options(iterations, cutoff, time_limit, callback)
    reported_objective, solution =
      TenSolver.maximize(instance.values; constraints = [constraint], options...)
    sampling = @timed best_projection_sample(instance, TenSolver.sample(solution, reads))
    if !(isassigned(observer_setup))
      error("projection variance observer did not run")
    end
    return (;
      reported_objective,
      solution,
      items = sampling.value,
      variances,
      observer_setup = observer_setup[],
      sampling = (time = sampling.time, gctime = sampling.gctime, memory = sampling.bytes),
    )
  end
  items = assert_stable_items(case_timed.outputs)
  output = case_timed.value
  resources = projection_resource_metrics(instance; cutoff)
  return benchmark_result_row(
    instance,
    exact,
    "projection",
    items,
    output.reported_objective,
    formulation_timed,
    case_timed,
    output.solution,
    resources;
    variances = output.variances,
    nvariables = nitems,
    iterations,
    reads,
    cutoff,
    time_limit,
    timing_samples = case_timed.samples,
    benchmark_id,
  )
end

function penalty_row(
  instance::KnapsackInstance,
  exact,
  penalty_factor;
  iterations,
  reads,
  cutoff,
  time_limit,
  timing_samples,
  benchmark_id,
)
  penalty = Float64(penalty_factor) * sum(instance.values)
  formulation_timed = benchmark_repeated(; samples = timing_samples) do
    return penalty_qubo(instance, penalty)
  end
  model = formulation_timed.value
  case_timed = benchmark_repeated(; samples = timing_samples) do
    variances, observer_setup, callback =
      variance_observer(sites -> qubo_hamiltonian(model.Q, model.l, sites; cutoff),)
    Random.seed!(SOLVER_SEED)
    options = solver_options(iterations, cutoff, time_limit, callback)
    reported_objective, solution =
      TenSolver.minimize(model.Q, model.l, model.constant; options...)
    sampling = @timed begin
      assignment = best_penalty_sample(instance, model, TenSolver.sample(solution, reads))
      (assignment = assignment, items = item_bits(assignment, model.nitems))
    end
    if !(isassigned(observer_setup))
      error("penalty variance observer did not run")
    end
    return (;
      reported_objective,
      solution,
      assignment = sampling.value.assignment,
      items = sampling.value.items,
      variances,
      observer_setup = observer_setup[],
      sampling = (time = sampling.time, gctime = sampling.gctime, memory = sampling.bytes),
    )
  end
  items = assert_stable_items(case_timed.outputs)
  output = case_timed.value
  resources = penalty_resource_metrics(model; cutoff)
  return benchmark_result_row(
    instance,
    exact,
    "penalty_qubo",
    items,
    output.reported_objective,
    formulation_timed,
    case_timed,
    output.solution,
    resources;
    variances = output.variances,
    nvariables = length(output.assignment),
    iterations,
    reads,
    cutoff,
    time_limit,
    timing_samples = case_timed.samples,
    benchmark_id,
    penalty_factor = Float64(penalty_factor),
    penalty,
    penalized_objective = penalty_value(model, output.assignment),
  )
end

function warmup_solvers(; cutoff)
  instance = KnapsackInstance("warmup", [1, 1], [2, 1], 1)
  exact = brute_force_optimum(instance)
  settings = (
    iterations = 1,
    reads = 1,
    cutoff,
    time_limit = Inf,
    timing_samples = 1,
    benchmark_id = "warmup",
  )
  projection_row(instance, exact; settings...)
  penalty_row(instance, exact, 1.1; settings...)
  return nothing
end

"""
    benchmark_rows([instances]; penalty_factors, iterations, reads, cutoff,
                   time_limit, timing_samples, benchmark_id, warmup, on_row)

Run the hard-projection solve and a penalty-QUBO sensitivity sweep for every
instance. Returned rows report the original knapsack objective and feasibility,
not just each solver's encoded objective.

Final-state variance is calculated from the MPS supplied to `on_iteration`.
Truncation error remains unavailable because the callback runs after discarded
singular values have been removed.

When provided, `on_row` is called after each completed row so long runs can
report progress without changing the returned table.
"""
function benchmark_rows(
  instances = default_instances();
  penalty_factors = (0.001, 0.01, 0.1, 1.1),
  iterations = 6,
  reads = 64,
  cutoff = 1e-10,
  time_limit = 120.0,
  timing_samples = 3,
  benchmark_id = "unversioned",
  warmup = true,
  on_row = nothing,
)
  if !(iterations > 0)
    throw(ArgumentError("iterations must be positive"))
  end
  if !(reads > 0)
    throw(ArgumentError("reads must be positive"))
  end
  if !(time_limit > 0)
    throw(ArgumentError("time_limit must be positive"))
  end
  if !(timing_samples > 0)
    throw(ArgumentError("timing_samples must be positive"))
  end
  if !(isodd(timing_samples))
    throw(ArgumentError("timing_samples must be odd"))
  end
  if !(all(>(0), penalty_factors))
    throw(ArgumentError("penalty factors must be positive"))
  end

  if warmup
    warmup_solvers(; cutoff)
  end

  rows = NamedTuple[]
  for instance in instances
    exact = brute_force_optimum(instance)
    settings = (; iterations, reads, cutoff, time_limit, timing_samples, benchmark_id)
    projection = projection_row(instance, exact; settings...)
    push!(rows, projection)
    if !(isnothing(on_row))
      on_row(projection)
    end

    for factor in penalty_factors
      penalty = penalty_row(
        instance,
        exact,
        factor;
        iterations,
        reads,
        cutoff,
        time_limit,
        timing_samples,
        benchmark_id,
      )
      push!(rows, penalty)
      isnothing(on_row) || on_row(penalty)
    end
  end

  return rows
end

csv_value(::Missing) = ""
csv_value(value::AbstractString) = "\"" * replace(value, "\"" => "\"\"") * "\""
csv_value(value) = string(value)

"""
Write benchmark rows as CSV to `io`.
"""
function write_csv(io::IO, rows)
  if isempty(rows)
    return nothing
  end
  columns = keys(first(rows))
  println(io, join(string.(columns), ','))
  for row in rows
    println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
  end
  return nothing
end

end
