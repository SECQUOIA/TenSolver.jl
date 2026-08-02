module KnapsackBenchmark

using DelimitedFiles: writedlm
using LinearAlgebra: dot
using Random: AbstractRNG, MersenneTwister, Random, rand

import BenchmarkTools
import TenSolver

export projection_scaling_rows, run_benchmarks, write_csv

const RNG_SEED = 66
const INSTANCE_RNG = MersenneTwister(RNG_SEED)
Random.seed!(RNG_SEED)

# -----------------------------------------------------------------------------
# Types and infrastructure
# -----------------------------------------------------------------------------

struct KnapsackInstance
  name::String
  weights::Vector{Int}
  values::Vector{Float64}
  capacity::Int

  function KnapsackInstance(name, weights, values, capacity)
    length(weights) == length(values) ||
      throw(DimensionMismatch("weights and values must have the same length"))
    isempty(weights) &&
      throw(ArgumentError("a knapsack instance must contain at least one item"))
    all(>(0), weights) || throw(ArgumentError("item weights must be positive integers"))
    all(>(0), values) || throw(ArgumentError("item values must be positive"))
    capacity >= 0 || throw(ArgumentError("capacity must be nonnegative"))
    return new(String(name), Int.(weights), Float64.(values), Int(capacity))
  end
end

struct BenchmarkSettings
  iterations::Int
  reads::Int
  cutoff::Float64
  time_limit::Float64
  timing_samples::Int
end

struct KnapsackMethod
  name::String
  penalty_factor::Union{Missing, Float64}
  formulate::Function
  solve::Function
end

struct SuiteEntry
  instance_key::String
  method_key::String
  outputs::Vector{Any}
end

struct BenchmarkReport
  metadata::Vector{Pair{String, String}}
  rows::Vector{NamedTuple}
end

function validate(settings::BenchmarkSettings, penalty_factors)
  settings.iterations > 0 || throw(ArgumentError("iterations must be positive"))
  settings.reads > 0 || throw(ArgumentError("reads must be positive"))
  settings.cutoff > 0 || throw(ArgumentError("cutoff must be positive"))
  settings.time_limit > 0 || throw(ArgumentError("time_limit must be positive"))
  settings.timing_samples > 0 || throw(ArgumentError("timing_samples must be positive"))
  isodd(settings.timing_samples) ||
    throw(ArgumentError("timing_samples must be odd"))
  all(>(0), penalty_factors) ||
    throw(ArgumentError("penalty factors must be positive"))
  return nothing
end

function runtime_metadata(settings::BenchmarkSettings, benchmark_id)
  return [
    "benchmark_id" => string(benchmark_id),
    "julia_version" => string(VERSION),
    "julia_threads" => string(Threads.nthreads()),
    "system" => string(Sys.KERNEL),
    "architecture" => string(Sys.ARCH),
    "tensolver_version" => string(Base.pkgversion(TenSolver)),
    "benchmarktools_version" => string(Base.pkgversion(BenchmarkTools)),
    "rng_seed" => string(RNG_SEED),
    "iterations" => string(settings.iterations),
    "reads" => string(settings.reads),
    "cutoff" => string(settings.cutoff),
    "time_limit_seconds" => string(settings.time_limit),
    "timing_samples" => string(settings.timing_samples),
  ]
end

function write_csv(io::IO, rows)
  isempty(rows) && return nothing
  columns = propertynames(first(rows))
  table = Matrix{Any}(undef, length(rows) + 1, length(columns))
  table[1, :] .= string.(columns)
  for (i, row) in enumerate(rows), (j, column) in enumerate(columns)
    value = getproperty(row, column)
    table[i + 1, j] = ismissing(value) ? "" : value
  end
  writedlm(io, table, ',')
  return nothing
end

function write_csv(io::IO, report::BenchmarkReport)
  for (key, value) in report.metadata
    println(io, "# $(key)=$(value)")
  end
  return write_csv(io, report.rows)
end

# -----------------------------------------------------------------------------
# Instance definitions and building helpers
# -----------------------------------------------------------------------------

reference_instance() =
  KnapsackInstance("reference_4", [4, 3, 2, 3], [8, 4, 5, 3], 6)

function pisinger_instance(
  rng::AbstractRNG,
  kind,
  n;
  coefficient_range = 10,
)
  n > 0 || throw(ArgumentError("instance size must be positive"))
  coefficient_range >= 10 ||
    throw(ArgumentError("coefficient range must be at least 10"))

  weights = rand(rng, 1:coefficient_range, n)
  correlation_range = div(coefficient_range, 10)
  values = if kind == :uncorrelated
    rand(rng, 1:coefficient_range, n)
  elseif kind == :weakly_correlated
    max.(1, weights .+ rand(rng, (-correlation_range):correlation_range, n))
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

function default_instances(rng::AbstractRNG = copy(INSTANCE_RNG))
  specifications = (
    (:uncorrelated, 8),
    (:weakly_correlated, 12),
    (:strongly_correlated, 16),
    (:subset_sum, 16),
  )
  generated = map(specifications) do (kind, n)
    return pisinger_instance(rng, kind, n)
  end
  return [reference_instance(), generated...]
end

item_weight(instance::KnapsackInstance, items) = dot(instance.weights, items)
item_value(instance::KnapsackInstance, items) = dot(instance.values, items)
is_capacity_feasible(instance::KnapsackInstance, items) =
  item_weight(instance, items) <= instance.capacity

function brute_force_optimum(instance::KnapsackInstance)
  best = (value = -Inf, weight = typemax(Int), items = Int[])
  for assignment in Iterators.product(fill(0:1, length(instance.weights))...)
    items = collect(assignment)
    weight = item_weight(instance, items)
    value = item_value(instance, items)
    if weight <= instance.capacity &&
       (value > best.value || (value == best.value && weight < best.weight))
      best = (; value, weight, items)
    end
  end
  return best
end

function slack_weights(capacity::Integer)
  capacity >= 0 || throw(ArgumentError("capacity must be nonnegative"))
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

function penalty_qubo(instance::KnapsackInstance, penalty::Real)
  penalty > 0 || throw(ArgumentError("penalty must be positive"))
  slack = slack_weights(instance.capacity)
  coefficients = Float64.([instance.weights; slack])
  values = [instance.values; zeros(length(slack))]
  lambda = Float64(penalty)
  Q = lambda .* (coefficients * coefficients')
  l = -values .- (2lambda * instance.capacity) .* coefficients
  constant = lambda * instance.capacity^2
  return (; Q, l, constant, nitems = length(instance.weights))
end

penalty_value(model, assignment) =
  dot(assignment, model.Q, assignment) +
  dot(model.l, assignment) +
  model.constant

item_bits(sample, nitems) = round.(Int, sample[1:nitems])

function best_sample(samples, decode, rank)
  best = decode(first(samples))
  best_rank = rank(best)
  for sample in Iterators.drop(samples, 1)
    candidate = decode(sample)
    candidate_rank = rank(candidate)
    if candidate_rank < best_rank
      best = candidate
      best_rank = candidate_rank
    end
  end
  return best
end

function projection_method()
  formulate = function (instance)
    return TenSolver.SumConstraint(
      collect(eachindex(instance.weights)),
      instance.weights,
      instance.capacity;
      relation = :(<=),
    )
  end
  solve = function (instance, constraint, settings)
    reported_objective, solution = TenSolver.maximize(
      instance.values;
      constraints = [constraint],
      solver_options(settings)...,
    )
    decode = sample -> item_bits(sample, length(instance.weights))
    rank = items -> (
      !is_capacity_feasible(instance, items),
      -item_value(instance, items),
      item_weight(instance, items),
    )
    items = best_sample(TenSolver.sample(solution, settings.reads), decode, rank)
    return (;
      reported_objective,
      solution,
      items,
      nvariables = length(instance.weights),
      penalty = missing,
      penalized_objective = missing,
    )
  end
  return KnapsackMethod("projection", missing, formulate, solve)
end

function penalty_method(penalty_factor)
  factor = Float64(penalty_factor)
  formulate = function (instance)
    return penalty_qubo(instance, factor * sum(instance.values))
  end
  solve = function (instance, model, settings)
    reported_objective, solution = TenSolver.minimize(
      model.Q,
      model.l,
      model.constant;
      solver_options(settings)...,
    )
    decode = function (sample)
      assignment = round.(Int, sample)
      return (; assignment, items = item_bits(assignment, model.nitems))
    end
    rank = candidate -> (
      penalty_value(model, candidate.assignment),
      -item_value(instance, candidate.items),
      item_weight(instance, candidate.items),
    )
    best = best_sample(TenSolver.sample(solution, settings.reads), decode, rank)
    return (;
      reported_objective,
      solution,
      items = best.items,
      nvariables = length(best.assignment),
      penalty = factor * sum(instance.values),
      penalized_objective = penalty_value(model, best.assignment),
    )
  end
  return KnapsackMethod("penalty_$(factor)", factor, formulate, solve)
end

function benchmark_methods(penalty_factors)
  return (projection_method(), (penalty_method(factor) for factor in penalty_factors)...)
end

function projection_scaling_instances()
  probes = NamedTuple[]
  for capacity in (1, 2, 4, 8)
    nitems = 16
    instance =
      KnapsackInstance("capacity_$(capacity)", ones(Int, nitems), ones(Int, nitems), capacity)
    push!(probes, (sweep = "capacity", instance))
  end
  for nitems in (8, 16, 32)
    instance =
      KnapsackInstance("items_$(nitems)", ones(Int, nitems), ones(Int, nitems), 3)
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

# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------

function solver_options(settings::BenchmarkSettings)
  return (
    iterations = settings.iterations,
    time_limit = settings.time_limit,
    cutoff = settings.cutoff,
    inidim = 8,
    maxdim = [10, 20, 40, 80, 120, 200],
    noise = [1e-6, 1e-8, 0.0],
    check_variance_every_iteration = typemax(Int),
    vtol = -Inf,
    verbosity = 0,
  )
end

function result_row(
  instance::KnapsackInstance,
  exact,
  method::KnapsackMethod,
  output,
  settings::BenchmarkSettings,
)
  stats = output.solution.stats
  bonds = stats.max_bonds
  elapsed = isempty(stats.elapsed_times) ? 0.0 : last(stats.elapsed_times)
  feasible = is_capacity_feasible(instance, output.items)
  value = item_value(instance, output.items)
  return (
    instance = instance.name,
    method = method.name,
    nitems = length(instance.weights),
    nvariables = output.nvariables,
    capacity = instance.capacity,
    penalty_factor = method.penalty_factor,
    penalty = output.penalty,
    exact_value = exact.value,
    original_value = value,
    feasible,
    optimality_gap = feasible ? exact.value - value : missing,
    penalized_objective = output.penalized_objective,
    solver_reported_objective = output.reported_objective,
    sweeps = length(stats.energies),
    time_limit_reached =
      length(stats.energies) < settings.iterations && elapsed >= settings.time_limit,
    solver_elapsed_seconds = elapsed,
    solution_max_bond = isempty(stats.bond_dims) ? 0 : maximum(stats.bond_dims),
    objective_mpo_bond = bonds.objective,
    projection_mpo_bond =
      isempty(bonds.projections) ? missing : maximum(bonds.projections),
    effective_hamiltonian_bond = bonds.hamiltonian,
  )
end

function execute_case(
  instance::KnapsackInstance,
  exact,
  method::KnapsackMethod,
  settings::BenchmarkSettings,
)
  model = method.formulate(instance)
  output = method.solve(instance, model, settings)
  return result_row(instance, exact, method, output, settings)
end

function benchmark_suite(instances, methods, settings::BenchmarkSettings)
  suite = BenchmarkTools.BenchmarkGroup()
  entries = SuiteEntry[]
  samples = settings.timing_samples
  for instance in instances
    suite[instance.name] = BenchmarkTools.BenchmarkGroup()
    exact = brute_force_optimum(instance)
    for method in methods
      outputs = Any[]
      runner = function ()
        push!(outputs, execute_case(instance, exact, method, settings))
        return nothing
      end
      suite[instance.name][method.name] =
        BenchmarkTools.@benchmarkable $runner() samples=samples evals=1 seconds=3600
      push!(entries, SuiteEntry(instance.name, method.name, outputs))
    end
  end
  return suite, entries
end

function measured_rows(trials, entries)
  rows = NamedTuple[]
  for entry in entries
    trial = trials[entry.instance_key][entry.method_key]
    sample_count = length(trial.times)
    length(entry.outputs) >= sample_count ||
      error("BenchmarkTools did not capture every measured output")
    outputs = last(entry.outputs, sample_count)
    median_index = sortperm(trial.times)[cld(sample_count, 2)]
    row = merge(
      outputs[median_index],
      (end_to_end_wall_seconds = trial.times[median_index] / 1e9,),
    )
    push!(rows, row)
  end
  return rows
end

"""
Run the penalty-versus-projection benchmark as a declarative BenchmarkTools
suite. Pass keywords directly when calling this function from the REPL.
"""
function run_benchmarks(
  instances = default_instances();
  penalty_factors = (0.001, 0.01, 0.1, 1.1),
  iterations = 6,
  reads = 64,
  cutoff = 1e-10,
  time_limit = 120.0,
  timing_samples = 3,
  benchmark_id = "unversioned",
  verbose = true,
)
  settings = BenchmarkSettings(
    Int(iterations),
    Int(reads),
    Float64(cutoff),
    Float64(time_limit),
    Int(timing_samples),
  )
  validate(settings, penalty_factors)
  suite, entries = benchmark_suite(
    instances,
    benchmark_methods(penalty_factors),
    settings,
  )
  trials = BenchmarkTools.run(suite; verbose)
  return BenchmarkReport(
    runtime_metadata(settings, benchmark_id),
    measured_rows(trials, entries),
  )
end

"""
Build the projection resource table through TenSolver's public solver API.
"""
function projection_scaling_rows(; cutoff = 1e-10, time_limit = 60.0)
  settings = BenchmarkSettings(1, 1, Float64(cutoff), Float64(time_limit), 1)
  return map(projection_scaling_instances()) do probe
    instance = probe.instance
    constraint = TenSolver.SumConstraint(
      collect(eachindex(instance.weights)),
      instance.weights,
      instance.capacity;
      relation = :(<=),
    )
    _, solution = TenSolver.maximize(
      instance.values;
      constraints = [constraint],
      solver_options(settings)...,
    )
    bonds = solution.stats.max_bonds
    return (
      sweep = probe.sweep,
      instance = instance.name,
      nitems = length(instance.weights),
      capacity = instance.capacity,
      max_weight = maximum(instance.weights),
      capacity_state_bound = instance.capacity + 2,
      objective_mpo_bond = bonds.objective,
      projection_mpo_bond = maximum(bonds.projections),
      effective_hamiltonian_bond = bonds.hamiltonian,
    )
  end
end

end
