include("KnapsackBenchmark.jl")

using .KnapsackBenchmark

const USAGE = """
usage: julia --project=benchmarks benchmarks/knapsack/run.jl [options] [output.csv]

options:
  --resources              emit only the projection resource-scaling table
  --pilot                  run only the four-item reference instance
  --iterations=N           maximum DMRG sweeps per timing sample (default: 6)
  --reads=N                samples drawn from each solution (default: 64)
  --timing-samples=N       odd number of fixed-seed timing repetitions (default: 3)
  --time-limit=SECONDS     per-solve soft limit, checked after each sweep (default: 120)
  --cutoff=VALUE           tensor truncation cutoff (default: 1e-10)
  --benchmark-id=ID        provenance label copied into every solver row
"""

function option_value(argument, name, parser)
  prefix = "--$(name)="
  if !(startswith(argument, prefix))
    return nothing
  end
  value = argument[(length(prefix) + 1):end]
  if isempty(value)
    throw(ArgumentError("--$(name) requires a value"))
  end
  return parser(value)
end

function parse_arguments(arguments)
  resource_mode = false
  pilot_mode = false
  output_path = nothing
  iterations = 6
  reads = 64
  timing_samples = 3
  time_limit = 120.0
  cutoff = 1e-10
  benchmark_id = "unversioned"

  for argument in arguments
    if argument == "--resources"
      resource_mode = true
    elseif argument == "--pilot"
      pilot_mode = true
    elseif argument == "--help"
      return (; help = true)
    elseif startswith(argument, "--iterations=")
      iterations =
        something(option_value(argument, "iterations", value -> parse(Int, value)),)
    elseif startswith(argument, "--reads=")
      reads = something(option_value(argument, "reads", value -> parse(Int, value)))
    elseif startswith(argument, "--timing-samples=")
      timing_samples =
        something(option_value(argument, "timing-samples", value -> parse(Int, value)))
    elseif startswith(argument, "--time-limit=")
      time_limit =
        something(option_value(argument, "time-limit", value -> parse(Float64, value)))
    elseif startswith(argument, "--cutoff=")
      cutoff = something(option_value(argument, "cutoff", value -> parse(Float64, value)),)
    elseif startswith(argument, "--benchmark-id=")
      benchmark_id = something(option_value(argument, "benchmark-id", identity))
    elseif startswith(argument, "-")
      throw(ArgumentError("unknown option $(repr(argument))\n$(USAGE)"))
    elseif isnothing(output_path)
      output_path = argument
    else
      throw(ArgumentError("only one output path is supported\n$(USAGE)"))
    end
  end

  return (;
    help = false,
    resource_mode,
    pilot_mode,
    output_path,
    iterations,
    reads,
    timing_samples,
    time_limit,
    cutoff,
    benchmark_id,
  )
end

function main(arguments)
  options = parse_arguments(arguments)
  if options.help
    print(USAGE)
    return nothing
  end

  rows = if options.resource_mode
    projection_scaling_rows(; cutoff = options.cutoff)
  else
    report_progress = function (row)
      penalty = ismissing(row.penalty_factor) ? "" : " penalty=$(row.penalty_factor)"
      println(
        stderr,
        "completed $(row.instance) $(row.method)$(penalty): " *
        "value=$(row.original_value) feasible=$(row.feasible) " *
        "gap=$(row.optimality_gap) time=$(round(row.end_to_end_wall_seconds; digits=3))s",
      )
      return nothing
    end
    instances = if options.pilot_mode
      [KnapsackBenchmark.reference_instance()]
    else
      KnapsackBenchmark.default_instances()
    end
    benchmark_rows(
      instances;
      iterations = options.iterations,
      reads = options.reads,
      timing_samples = options.timing_samples,
      time_limit = options.time_limit,
      cutoff = options.cutoff,
      benchmark_id = options.benchmark_id,
      on_row = report_progress,
    )
  end
  write_csv(stdout, rows)

  if !isnothing(options.output_path)
    open(options.output_path, "w") do io
      return write_csv(io, rows)
    end
  end
  return nothing
end

main(ARGS)
