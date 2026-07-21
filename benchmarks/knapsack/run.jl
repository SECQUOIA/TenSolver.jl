include("KnapsackBenchmark.jl")

using .KnapsackBenchmark

resource_mode = !isempty(ARGS) && first(ARGS) == "--resources"
output_args = resource_mode ? ARGS[2:end] : ARGS
length(output_args) <= 1 || throw(
  ArgumentError(
    "usage: julia --project=. benchmarks/knapsack/run.jl [--resources] [output.csv]",
  ),
)

rows = resource_mode ? projection_scaling_rows() : benchmark_rows()
write_csv(stdout, rows)

if !isempty(output_args)
  open(only(output_args), "w") do io
    write_csv(io, rows)
  end
end
