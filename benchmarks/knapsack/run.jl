include("KnapsackBenchmark.jl")

using .KnapsackBenchmark

write_csv(stdout, run_benchmarks())
