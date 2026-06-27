include(joinpath(@__DIR__, "peps_common.jl"))

using .PEPSBenchmarks
using Random

Random.seed!(2027)

problem = king_problem(2, 3; seed = 2027)
run = run_benchmark(
  problem;
  dmrg_kwargs = (;
    iterations = 10,
    maxdim = [4, 8, 16],
    noise = [1e-5, 1e-7, 0.0],
  ),
  peps_kwargs = (;
    beta = 2.0,
    bond_dim = 8,
    max_states = 64,
    cutoff_prob = 0.0,
    contraction = :svd,
    transformations = :identity,
    no_cache = true,
  ),
  brute_force_limit = 20,
)

print_results(run)
