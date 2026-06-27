module PEPSBenchmarks

using LinearAlgebra
using Printf
using Random

using TenSolver

export BenchmarkProblem,
       brute_force,
       has_peps_components,
       king_problem,
       objective_value,
       print_results,
       run_benchmark,
       square_problem

const PEPS_COMPONENTS = (
  "SpinGlassNetworks",
  "SpinGlassEngine",
  "SpinGlassTensors",
)

struct BenchmarkProblem{T, B}
  name :: String
  topology :: B
  Q :: Matrix{T}
  l :: Vector{T}
  c :: T
end

_grid_index(row::Integer, col::Integer, n::Integer) = (row - 1) * n + col

function _add_interaction!(Q::AbstractMatrix, a::Integer, b::Integer, value)
  i, j = minmax(a, b)
  Q[i, j] += value
  return Q
end

function _grid_problem(name, topology, m::Integer, n::Integer; seed::Integer, diagonal::Bool)
  rng = MersenneTwister(seed)
  variables = m * n
  Q = zeros(Float64, variables, variables)
  l = -0.85 .+ 0.25 .* randn(rng, variables)

  for row in 1:m, col in 1:n
    here = _grid_index(row, col, n)

    if col < n
      _add_interaction!(Q, here, _grid_index(row, col + 1, n), 0.45 + 0.15 * rand(rng))
    end

    if row < m
      _add_interaction!(Q, here, _grid_index(row + 1, col, n), 0.45 + 0.15 * rand(rng))
    end

    if diagonal && row < m && col < n
      _add_interaction!(Q, here, _grid_index(row + 1, col + 1, n), 0.20 + 0.10 * rand(rng))
    end

    if diagonal && row < m && col > 1
      _add_interaction!(Q, here, _grid_index(row + 1, col - 1, n), 0.20 + 0.10 * rand(rng))
    end
  end

  return BenchmarkProblem(name, topology, Q, l, 0.0)
end

function square_problem(m::Integer = 3, n::Integer = 3; seed::Integer = 2026)
  return _grid_problem("square-$m-x-$n", TenSolver.SquareGrid(m, n), m, n; seed, diagonal = false)
end

function king_problem(m::Integer = 3, n::Integer = 3; seed::Integer = 2027)
  return _grid_problem("king-$m-x-$n", TenSolver.KingGrid(m, n), m, n; seed, diagonal = true)
end

function objective_value(problem::BenchmarkProblem, x::AbstractVector)
  return dot(x, problem.Q, x) + dot(problem.l, x) + problem.c
end

function brute_force(problem::BenchmarkProblem; max_variables::Integer = 20)
  variables = length(problem.l)
  variables <= max_variables || return nothing

  best_value = Inf
  best_state = zeros(Int, variables)
  limit = UInt(1) << variables

  for mask in UInt(0):(limit - UInt(1))
    state = Vector{Int}(undef, variables)
    for i in 1:variables
      state[i] = Int((mask >> (i - 1)) & UInt(1))
    end

    value = objective_value(problem, state)
    if value < best_value
      best_value = value
      best_state = state
    end
  end

  return (; value = best_value, state = best_state)
end

has_peps_components() = all(pkg -> !isnothing(Base.find_package(pkg)), PEPS_COMPONENTS)

function load_peps_components()
  has_peps_components() || return false

  try
    @eval import SpinGlassNetworks
    @eval import SpinGlassEngine
    @eval import SpinGlassTensors
    return true
  catch err
    @warn "Could not load optional SpinGlassPEPS component packages" exception = (err, catch_backtrace())
    return false
  end
end

function _gap(value, exact)
  exact === nothing && return missing
  ismissing(value) && return missing
  return Float64(value - exact.value)
end

function _result(backend, status, objective, gap, runtime, states, discarded, transform, note)
  return (;
    backend,
    status,
    objective,
    gap,
    runtime,
    states,
    discarded,
    transform,
    note,
  )
end

function _error_result(backend, err)
  return _result(
    backend,
    "error",
    missing,
    missing,
    missing,
    missing,
    missing,
    missing,
    sprint(showerror, err),
  )
end

function _brute_force_result(problem, exact; max_variables::Integer)
  elapsed = @elapsed exact[] = brute_force(problem; max_variables)
  if exact[] === nothing
    return _result(
      "Brute force",
      "skipped",
      missing,
      missing,
      elapsed,
      missing,
      missing,
      missing,
      "more than $max_variables variables",
    )
  end

  return _result(
    "Brute force",
    "ok",
    Float64(exact[].value),
    0.0,
    elapsed,
    1,
    missing,
    missing,
    "",
  )
end

function _dmrg_result(problem, exact; kwargs...)
  try
    energy = nothing
    elapsed = @elapsed energy, _ = TenSolver.minimize(
      problem.Q,
      problem.l,
      problem.c;
      backend = :dmrg,
      verbosity = 0,
      kwargs...,
    )

    return _result(
      "DMRG",
      "ok",
      Float64(energy),
      _gap(energy, exact[]),
      elapsed,
      missing,
      missing,
      missing,
      "",
    )
  catch err
    return _error_result("DMRG", err)
  end
end

function _peps_result(problem, exact; kwargs...)
  if !load_peps_components()
    return _result(
      "PEPS",
      "skipped",
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      "SpinGlassPEPS component stack unavailable or not importable",
    )
  end

  try
    backend = TenSolver.PEPSBackend(problem.topology; kwargs...)
    energy = nothing
    solution = nothing
    elapsed = @elapsed energy, solution = TenSolver.minimize(
      problem.Q,
      problem.l,
      problem.c;
      backend,
      verbosity = 0,
    )
    metadata = solution.metadata

    return _result(
      "PEPS",
      "ok",
      Float64(energy),
      _gap(energy, exact[]),
      elapsed,
      length(solution.states),
      get(metadata, "largest_discarded_probability", missing),
      get(metadata, "selected_transformation", missing),
      "",
    )
  catch err
    return _error_result("PEPS", err)
  end
end

function run_benchmark(problem::BenchmarkProblem;
                       dmrg_kwargs = (;
                         iterations = 20,
                         maxdim = [4, 8, 16, 32],
                         noise = [1e-5, 1e-6, 1e-8, 0.0],
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
                       brute_force_limit::Integer = 20)
  exact = Ref{Union{Nothing, NamedTuple}}(nothing)
  results = [
    _brute_force_result(problem, exact; max_variables = brute_force_limit),
    _dmrg_result(problem, exact; dmrg_kwargs...),
    _peps_result(problem, exact; peps_kwargs...),
  ]

  return (; problem, exact = exact[], results)
end

function _format_value(value)
  ismissing(value) && return "-"
  value === nothing && return "-"
  value isa AbstractString && return value
  return @sprintf("%.6g", Float64(value))
end

function _format_text(value; limit::Integer = 36)
  ismissing(value) && return "-"
  value === nothing && return "-"
  text = string(value)
  return length(text) <= limit ? text : string(first(text, limit - 3), "...")
end

function print_results(run)
  problem = run.problem
  println("Problem: ", problem.name)
  println("Variables: ", length(problem.l))
  println("Topology: ", typeof(problem.topology))
  if run.exact === nothing
    println("Exact value: not computed")
  else
    println("Exact value: ", _format_value(run.exact.value))
  end
  println()
  @printf("%-12s %-9s %14s %12s %10s %8s %12s %s\n",
          "backend", "status", "objective", "gap", "time_s", "states", "discarded", "transform/note")
  @printf("%-12s %-9s %14s %12s %10s %8s %12s %s\n",
          repeat("-", 12), repeat("-", 9), repeat("-", 14), repeat("-", 12),
          repeat("-", 10), repeat("-", 8), repeat("-", 12), repeat("-", 24))

  for result in run.results
    note = isempty(result.note) ? result.transform : result.note
    @printf("%-12s %-9s %14s %12s %10s %8s %12s %s\n",
            result.backend,
            result.status,
            _format_value(result.objective),
            _format_value(result.gap),
            _format_value(result.runtime),
            _format_value(result.states),
            _format_value(result.discarded),
            _format_text(note))
  end

  return nothing
end

end # module
