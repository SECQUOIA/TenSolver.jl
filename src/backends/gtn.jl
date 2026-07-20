"""
    GTNBackend()

Select the optional exact GenericTensorNetworks backend.

Load both GenericTensorNetworks.jl and ProblemReductions.jl before solving.
Backend settings such as `property`, `k`, and `usecuda` are solve-call keywords,
so the backend object carries no tunable state.
"""
struct GTNBackend <: AbstractTenSolverBackend end

normalize_backend(::Val{:gtn}) = GTNBackend()

function minimize(::GTNBackend, args...; kwargs...)
  throw(ArgumentError(
    "GTNBackend requires GenericTensorNetworks and ProblemReductions. " *
    "Load them with `using GenericTensorNetworks, ProblemReductions` and retry.",
  ))
end

"""
    solution_space(args...; property=:configs, backend=GTNBackend(), kwargs...)

Compute exact solution-space information with a backend that supports it.

The optional [`GTNBackend`](@ref) supports `:size`, `:single`, `:count`,
`:configs`, and `:kbest_sizes`. Use `k` to request k-best sizes or
representative configurations. Full configuration enumeration and counting
currently require `k == 1`.
"""
function solution_space(
  args...;
  backend=GTNBackend(),
  kwargs...,
)
  return solution_space(normalize_backend(backend), args...; kwargs...)
end

function solution_space(::AbstractTenSolverBackend, args...; kwargs...)
  throw(ArgumentError(
    "The selected backend does not provide exact solution-space support; " *
    "load GenericTensorNetworks and ProblemReductions and use backend = :gtn.",
  ))
end

function solution_space(
  backend::GTNBackend,
  args...;
  property::Symbol=:configs,
  kwargs...,
)
  return minimize(args...; backend, property, kwargs...)
end
