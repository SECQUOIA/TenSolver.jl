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
