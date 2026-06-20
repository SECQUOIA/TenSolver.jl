"""
    AbstractBackend

Solver backend selector for TenSolver objective dispatch.
"""
abstract type AbstractBackend end

"""
    DMRGBackend()

Default approximate backend based on ITensors/DMRG.
"""
struct DMRGBackend <: AbstractBackend end

"""
    GTNBackend(; property=:single, k=1, usecuda=false, T=Float64, optimizer=nothing,
                 slicer=nothing, bounded=true, tree_storage=false)

Optional exact contraction backend powered by GenericTensorNetworks.jl.

The backend is available when the `TenSolverGenericTensorNetworksExt` package
extension is loaded, which requires `using GenericTensorNetworks, ProblemReductions`.
Supported properties are `:size`, `:single`, `:count`, `:configs`, and
`:kbest_sizes`. Values `k > 1` are supported for `:size`, `:single`, and
`:kbest_sizes`; `:count` and `:configs` currently require `k == 1`.
"""
Base.@kwdef struct GTNBackend <: AbstractBackend
  property     :: Symbol = :single
  k            :: Int = 1
  usecuda      :: Bool = false
  T            :: Type = Float64
  optimizer             = nothing
  slicer                = nothing
  bounded      :: Bool = true
  tree_storage :: Bool = false
end

function _solve_backend(::GTNBackend, model; kwargs...)
  throw(ArgumentError("GTNBackend requires loading GenericTensorNetworks and ProblemReductions before use. Run `using GenericTensorNetworks, ProblemReductions` and retry."))
end

"""
    solution_space(args...; property=nothing, backend=GTNBackend(property=:configs), kwargs...)

Compute exact solution-space information with a backend that supports it.
Currently this is implemented by `GTNBackend` when GenericTensorNetworks is loaded.
Use `GTNBackend(property=:single, k=n)` for representative k-best configurations;
full enumeration and degeneracy counting currently require `k == 1`.
"""
function solution_space(args...; property::Union{Symbol, Nothing}=nothing, backend=GTNBackend(property=isnothing(property) ? :configs : property), kwargs...)
  model = pseudoboolean(args...)
  selected_backend = backend isa AbstractBackend ? backend : backend_from_attribute(backend; property=isnothing(property) ? :configs : property)
  selected_backend isa GTNBackend || throw(ArgumentError("solution_space requires a backend that supports exact solution-space properties; use GTNBackend after loading GenericTensorNetworks and ProblemReductions."))
  selected_property = isnothing(property) && selected_backend isa GTNBackend ? selected_backend.property : (isnothing(property) ? :configs : property)
  return _solve_backend(selected_backend, model; property=selected_property, kwargs...)
end

_with_objective(solution, objective) = solution

function backend_from_attribute(value; property::Symbol=:single, k::Int=1, usecuda::Bool=false, T::Type=Float64)
  if value isa AbstractBackend
    return value
  elseif value == "dmrg" || value == :dmrg
    return DMRGBackend()
  elseif value == "gtn" || value == :gtn
    return GTNBackend(; property, k, usecuda, T)
  else
    throw(ArgumentError("Unsupported TenSolver backend attribute `$(value)`. Use `\"dmrg\"`, `\"gtn\"`, `DMRGBackend()`, or `GTNBackend()`."))
  end
end
