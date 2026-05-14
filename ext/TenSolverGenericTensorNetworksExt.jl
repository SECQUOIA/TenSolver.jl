module TenSolverGenericTensorNetworksExt

import TenSolver
import GenericTensorNetworks
import ProblemReductions

struct PseudoBooleanCSP{T <: Real} <: ProblemReductions.ConstraintSatisfactionProblem{T}
  n     :: Int
  terms :: Vector{Pair{Vector{Int}, T}}
end

ProblemReductions.num_variables(problem::PseudoBooleanCSP) = problem.n
ProblemReductions.num_flavors(::Type{<:PseudoBooleanCSP}) = 2
ProblemReductions.problem_size(problem::PseudoBooleanCSP) = (; num_variables=problem.n, num_terms=length(problem.terms))
ProblemReductions.constraints(::PseudoBooleanCSP) = ProblemReductions.LocalConstraint[]
ProblemReductions.energy_mode(::Type{<:PseudoBooleanCSP}) = ProblemReductions.SmallerSizeIsBetter()
ProblemReductions.weights(problem::PseudoBooleanCSP) = [coeff for (_vars, coeff) in problem.terms]

function ProblemReductions.objectives(problem::PseudoBooleanCSP{T}) where {T}
  return map(problem.terms) do term
    vars, coeff = term.first, term.second
    spec = zeros(T, 2^length(vars))
    spec[end] = coeff
    ProblemReductions.LocalSolutionSize(2, vars, spec)
  end
end

function _qubo_matrix(model::TenSolver.PseudoBooleanModel{T}) where {T}
  matrix = zeros(T, length(model), length(model))

  for (vars, coeff) in model.terms
    if length(vars) == 1
      matrix[vars[1], vars[1]] += coeff
    elseif length(vars) == 2
      matrix[vars[1], vars[2]] += coeff
    else
      throw(ArgumentError("Cannot map degree-$(length(vars)) term to ProblemReductions.QUBO."))
    end
  end

  return matrix
end

function _gtn_problem(model::TenSolver.PseudoBooleanModel{T}) where {T}
  if TenSolver.isquadratic(model)
    return ProblemReductions.QUBO(_qubo_matrix(model))
  else
    return PseudoBooleanCSP{T}(length(model), collect(model.terms))
  end
end

function _generic_tensor_network(problem, backend::TenSolver.GTNBackend)
  optimizer = isnothing(backend.optimizer) ? GenericTensorNetworks.TreeSA() : backend.optimizer
  return GenericTensorNetworks.GenericTensorNetwork(problem; optimizer, slicer=backend.slicer)
end

function _property(property::Symbol, backend::TenSolver.GTNBackend)
  k = backend.k
  k >= 1 || throw(ArgumentError("GTNBackend k must be >= 1, got $k."))

  if property in (:size, :value)
    return k == 1 ? GenericTensorNetworks.SizeMin() : GenericTensorNetworks.SizeMin(k)
  elseif property in (:single, :config)
    return k == 1 ? GenericTensorNetworks.SingleConfigMin(; bounded=backend.bounded) : GenericTensorNetworks.SingleConfigMin(k; bounded=false)
  elseif property in (:count, :degeneracy)
    return k == 1 ? GenericTensorNetworks.CountingMin() : GenericTensorNetworks.CountingMin(k)
  elseif property in (:configs, :enumerate)
    return k == 1 ? GenericTensorNetworks.ConfigsMin(; bounded=backend.bounded, tree_storage=backend.tree_storage) :
                    GenericTensorNetworks.ConfigsMin(k; bounded=backend.bounded, tree_storage=backend.tree_storage)
  elseif property in (:kbest_sizes, :spectrum)
    return GenericTensorNetworks.SizeMin(k)
  else
    throw(ArgumentError("Unsupported GTN property `$(property)`. Supported properties are :size, :single, :count, :configs, and :kbest_sizes."))
  end
end

_scalar(x::AbstractArray) = x[]
_scalar(x) = x

_number(x::Number) = x
_number(x) = hasproperty(x, :n) ? getproperty(x, :n) : x

function _numbers(x)
  if x isa AbstractVector
    return map(_number, x)
  else
    return _number(x)
  end
end

function _primary_size(size)
  numbers = _numbers(size)
  return numbers isa AbstractVector ? minimum(numbers) : numbers
end

function _config_vector(config)
  return [Int(v) for v in collect(config)]
end

function _is_config(config)
  return config isa AbstractVector && all(v -> v isa Integer, config)
end

function _append_configs!(out, config)
  if config isa Pair
    _append_configs!(out, config.second)
  elseif _is_config(config)
    push!(out, _config_vector(config))
  elseif config isa AbstractVector
    foreach(c -> _append_configs!(out, c), config)
  else
    push!(out, _config_vector(config))
  end

  return out
end

function _configs_from_read_config(config)
  return _append_configs!(Vector{Int}[], config)
end

function _try_read_size(item)
  try
    return GenericTensorNetworks.read_size(item)
  catch
    return nothing
  end
end

function _try_read_count(item)
  try
    return GenericTensorNetworks.read_count(item)
  catch
    return nothing
  end
end

function _try_read_configs(item)
  try
    return _configs_from_read_config(GenericTensorNetworks.read_config(item; keeptree=false))
  catch
    return Vector{Int}[]
  end
end

function TenSolver._solve_backend(backend::TenSolver.GTNBackend, model::TenSolver.PseudoBooleanModel; cutoff=1e-8, property::Symbol=backend.property, kwargs...)
  problem = _gtn_problem(model)
  gtn = _generic_tensor_network(problem, backend)
  gtn_property = _property(property, backend)
  raw = GenericTensorNetworks.solve(gtn, gtn_property; T=backend.T, usecuda=backend.usecuda)
  item = _scalar(raw)

  size = _try_read_size(item)
  objective = isnothing(size) ? model.constant : model.constant + _primary_size(size)
  configs = _try_read_configs(item)
  count = _try_read_count(item)

  metadata = Dict{String, Any}(
    "backend" => "GenericTensorNetworks",
    "property" => property,
    "constant" => model.constant,
    "quadratic" => TenSolver.isquadratic(model),
    "size" => _numbers(size),
    "count" => count,
    "contraction_complexity" => GenericTensorNetworks.contraction_complexity(gtn),
  )

  try
    metadata["estimated_memory"] = GenericTensorNetworks.estimate_memory(gtn, gtn_property; T=backend.T)
  catch err
    metadata["estimated_memory_error"] = sprint(showerror, err)
  end

  return objective, TenSolver.GTNSolution(objective, configs, raw, property, metadata)
end

end
