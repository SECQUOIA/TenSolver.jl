module TenSolverGenericTensorNetworksExt

import TenSolver
import GenericTensorNetworks
import ProblemReductions

import MultivariatePolynomials: AbstractPolynomial, coefficient, effective_variables,
  isconstant, terms

struct PseudoBooleanCSP{T <: Real} <: ProblemReductions.ConstraintSatisfactionProblem{T}
  n::Int
  terms::Vector{Pair{Vector{Int}, T}}
end

ProblemReductions.num_variables(problem::PseudoBooleanCSP) = problem.n
ProblemReductions.num_flavors(::Type{<:PseudoBooleanCSP}) = 2
ProblemReductions.problem_size(problem::PseudoBooleanCSP) =
  (; num_variables=problem.n, num_terms=length(problem.terms))
ProblemReductions.constraints(::PseudoBooleanCSP) = ProblemReductions.LocalConstraint[]
ProblemReductions.energy_mode(::Type{<:PseudoBooleanCSP}) =
  ProblemReductions.SmallerSizeIsBetter()
ProblemReductions.weights(problem::PseudoBooleanCSP{T}) where {T} =
  T[term.second for term in problem.terms]

function ProblemReductions.objectives(problem::PseudoBooleanCSP{T}) where {T}
  return ProblemReductions.LocalSolutionSize{T}[
    ProblemReductions.LocalSolutionSize(
      2,
      variables,
      vcat(zeros(T, 2^length(variables) - 1), coefficient),
    )
    for (variables, coefficient) in problem.terms
  ]
end

function validate_gtn_inputs(domain, constraints, k, cutoff)
  domain_values = collect(TenSolver.validate_solve_domain(domain))
  domain_values == [0, 1] || throw(ArgumentError(
    "GTNBackend currently supports only the Boolean domain [0, 1], got $(repr(domain_values)).",
  ))
  isempty(constraints) || throw(ArgumentError(
    "GTNBackend does not yet support TenSolver constraints; use backend = :dmrg.",
  ))
  k >= 1 || throw(ArgumentError("GTNBackend k must be >= 1, got $k."))
  cutoff >= 0 || throw(ArgumentError("GTNBackend cutoff must be nonnegative, got $cutoff."))
  return nothing
end

function reject_gtn_keywords(kwargs)
  isempty(kwargs) && return nothing
  names = join(("`$name`" for name in keys(kwargs)), ", ")
  throw(ArgumentError("Unsupported GTNBackend solver keyword(s): $names"))
end

function qubo_problem(Q::AbstractMatrix{T}, l::AbstractVector{T}; cutoff) where {T <: Real}
  size(Q, 1) == size(Q, 2) || throw(DimensionMismatch(
    "QUBO matrix must be square, got size $(size(Q)).",
  ))
  n = size(Q, 1)
  length(l) == n || throw(DimensionMismatch(
    "Linear term length $(length(l)) does not match QUBO dimension $n.",
  ))

  matrix = zeros(T, n, n)
  for i in 1:n
    linear = Q[i, i] + l[i]
    abs(linear) > cutoff && (matrix[i, i] = linear)
    for j in (i + 1):n
      quadratic = Q[i, j] + Q[j, i]
      abs(quadratic) > cutoff && (matrix[i, j] = quadratic)
    end
  end

  return ProblemReductions.QUBO(matrix)
end

function polynomial_problem(p::AbstractPolynomial{T}; cutoff) where {T <: Real}
  model_variables = collect(effective_variables(p))
  indices = Dict(variable => i for (i, variable) in enumerate(model_variables))
  coefficients = Dict{Vector{Int}, T}()
  constant = zero(T)

  for term in terms(p)
    term_coefficient = coefficient(term)
    if isconstant(term)
      constant += term_coefficient
      continue
    end

    term_variables = sort!([indices[variable] for variable in effective_variables(term)])
    coefficients[term_variables] =
      get(coefficients, term_variables, zero(T)) + term_coefficient
  end

  model_terms = Pair{Vector{Int}, T}[]
  for (term_variables, term_coefficient) in coefficients
    abs(term_coefficient) > cutoff || continue
    push!(model_terms, term_variables => term_coefficient)
  end
  sort!(model_terms; by=term -> Tuple(term.first))

  return PseudoBooleanCSP{T}(length(model_variables), model_terms), constant
end

function generic_tensor_network(problem; optimizer, slicer)
  contraction_optimizer =
    isnothing(optimizer) ? GenericTensorNetworks.TreeSA() : optimizer
  return GenericTensorNetworks.GenericTensorNetwork(
    problem;
    optimizer=contraction_optimizer,
    slicer,
  )
end

function gtn_property(property::Symbol; k, bounded, tree_storage)
  if property in (:size, :value)
    return k == 1 ? GenericTensorNetworks.SizeMin() : GenericTensorNetworks.SizeMin(k)
  elseif property in (:single, :config)
    return k == 1 ?
      GenericTensorNetworks.SingleConfigMin(; bounded) :
      GenericTensorNetworks.SingleConfigMin(k; bounded=false)
  elseif property in (:count, :degeneracy)
    k == 1 || throw(ArgumentError(
      "GTNBackend property `$(property)` currently supports only k == 1.",
    ))
    return GenericTensorNetworks.CountingMin()
  elseif property in (:configs, :enumerate)
    k == 1 || throw(ArgumentError(
      "GTNBackend property `$(property)` currently supports only k == 1; " *
      "use property = :single with k > 1 for representative k-best configurations.",
    ))
    return GenericTensorNetworks.ConfigsMin(; bounded, tree_storage)
  elseif property in (:kbest_sizes, :spectrum)
    return GenericTensorNetworks.SizeMin(k)
  end

  throw(ArgumentError(
    "Unsupported GTN property `$(property)`. Supported properties are " *
    ":size, :single, :count, :configs, and :kbest_sizes.",
  ))
end

scalar_result(x::AbstractArray) = x[]
scalar_result(x) = x

unwrap_number(x) = hasproperty(x, :n) ? getproperty(x, :n) : x
unwrap_numbers(x::AbstractArray) = map(unwrap_number, x)
unwrap_numbers(x::Tuple) = map(unwrap_number, x)
unwrap_numbers(x) = unwrap_number(x)

function primary_size(raw_size)
  values = unwrap_numbers(raw_size)
  return values isa Union{AbstractArray, Tuple} ? minimum(values) : values
end

config_vector(config) = [Int(value) for value in collect(config)]
is_config(config) =
  config isa AbstractVector && all(value -> value isa Integer, config)

function append_configs!(output, config)
  if config isa Pair
    append_configs!(output, config.second)
  elseif is_config(config)
    push!(output, config_vector(config))
  elseif config isa AbstractVector
    foreach(item -> append_configs!(output, item), config)
  else
    push!(output, config_vector(config))
  end
  return output
end

configs_from_result(config) = append_configs!(Vector{Int}[], config)

function read_configs(item, property)
  property in (:single, :config, :configs, :enumerate) || return Vector{Int}[]
  return configs_from_result(
    GenericTensorNetworks.read_config(item; keeptree=false),
  )
end

function read_count(item, property)
  property in (:count, :degeneracy) || return nothing
  return GenericTensorNetworks.read_count(item)
end

function solve_gtn(
  problem,
  constant;
  property,
  k,
  usecuda,
  element_type,
  optimizer,
  slicer,
  bounded,
  tree_storage,
)
  network = generic_tensor_network(problem; optimizer, slicer)
  selected_property = gtn_property(property; k, bounded, tree_storage)
  raw = GenericTensorNetworks.solve(
    network,
    selected_property;
    T=element_type,
    usecuda,
  )
  item = scalar_result(raw)

  raw_size = GenericTensorNetworks.read_size(item)
  objective = constant + primary_size(raw_size)
  configs = read_configs(item, property)
  count = read_count(item, property)

  metadata = Dict{String, Any}(
    "backend" => "GenericTensorNetworks",
    "property" => property,
    "constant" => constant,
    "size" => unwrap_numbers(raw_size),
    "count" => count,
  )

  try
    metadata["contraction_complexity"] =
      GenericTensorNetworks.contraction_complexity(network)
  catch error
    metadata["contraction_complexity_error"] = sprint(showerror, error)
  end

  try
    metadata["estimated_memory"] = GenericTensorNetworks.estimate_memory(
      network,
      selected_property;
      T=element_type,
    )
  catch error
    metadata["estimated_memory_error"] = sprint(showerror, error)
  end

  return objective, TenSolver.GTNSolution(configs, raw, property, metadata)
end

function TenSolver.minimize(
  ::TenSolver.GTNBackend,
  Q::AbstractMatrix{T},
  l::AbstractVector{T},
  c::T;
  property::Symbol=:single,
  k::Int=1,
  usecuda::Bool=false,
  element_type::Type=Float64,
  optimizer=nothing,
  slicer=nothing,
  bounded::Bool=true,
  tree_storage::Bool=false,
  cutoff::Real=1e-8,
  domain=0:1,
  constraints=TenSolver.AbstractConstraint[],
  kwargs...,
) where {T <: Real}
  validate_gtn_inputs(domain, constraints, k, cutoff)
  reject_gtn_keywords(kwargs)
  problem = qubo_problem(Q, l; cutoff)
  return solve_gtn(
    problem,
    c;
    property,
    k,
    usecuda,
    element_type,
    optimizer,
    slicer,
    bounded,
    tree_storage,
  )
end

function TenSolver.minimize(
  ::TenSolver.GTNBackend,
  p::AbstractPolynomial{T};
  property::Symbol=:single,
  k::Int=1,
  usecuda::Bool=false,
  element_type::Type=Float64,
  optimizer=nothing,
  slicer=nothing,
  bounded::Bool=true,
  tree_storage::Bool=false,
  cutoff::Real=1e-8,
  domain=0:1,
  constraints=TenSolver.AbstractConstraint[],
  kwargs...,
) where {T <: Real}
  validate_gtn_inputs(domain, constraints, k, cutoff)
  reject_gtn_keywords(kwargs)
  problem, constant = polynomial_problem(p; cutoff)
  return solve_gtn(
    problem,
    constant;
    property,
    k,
    usecuda,
    element_type,
    optimizer,
    slicer,
    bounded,
    tree_storage,
  )
end

end
