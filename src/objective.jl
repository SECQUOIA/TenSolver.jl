import MultivariatePolynomials: AbstractPolynomial, coefficient, effective_variables, isconstant, terms

function issquare(a :: AbstractArray)
  return allequal(size(a))
end

"""
    PseudoBooleanModel

Internal canonical form for a pseudo-Boolean objective over variables in `{0, 1}`.

The objective is

    constant + sum(coeff * prod(x[i] for i in vars) for (vars, coeff) in terms)

where every `vars` vector is sorted and contains no duplicates. This canonicalization
uses the binary identity `x_i^k == x_i` for positive integer powers.
"""
struct PseudoBooleanModel{T <: Real, V}
  variables :: Vector{V}
  terms     :: Dict{Vector{Int}, T}
  constant  :: T
end

Base.length(model::PseudoBooleanModel) = length(model.variables)

function _add_term!(terms::Dict{Vector{Int}, T}, vars::Vector{Int}, coeff::T; cutoff = zero(T)) where {T}
  abs(coeff) > cutoff || return terms
  key = sort!(unique(vars))
  value = get(terms, key, zero(T)) + coeff

  if abs(value) > cutoff
    terms[key] = value
  else
    delete!(terms, key)
  end

  return terms
end

function pseudoboolean(Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); cutoff=zero(T)) where {T <: Real}
  issquare(Q) || throw(DimensionMismatch("QUBO matrix must be square, got size $(size(Q))."))

  n = size(Q, 1)
  if !isnothing(l) && length(l) != n
    throw(DimensionMismatch("Linear term length $(length(l)) does not match QUBO dimension $n."))
  end

  terms = Dict{Vector{Int}, T}()

  for i in 1:n
    coeff = Q[i, i] + (isnothing(l) ? zero(T) : l[i])
    _add_term!(terms, [i], coeff; cutoff)

    for j in (i + 1):n
      _add_term!(terms, [i, j], Q[i, j] + Q[j, i]; cutoff)
    end
  end

  return PseudoBooleanModel{T, Int}(collect(1:n), terms, c)
end

function pseudoboolean(Q::AbstractMatrix{T}, c::T; cutoff=zero(T)) where {T <: Real}
  return pseudoboolean(Q, nothing, c; cutoff)
end

function pseudoboolean(p::AbstractPolynomial{T}; cutoff=zero(T)) where {T <: Real}
  vars = collect(effective_variables(p))
  indices = Dict(v => i for (i, v) in enumerate(vars))
  terms_dict = Dict{Vector{Int}, T}()
  constant = zero(T)

  for t in terms(p)
    coeff = coefficient(t)

    if isconstant(t)
      constant += coeff
    else
      term_vars = [indices[v] for v in effective_variables(t)]
      _add_term!(terms_dict, term_vars, coeff; cutoff)
    end
  end

  return PseudoBooleanModel{T, eltype(vars)}(vars, terms_dict, constant)
end

function evaluate(model::PseudoBooleanModel{T}, xs) where {T}
  length(xs) == length(model) || throw(DimensionMismatch("Configuration length $(length(xs)) does not match model dimension $(length(model))."))

  value = model.constant
  for (vars, coeff) in model.terms
    selected = true
    for i in vars
      selected &= !iszero(xs[i])
    end
    selected && (value += coeff)
  end

  return value
end

function negate(model::PseudoBooleanModel{T, V}) where {T, V}
  return PseudoBooleanModel{T, V}(model.variables, Dict(vars => -coeff for (vars, coeff) in model.terms), -model.constant)
end

Base.:-(model::PseudoBooleanModel) = negate(model)

function isquadratic(model::PseudoBooleanModel)
  return all(vars -> length(vars) <= 2, keys(model.terms))
end
