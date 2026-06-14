import SparseArrays: dropzeros!, sparse

function _conversion_type(Q::AbstractMatrix, l, c)
  T = promote_type(eltype(Q), isnothing(l) ? eltype(Q) : eltype(l), typeof(c))
  T <: Real || throw(ArgumentError("QUBO/Ising conversion only supports real coefficients. Got coefficient type $T."))
  return typeof(one(T) / 2)
end

function _check_qubo_dimensions(Q::AbstractMatrix, l)
  issquare(Q) || throw(DimensionMismatch("The QUBO matrix must be square. Encountered dimensions $(size(Q))."))
  if !isnothing(l) && length(l) != size(Q, 1)
    throw(DimensionMismatch("The QUBO linear vector length must match the matrix size. Encountered matrix size $(size(Q)) and vector length $(length(l))."))
  end
end

function _check_ising_dimensions(J::AbstractMatrix, h::AbstractVector)
  issquare(J) || throw(DimensionMismatch("The Ising coupling matrix must be square. Encountered dimensions $(size(J))."))
  size(J, 1) == length(h) || throw(DimensionMismatch("The Ising field vector length must match the coupling matrix size. Encountered dimensions $(size(J)) and length $(length(h))."))
end

function _check_spin_convention(convention)
  convention === :spin || throw(ArgumentError("Only the `:spin` convention is supported. Use x = (s + 1) / 2 and s = 2x - 1."))
end

function _checked_bool_state(x::AbstractVector)
  return map(x) do xi
    if iszero(xi)
      0
    elseif isone(xi)
      1
    else
      throw(ArgumentError("Boolean vectors must contain only 0/1 values. Encountered $xi."))
    end
  end
end

function _checked_spin_state(s::AbstractVector)
  return map(s) do si
    if si == -1
      -1
    elseif si == 1
      1
    else
      throw(ArgumentError("Spin vectors must contain only -1/+1 values. Encountered $si."))
    end
  end
end

function _drop_form_zeros!(form::QUBOTools.AbstractForm)
  _, l, Q, _, _, _, _ = form
  dropzeros!(l)
  dropzeros!(Q)
  return form
end

function _qubo_form(Q::AbstractMatrix, l::Union{Nothing, AbstractVector}, c::Real)
  _check_qubo_dimensions(Q, l)

  n = size(Q, 1)
  T = _conversion_type(Q, l, c)
  L = isnothing(l) ? zeros(T, n) : collect(T, l)

  return _drop_form_zeros!(
    QUBOTools.SparseForm{T}(n, L, sparse(T.(Q)), one(T), T(c); sense = :min, domain = :bool),
  )
end

function _ising_form(J::AbstractMatrix, h::AbstractVector, offset::Real)
  _check_ising_dimensions(J, h)

  n = size(J, 1)
  T = _conversion_type(J, h, offset)

  return _drop_form_zeros!(
    QUBOTools.SparseForm{T}(n, collect(T, h), sparse(T.(J)), one(T), T(offset); sense = :min, domain = :spin),
  )
end

function _check_form_domain(form::QUBOTools.AbstractForm, domain, label::AbstractString)
  QUBOTools.domain(form) === domain || throw(ArgumentError("$label conversion expected a QUBOTools form in domain $domain. Encountered $(QUBOTools.domain(form))."))
end

"""
    bool_to_spin(x)

Convert a Boolean bit vector `x_i in {0, 1}` to Ising spins
`s_i in {-1, +1}` using QUBOTools' `BoolDomain => SpinDomain` cast.
"""
function bool_to_spin(x::AbstractVector)
  return QUBOTools.cast(QUBOTools.BoolDomain => QUBOTools.SpinDomain, _checked_bool_state(x))
end

"""
    spin_to_bool(s)

Convert an Ising spin vector `s_i in {-1, +1}` to Boolean bits
`x_i in {0, 1}` using QUBOTools' `SpinDomain => BoolDomain` cast.
"""
function spin_to_bool(s::AbstractVector)
  return QUBOTools.cast(QUBOTools.SpinDomain => QUBOTools.BoolDomain, _checked_spin_state(s))
end

"""
    qubo_to_ising(Q[, l[, c]]; convention = :spin)
    qubo_to_ising(form; convention = :spin)

Convert TenSolver's Boolean QUBO objective

```julia
dot(x, Q, x) + dot(l, x) + c
```

with `x_i in {0, 1}` into a sparse QUBOTools form in `SpinDomain`.
For non-symmetric matrices, the QUBOTools form constructor preserves
TenSolver's `dot(x, Q, x)` convention by storing the effective pair
coefficient `Q[i, j] + Q[j, i]` once in the upper triangle.

The Boolean-to-spin conversion introduces halves and quarters. Integer
coefficient inputs therefore return floating-point forms, while rational inputs
preserve exact rational arithmetic.

The returned form includes the constant offset, so

```julia
dot(x, Q, x) + dot(l, x) + c == QUBOTools.value(bool_to_spin(x), qubo_to_ising(Q, l, c))
```

for every Boolean vector `x`.
"""
function qubo_to_ising(Q::AbstractMatrix, l::Union{Nothing, AbstractVector}=nothing, c::Real=0; convention::Symbol=:spin)
  _check_spin_convention(convention)
  return qubo_to_ising(_qubo_form(Q, l, c); convention)
end

function qubo_to_ising(form::QUBOTools.AbstractForm; convention::Symbol=:spin)
  _check_spin_convention(convention)
  _check_form_domain(form, QUBOTools.BoolDomain, "QUBO-to-Ising")
  return _drop_form_zeros!(QUBOTools.cast(QUBOTools.SpinDomain, form))
end

"""
    ising_to_qubo(form)
    ising_to_qubo(J, h[, offset])

Convert a QUBOTools spin-domain Ising form back to a sparse Boolean-domain
QUBOTools form. The matrix/vector method first builds a QUBOTools spin form
from

```julia
dot(s, J, s) + dot(h, s) + offset
```

with `s_i in {-1, +1}`. QUBOTools folds diagonal quadratic spin terms into the
constant offset and stores each off-diagonal unordered pair once in the upper
triangle.

As with [`qubo_to_ising`](@ref), integer coefficient inputs return
floating-point forms when the conversion introduces fractional coefficients,
while rational inputs preserve exact rational arithmetic.
"""
function ising_to_qubo(form::QUBOTools.AbstractForm)
  _check_form_domain(form, QUBOTools.SpinDomain, "Ising-to-QUBO")
  return _drop_form_zeros!(QUBOTools.cast(QUBOTools.BoolDomain, form))
end

function ising_to_qubo(J::AbstractMatrix, h::AbstractVector, offset::Real=0)
  return ising_to_qubo(_ising_form(J, h, offset))
end
