import SparseArrays: SparseMatrixCSC, findnz, sparse

"""
    IsingModel

Pairwise Ising objective with spins `s_i in {-1, +1}`.

The objective value is

```julia
offset + sum(h[i] * s[i] for i in eachindex(s)) +
sum(J[i, j] * s[i] * s[j] for i < j)
```

`J` stores one coupling per unordered pair, conventionally in the upper
triangle, and the diagonal is ignored. This convention avoids double-counting
pair terms and matches the output of [`qubo_to_ising`](@ref).
"""
struct IsingModel{T <: Real}
    J      :: SparseMatrixCSC{T, Int}
    h      :: Vector{T}
    offset :: T
end

function IsingModel(J::AbstractMatrix{<:Real}, h::AbstractVector{<:Real}, offset::Real=0)
  issquare(J) || throw(DimensionMismatch("The Ising coupling matrix must be square. Encountered dimensions $(size(J))."))
  size(J, 1) == length(h) || throw(DimensionMismatch("The Ising field vector length must match the coupling matrix size. Encountered dimensions $(size(J)) and length $(length(h))."))

  T = promote_type(eltype(J), eltype(h), typeof(offset))
  return IsingModel{T}(sparse(T.(J)), collect(T, h), T(offset))
end

function _conversion_type(Q::AbstractMatrix, l, c)
  T = promote_type(eltype(Q), isnothing(l) ? eltype(Q) : eltype(l), typeof(c))
  T <: Real || throw(ArgumentError("QUBO-to-Ising conversion only supports real coefficients. Got coefficient type $T."))
  return typeof(one(T) / 2)
end

function _check_qubo_dimensions(Q::AbstractMatrix, l)
  issquare(Q) || throw(DimensionMismatch("The QUBO matrix must be square. Encountered dimensions $(size(Q))."))
  if !isnothing(l) && length(l) != size(Q, 1)
    throw(DimensionMismatch("The QUBO linear vector length must match the matrix size. Encountered matrix size $(size(Q)) and vector length $(length(l))."))
  end
end

function _check_spin_convention(convention)
  convention === :spin || throw(ArgumentError("Only the `:spin` convention is supported. Use x = (s + 1) / 2 and s = 2x - 1."))
end

function _linear_coefficient(l, i, ::Type{T}) where {T}
  return isnothing(l) ? zero(T) : T(l[i])
end

"""
    bool_to_spin(x)

Convert a Boolean bit vector `x_i in {0, 1}` to Ising spins
`s_i in {-1, +1}` using `s_i = 2x_i - 1`.
"""
function bool_to_spin(x::AbstractVector)
  return map(x) do xi
    if iszero(xi)
      -1
    elseif isone(xi)
      1
    else
      throw(ArgumentError("Boolean vectors must contain only 0/1 values. Encountered $xi."))
    end
  end
end

"""
    spin_to_bool(s)

Convert an Ising spin vector `s_i in {-1, +1}` to Boolean bits
`x_i in {0, 1}` using `x_i = (s_i + 1) / 2`.
"""
function spin_to_bool(s::AbstractVector)
  return map(s) do si
    if si == -1
      0
    elseif si == 1
      1
    else
      throw(ArgumentError("Spin vectors must contain only -1/+1 values. Encountered $si."))
    end
  end
end

"""
    qubo_to_ising(Q[, l[, c]]; convention = :spin)

Convert TenSolver's Boolean QUBO objective

```julia
dot(x, Q, x) + dot(l, x) + c
```

with `x_i in {0, 1}` into an [`IsingModel`](@ref) with spins
`s_i in {-1, +1}`. The returned model includes the constant offset, so

```julia
dot(x, Q, x) + dot(l, x) + c == ising_energy(model, bool_to_spin(x))
```

for every Boolean vector `x`.

Off-diagonal QUBO coefficients follow TenSolver's `dot(x, Q, x)` convention:
for `i < j`, the effective pair coefficient is `Q[i, j] + Q[j, i]`.
The returned Ising coupling matrix stores one coupling per unordered pair in
the upper triangle.
"""
function qubo_to_ising(Q::AbstractMatrix, l::Union{Nothing, AbstractVector}=nothing, c::Real=0; convention::Symbol=:spin)
  _check_spin_convention(convention)
  _check_qubo_dimensions(Q, l)

  n = size(Q, 1)
  T = _conversion_type(Q, l, c)

  rows = Int[]
  cols = Int[]
  vals = T[]
  h = zeros(T, n)
  offset = T(c)

  for i in 1:n
    a = T(Q[i, i]) + _linear_coefficient(l, i, T)
    h[i] += a / 2
    offset += a / 2
  end

  for j in 2:n
    for i in 1:(j - 1)
      b = T(Q[i, j]) + T(Q[j, i])
      if !iszero(b)
        coupling = b / 4
        push!(rows, i)
        push!(cols, j)
        push!(vals, coupling)
        h[i] += coupling
        h[j] += coupling
        offset += coupling
      end
    end
  end

  return IsingModel{T}(sparse(rows, cols, vals, n, n), h, offset)
end

"""
    ising_energy(model, s)

Evaluate an [`IsingModel`](@ref) at a spin vector `s_i in {-1, +1}`.
"""
function ising_energy(model::IsingModel{T}, s::AbstractVector) where {T}
  length(s) == length(model.h) || throw(DimensionMismatch("Spin vector length must match the Ising model size. Encountered length $(length(s)) and model size $(length(model.h))."))
  spin_to_bool(s)

  energy = model.offset + sum(model.h[i] * T(s[i]) for i in eachindex(model.h))
  rows, cols, vals = findnz(model.J)
  for k in eachindex(vals)
    i = rows[k]
    j = cols[k]
    if i < j
      energy += vals[k] * T(s[i]) * T(s[j])
    end
  end

  return energy
end

"""
    ising_to_qubo(model)
    ising_to_qubo(J, h[, offset])

Convert an Ising model back to a TenSolver QUBO representation.

Returns a named tuple `(; Q, l, c)` such that

```julia
ising_energy(model, bool_to_spin(x)) == dot(x, Q, x) + dot(l, x) + c
```

for every Boolean vector `x`. The returned `Q` is sparse, has zero diagonal,
and stores pair coefficients in the upper triangle.
"""
ising_to_qubo(model::IsingModel) = ising_to_qubo(model.J, model.h, model.offset)

function ising_to_qubo(J::AbstractMatrix, h::AbstractVector, offset::Real=0)
  issquare(J) || throw(DimensionMismatch("The Ising coupling matrix must be square. Encountered dimensions $(size(J))."))
  size(J, 1) == length(h) || throw(DimensionMismatch("The Ising field vector length must match the coupling matrix size. Encountered dimensions $(size(J)) and length $(length(h))."))

  n = length(h)
  T = _conversion_type(J, h, offset)
  rows = Int[]
  cols = Int[]
  vals = T[]
  l = [2 * T(h[i]) for i in 1:n]
  c = T(offset) - sum(T, h)

  for j in 2:n
    for i in 1:(j - 1)
      coupling = T(J[i, j]) + T(J[j, i])
      if !iszero(coupling)
        push!(rows, i)
        push!(cols, j)
        push!(vals, 4 * coupling)
        l[i] -= 2 * coupling
        l[j] -= 2 * coupling
        c += coupling
      end
    end
  end

  return (; Q = sparse(rows, cols, vals, n, n), l, c)
end
