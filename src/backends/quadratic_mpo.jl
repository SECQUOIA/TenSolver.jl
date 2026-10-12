import SparseArrays: nnz, nonzeros, nzrange, rowvals

function tensorize(
  Q::AbstractMatrix{T};
  cutoff = zero(T),
  domain::Domains,
) where {T<:Union{Float32,Float64}}
  return tensorize(Q, zeros(T, size(Q, 1)); cutoff, domain)
end

function tensorize(
  Q::AbstractMatrix{T},
  l::AbstractVector{T};
  cutoff = zero(T),
  domain::Domains,
) where {T<:Union{Float32,Float64}}
  n = LinearAlgebra.checksquare(Q)
  if length(l) != n || length(domain) != n
    throw(DimensionMismatch("The matrix, linear term, and domains must have the same size."))
  end
  Base.require_one_based_indexing(Q, l)

  # Combine commuting terms before filtering, as in the generic OpSum path.
  sparse_Q = sparse(Q)
  J = triu(sparse_Q + transpose(sparse_Q), 1)
  for k in eachindex(nonzeros(J))
    if !(abs(nonzeros(J)[k]) > cutoff)
      nonzeros(J)[k] = zero(T)
    end
  end
  dropzeros!(J)
  linear = [abs(v) > cutoff ? v : zero(T) for v in l]
  diagonal = [abs(Q[i, i]) > cutoff ? Q[i, i] : zero(T) for i in 1:n]
  sites = [ITensors.siteind("Qudit"; dim = length(domain[i])) for i in 1:n]
  if iszero(nnz(J)) && all(iszero, linear) && all(iszero, diagonal)
    return MPO(T, sites)
  end

  # Track the active rows and columns of the coupling block at each cut.
  # Banded/sparse objectives then factor only their crossing interactions.
  Jt = sparse(transpose(J))
  remaining = [length(nzrange(Jt, i)) for i in 1:n]
  active_left = BitSet()
  active_right = BitSet()
  previous_right = Int[]
  previous_basis = zeros(Float64, 0, 0)
  left_link = ITensors.Index(2, "Link,l=0")
  H = MPO(n)

  for i in 1:n
    for k in nzrange(J, i)
      row = rowvals(J)[k]
      remaining[row] -= 1
      if iszero(remaining[row])
        delete!(active_left, row)
      end
    end
    if remaining[i] > 0
      push!(active_left, i)
    end
    delete!(active_right, i)
    for k in nzrange(Jt, i)
      push!(active_right, rowvals(Jt)[k])
    end
    left = collect(active_left)
    right = collect(active_right)

    if isempty(right)
      basis = zeros(Float64, 0, 0)
    else
      # OpSum also factors real coefficients in Float64 before converting
      # the assembled MPO to T. Keep its relative discarded-weight cutoff.
      F = svd(Matrix{Float64}(J[left, right]))
      basis = Matrix(F.V[:, 1:quadratic_mpo_rank(F.S)])
    end

    closes = !isempty(previous_right) && first(previous_right) == i
    first_continuing = closes ? 2 : 1
    continuing = previous_right[first_continuing:end]
    positions = searchsortedfirst.(Ref(right), continuing)
    carry = transpose(previous_basis[first_continuing:end, :]) * basis[positions, :]
    # OpSum skips individual starting/onsite coefficients at machine epsilon
    # during assembly, even when the input coefficient cutoff is smaller.
    start_coefficients = Vector{Float64}(J[i, right])
    for k in eachindex(start_coefficients)
      if abs(start_coefficients[k]) <= eps(Float64)
        start_coefficients[k] = 0
      end
    end
    starts = transpose(start_coefficients) * basis
    closing = closes ? previous_basis[1, :] : zeros(size(previous_basis, 2))

    right_link = ITensors.Index(size(basis, 2) + 2, "Link,l=$i")
    H[i] = quadratic_mpo_tensor(
      T,
      sites[i],
      left_link,
      right_link,
      domain[i],
      linear[i],
      diagonal[i],
      closing,
      carry,
      starts,
    )
    if i == 1
      H[i] *= ITensors.onehot(T, left_link => ITensors.dim(left_link))
    end
    if i == n
      H[i] *= ITensors.onehot(T, right_link => 1)
    end
    previous_right = right
    previous_basis = basis
    left_link = right_link
  end
  return H
end

# Match OpSum's default relative discarded squared-singular-value weight,
# retaining at least one direction for a nonempty coupling block. This is
# independent of the user's coefficient filter and DMRG truncation cutoff.
function quadratic_mpo_rank(singular_values)
  weights = abs2.(singular_values)
  budget = 1e-15 * sum(weights)
  discarded = zero(eltype(weights))
  retained = length(weights)
  while retained > 1 && discarded + weights[retained] <= budget
    discarded += weights[retained]
    retained -= 1
  end
  return retained
end

# The first channel carries completed terms; the last carries the identity
# before a term starts. Interior channels carry the crossing quadratic terms.
function quadratic_mpo_tensor(
  ::Type{T},
  site,
  left,
  right,
  domain,
  linear,
  diagonal,
  closing,
  carry,
  starts,
) where {T}
  dl, dr, d = ITensors.dim(left), ITensors.dim(right), length(domain)
  data = zeros(T, dl, dr, d, d)
  for (k, x) in enumerate(domain)
    data[1, 1, k, k] = one(T)
    data[end, end, k, k] = one(T)
    onsite = 0.0
    if abs(linear) > eps(Float64)
      onsite += Float64(linear) * x
    end
    if abs(diagonal) > eps(Float64)
      onsite += Float64(diagonal) * x^2
    end
    data[end, 1, k, k] = onsite
    for r in eachindex(closing)
      data[1 + r, 1, k, k] = closing[r] * x
    end
    for c in axes(carry, 2), r in axes(carry, 1)
      data[1 + r, 1 + c, k, k] = carry[r, c]
    end
    for c in eachindex(starts)
      data[end, 1 + c, k, k] = starts[c] * x
    end
  end
  return ITensors.ITensor(data, left, right, site', site)
end
