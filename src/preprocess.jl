function issquare(a :: AbstractArray)
  return allequal(size(a))
end

function edge_key(i::Integer, j::Integer)
  return i < j ? (Int(i), Int(j)) : (Int(j), Int(i))
end

function qmatrix_adjacency(Q::AbstractMatrix, cutoff)
  n = size(Q, 1)
  adjacency = [Int[] for _ in 1:n]
  weights = Dict{Tuple{Int, Int}, Float64}()
  candidate_edges = Set{Tuple{Int, Int}}()

  for ci in findall(!iszero, Q)
    i, j = Tuple(ci)
    i == j && continue
    push!(candidate_edges, edge_key(i, j))
  end

  for (i, j) in sort!(collect(candidate_edges))
    coeff = Q[i, j] + Q[j, i]

    if abs(coeff) > cutoff
      push!(adjacency[i], j)
      push!(adjacency[j], i)
      weights[(i, j)] = Float64(abs(coeff))
    end
  end

  return adjacency, weights
end

function reverse_cuthill_mckee(adjacency, weights)
  n = length(adjacency)
  degrees = length.(adjacency)
  weighted_degrees = [
    sum(weights[edge_key(i, j)] for j in adjacency[i]; init = 0.0)
    for i in 1:n
  ]

  visited = falses(n)
  order = Int[]

  while true
    start = nothing
    best = nothing

    for i in 1:n
      if !visited[i] && degrees[i] > 0
        candidate = (degrees[i], -weighted_degrees[i], i)
        if isnothing(best) || candidate < best
          start = i
          best = candidate
        end
      end
    end

    isnothing(start) && break

    queue = [start]
    visited[start] = true
    head = 1

    while head <= length(queue)
      i = queue[head]
      head += 1
      push!(order, i)

      neighbors = [j for j in adjacency[i] if !visited[j]]
      sort!(neighbors; by = j -> (degrees[j], -weights[edge_key(i, j)], j))

      for j in neighbors
        visited[j] = true
        push!(queue, j)
      end
    end
  end

  isempty(order) && return collect(1:n)

  isolated = [i for i in 1:n if degrees[i] == 0]
  return vcat(reverse(order), isolated)
end

"""
    qmatrix_permutation(Q; cutoff=0)

Return a deterministic permutation that places coupled QUBO variables closer
together in the one-dimensional MPS ordering.

`cutoff` controls which quadratic couplings are included in the ordering graph:
an undirected edge between variables `i` and `j` is used when
`abs(Q[i, j] + Q[j, i]) > cutoff`. The default `cutoff = 0` preserves every
nonzero coupling for callers that only want the ordering.

The returned permutation maps each tensor site to its original variable index:
entry `k` is the original QUBO variable represented by tensor site `k`.

# Examples

```julia
Q = [0.0 0.0 1.0;
     0.0 0.0 0.0;
     1.0 0.0 0.0]

permutation = qmatrix_permutation(Q)
Q[permutation, permutation]
```
"""
function qmatrix_permutation(Q::AbstractMatrix; cutoff = 0)
  issquare(Q) || throw(DimensionMismatch("Q must be square. Encountered dimensions $(size(Q))."))

  adjacency, weights = qmatrix_adjacency(Q, cutoff)
  return reverse_cuthill_mckee(adjacency, weights)
end

function is_identity_permutation(permutation)
  return all(i -> permutation[i] == i, eachindex(permutation))
end

"""
    preprocess_qubo(Q, l, cutoff)

Permute QUBO variables before Hamiltonian construction so coupled variables are
closer in the one-dimensional tensor order.

Returns `(Qp, lp, permutation)`, where `Qp` and `lp` are reordered for tensor-site
order and `permutation[k]` is the caller's original variable index represented
by tensor site `k`.

Standalone [`qmatrix_permutation`](@ref) uses `cutoff = 0` by default so direct
callers keep every nonzero coupling. The solve path passes the solver cutoff,
which defaults to `1e-8`, so numerically tiny couplings are ignored during
preprocessing.
"""
function preprocess_qubo(Q, l, cutoff)
  permutation = qmatrix_permutation(Q; cutoff)
  is_identity_permutation(permutation) && return Q, l, permutation

  return Q[permutation, permutation],
         isnothing(l) ? nothing : l[permutation],
         permutation
end
