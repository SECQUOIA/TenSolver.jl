function qmatrix_adjacency(Q::AbstractMatrix, cutoff)
  n = size(Q, 1)
  adjacency = [Int[] for _ in 1:n]
  weights = Dict{Tuple{Int, Int}, Float64}()

  for i in 1:n-1, j in i+1:n
    coeff = abs(Q[i, j] + Q[j, i])

    if coeff > cutoff
      push!(adjacency[i], j)
      push!(adjacency[j], i)
      weights[(i, j)] = Float64(coeff)
    end
  end

  return adjacency, weights
end

function breadth_first_search(start, next_nodes, visited)
  frontier = [start]
  visited[start] = true
  order = Int[]

  while !isempty(frontier)
    append!(order, frontier)

    next_frontier = Int[]
    for i in frontier, j in next_nodes(i)
      visited[j] = true
      push!(next_frontier, j)
    end

    unique!(next_frontier)
    frontier = next_frontier
  end

  return order
end

function reverse_cuthill_mckee(adjacency, weights)
  degrees = length.(adjacency)

  edge_key(i, j) = (min(i, j), max(i, j))
  weighted_degrees = map(eachindex(adjacency)) do i
    sum(weights[edge_key(i, j)] for j in adjacency[i]; init = 0.0)
  end
  score(i) = (degrees[i], -weighted_degrees[i], i)
  neighbor_score(i, j) = (degrees[j], -weights[edge_key(i, j)], j)

  visited = falses(length(adjacency))
  order = Int[]

  candidates = findall(!iszero, degrees)
  while !isempty(candidates)
    start = argmin(score, candidates)

    neighbours(i) = sort(
      filter(j -> !visited[j], adjacency[i]);
      by = j -> neighbor_score(i, j),
    )

    append!(order, breadth_first_search(start, neighbours, visited))
    candidates = filter(i -> !visited[i], candidates)
  end

  return vcat(reverse(order), findall(iszero, degrees))
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

# output

3×3 Matrix{Float64}:
 0.0  1.0  0.0
 1.0  0.0  0.0
 0.0  0.0  0.0
```
"""
function qmatrix_permutation(Q::AbstractMatrix; cutoff)
  LinearAlgebra.checksquare(Q)

  adjacency, weights = qmatrix_adjacency(Q, cutoff)
  return reverse_cuthill_mckee(adjacency, weights)
end

"""
    preprocess_model(Q, l, c; domain, constraints, cutoff)

Permute variables before Hamiltonian construction so coupled variables are
closer in the one-dimensional tensor order.
"""
function preprocess_model(Q, l, c; domain, constraints, cutoff)
  permutation = qmatrix_permutation(Q; cutoff)

  Qp          = Q[permutation, permutation]
  lp          = l[permutation]
  constraints = eltype(constraints)[permute(c, permutation) for c in constraints]
  domain      = permute(domain, permutation)

  return Qp, lp, c, domain, constraints, permutation
end
