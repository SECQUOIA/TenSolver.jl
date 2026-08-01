#=
  Utility functions for testing.
  Here we have slower but exact solvers.
=#

"""
    brute_force(f, n[, constraints]; domain = 0:1)

Return the minimum objective value and one minimizer found by exhaustive
enumeration. Use only for small test problems because the work grows as
`length(domain)^n`.
"""
function brute_force(obj, n, constraints = AbstractConstraint[]; domain = 0:1)
  best = +Inf
  solution = Vector{Float64}[]

  for bits in Iterators.product(fill(domain, n)...)
    x = collect(bits)

    if is_feasible(x, constraints)
      value = obj(x)
      if value < best
        best = value
        solution = x
      end
    end
  end

  isempty(solution) && throw(ArgumentError("no feasible bitstring"))
  return best, solution
end

function mpo_matrix_element(H, sites, bra_bits, ket_bits)
  bra = ITensorMPS.MPS(sites, string.(bra_bits))
  ket = ITensorMPS.MPS(sites, string.(ket_bits))
  return real(ITensors.inner(bra', H, ket))
end

function mps_amplitude(psi, sites, bits)
  basis = ITensorMPS.MPS(sites, string.(bits))
  return real(ITensors.inner(basis, psi))
end

function randpoly(x, maxdegree)
  dim = length(x)
  mkarray(i) = randn(Iterators.repeated(dim, i)...)
  form(a, x) = sum(a[t] * prod(x[i] for i in Tuple(t)) for t in CartesianIndices(a))

  return sum(form(mkarray(i), x) for i in 1:maxdegree) + randn()
end

function bandwidth(Q)
  bw = 0
  for i in axes(Q, 1), j in (i + 1):last(axes(Q, 2))
    if abs(Q[i, j] + Q[j, i]) > 0
      bw = max(bw, j - i)
    end
  end
  return bw
end
