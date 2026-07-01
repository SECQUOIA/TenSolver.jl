# Projection-MPO construction adapted from the CoTenN constraint projection
# design in Sharma, Ritvik, Cheng Peng, Siddharth Dangwal, and Sara Achour,
# "CoTenN: Constrained Optimization with Tensor Networks," PLDI 2026.
#
# Note on ITensors.jl / ITensorMPS.jl helpers: the on-site identity is built with
# `ITensors.delta` rather than a hand-rolled Kronecker tensor. The higher-level
# `OpSum`/`MPO(::OpSum, sites)` (AutoMPO) machinery was considered for the full
# assembly but is intentionally not used here: it heuristically compresses the
# operator and does not guarantee the exact, uncompressed one-path-per-feasible-
# assignment structure this module needs (bond dimension = number of feasible
# assignments). Bounding/compressing that bond dimension is deferred to #58, where
# projections meet solves and the downstream cost is measurable. The remaining
# path-threaded tensors have no direct high-level equivalent, so they are assembled
# from sparse nonzeros via `itensor_from_nonzeros`.

"""
    SparseTensorEntry{T}

One nonzero term in the sparse representation used to assemble a projection
MPO. `coordinates` maps 1-based register sites to 1-based local basis states;
sites omitted from the dictionary are unconstrained by this entry. `value` is
the coefficient carried by that partial assignment.
"""
struct SparseTensorEntry{T}
  coordinates::Dict{Int,Int}
  value::T
end

"""
    itensor_from_nonzeros([T], inds, nonzeros)

Build an `ITensor` over `inds` from sparse `(coordinate, value)` entries.
Coordinates are 1-based tuples matching `inds`.
"""
function itensor_from_nonzeros(::Type{T}, inds, nonzeros) where {T}
  ind_tuple = Tuple(inds)
  tensor = ITensors.ITensor(T, ind_tuple...)

  for (coordinate, value) in nonzeros
    coordinate_tuple = Tuple(coordinate)
    length(coordinate_tuple) == length(ind_tuple) ||
      throw(DimensionMismatch("nonzero coordinate length must match tensor order"))

    selector = map(i -> ind_tuple[i] => coordinate_tuple[i], eachindex(ind_tuple))
    tensor[selector...] = tensor[selector...] + convert(T, value)
  end

  return tensor
end

itensor_from_nonzeros(inds, nonzeros) = itensor_from_nonzeros(Float64, inds, nonzeros)

# Diagonal identity over a single physical site (`sum_s |s><s|`). This is exactly
# what `ITensors.delta` builds, so we reuse it instead of assembling by hand.
function identity_tensor(::Type{T}, site) where {T}
  return ITensors.delta(T, site, ITensors.prime(site))
end

function pass_through_tensor(::Type{T}, site, left_link, right_link) where {T}
  if isnothing(left_link) && isnothing(right_link)
    return identity_tensor(T, site)
  end

  inds = mpo_tensor_indices(site, left_link, right_link)
  num_paths = path_count(left_link, right_link)
  nonzeros = Tuple{Vector{Int},T}[]

  for path in 1:num_paths, state in 1:ITensors.dim(site)
    push!(nonzeros, (mpo_coordinate(state, path, left_link, right_link), one(T)))
  end

  return itensor_from_nonzeros(T, inds, nonzeros)
end

function path_count(left_link, right_link)
  !isnothing(left_link) && return ITensors.dim(left_link)
  !isnothing(right_link) && return ITensors.dim(right_link)

  return 1
end

function mpo_tensor_indices(site, left_link, right_link)
  inds = []
  !isnothing(left_link) && push!(inds, left_link)
  push!(inds, site, ITensors.prime(site))
  !isnothing(right_link) && push!(inds, right_link)

  return Tuple(inds)
end

function mpo_coordinate(state::Integer, path::Integer, left_link, right_link)
  coordinate = Int[]
  !isnothing(left_link) && push!(coordinate, path)
  push!(coordinate, state, state)
  !isnothing(right_link) && push!(coordinate, path)

  return coordinate
end

function tensor_to_mpo(::Type{T}, entries, sites) where {T}
  site_vec = collect(sites)
  isempty(site_vec) && throw(ArgumentError("sites must not be empty"))
  all(site -> ITensors.dim(site) == 2, site_vec) ||
    throw(ArgumentError("projection MPO sites must be binary Qudit indices"))

  entry_vec = collect(entries)
  validate_sparse_entries(entry_vec, site_vec)

  # Each feasible partial assignment gets one path threaded through the MPO
  # links. Constrained sites write that assignment; unconstrained sites pass the
  # path and physical basis state through unchanged.
  num_paths = max(length(entry_vec), 1)
  links = [
    ITensors.Index(num_paths, "Link,Projection,l=$i")
    for i in 1:(length(site_vec) - 1)
  ]

  # Each entry's `value` is written exactly once, at the first register site the
  # entry constrains. Anchoring on the first *constrained* site (rather than the
  # first register site) keeps the coefficient from being dropped when the leading
  # sites are pass-through, where the tensor is built without consulting `value`.
  # Entries that constrain no site fall back to the first register site.
  value_positions = [
    isempty(entry.coordinates) ? firstindex(site_vec) : minimum(keys(entry.coordinates))
    for entry in entry_vec
  ]

  tensors = Vector{ITensors.ITensor}(undef, length(site_vec))
  for site_position in eachindex(site_vec)
    site = site_vec[site_position]
    left_link = site_position == firstindex(site_vec) ? nothing : links[site_position - 1]
    right_link = site_position == lastindex(site_vec) ? nothing : links[site_position]
    if !isempty(entry_vec) && all(entry -> !haskey(entry.coordinates, site_position), entry_vec)
      tensors[site_position] = pass_through_tensor(T, site, left_link, right_link)
      continue
    end

    inds = mpo_tensor_indices(site, left_link, right_link)
    nonzeros = Tuple{Vector{Int},T}[]

    for (path, entry) in enumerate(entry_vec)
      states = get(entry.coordinates, site_position, nothing)
      states = isnothing(states) ? (1, 2) : (states,)
      coefficient = site_position == value_positions[path] ? entry.value : one(T)

      for state in states
        push!(nonzeros, (mpo_coordinate(state, path, left_link, right_link), coefficient))
      end
    end

    tensors[site_position] = itensor_from_nonzeros(T, inds, nonzeros)
  end

  return ITensorMPS.MPO(tensors)
end

function validate_sparse_entries(entries, sites)
  for entry in entries
    for (site_position, coordinate) in entry.coordinates
      checkbounds(sites, site_position)
      1 <= coordinate <= ITensors.dim(sites[site_position]) ||
        throw(BoundsError(1:ITensors.dim(sites[site_position]), coordinate))
    end
  end

  return nothing
end

# Enumerate feasible assignments on the sites touched by `constraint`. The
# resulting sparse entries intentionally omit untouched sites; `tensor_to_mpo`
# expands those missing coordinates into pass-through tensors.
function projection_entries(::Type{T}, constraint::AbstractConstraint) where {T}
  cs = constraint_sites(constraint)
  assignments = Iterators.product(fill(0:1, length(cs))...)
  entries = SparseTensorEntry{T}[]

  for assignment in assignments
    x = zeros(Int, maximum(cs))
    coordinates = Dict{Int,Int}()

    for (site, bit) in zip(cs, assignment)
      x[site] = bit
      coordinates[site] = bit + 1
    end

    if is_feasible(x, constraint)
      push!(entries, SparseTensorEntry{T}(coordinates, one(T)))
    end
  end

  return entries
end

"""
    projection_mpo([T], constraint, sites)

Build a diagonal projection MPO over the binary Qudit register `sites`.

The diagonal entry is `one(T)` for computational basis states whose bits satisfy
`constraint`, and zero otherwise. Constraint site numbers use the same 1-based
register indexing as `sites`. The construction is exact and uncompressed: each
feasible assignment of the constrained sites is represented as one MPO path.
"""
function projection_mpo(::Type{T}, constraint::AbstractConstraint, sites) where {T}
  cs = constraint_sites(constraint)
  site_vec = collect(sites)

  all(site -> 1 <= site <= length(site_vec), cs) ||
    throw(BoundsError(site_vec, maximum(cs)))

  return tensor_to_mpo(T, projection_entries(T, constraint), site_vec)
end

projection_mpo(constraint::AbstractConstraint, sites) =
  projection_mpo(Float64, constraint, sites)

"""
    projection_mpos([T], constraints, sites)

Build one projection MPO per constraint over the shared binary Qudit register
`sites`.

This is a convenience wrapper around [`projection_mpo`](@ref) that reuses the
same collected site register for every constraint. `T` controls the numeric
element type of the assembled MPO tensors.
"""
function projection_mpos(::Type{T}, constraints::AbstractVector{<:AbstractConstraint}, sites) where {T}
  site_vec = collect(sites)

  return [projection_mpo(T, constraint, site_vec) for constraint in constraints]
end

projection_mpos(constraints::AbstractVector{<:AbstractConstraint}, sites) =
  projection_mpos(Float64, constraints, sites)
