module TenSolverSpinGlassPEPSExt

import MultivariatePolynomials:
  AbstractPolynomial, coefficient, effective_variables, isconstant, maxdegree, powers, terms
import SparseArrays: findnz, sparse

import TenSolver
import TenSolver: KingGrid, PEPSBackend, PEPSSolution, SquareGrid

import SpinGlassEngine
import SpinGlassNetworks
import SpinGlassTensors

function peps_options(;
  maxdim::Integer = 16,
  iterations::Integer = 1,
  device::Function = TenSolver.cpu,
  beta::Real = 2.0,
  max_states::Integer = 2^8,
  cutoff_prob::Real = 1e-4,
  contraction::Symbol = :auto,
  graduate_truncation::Bool = true,
  transformations = :all,
  local_dimension::Union{Nothing,Integer} = nothing,
  no_cache::Bool = false,
)
  if !(beta > 0 && isfinite(beta))
    throw(ArgumentError("PEPS solves require finite beta > 0. Got $beta."),)
  end
  if !(maxdim >= 1)
    throw(ArgumentError("PEPS solves require maxdim >= 1. Got $maxdim."),)
  end
  if !(max_states >= 1)
    throw(ArgumentError("PEPS solves require max_states >= 1. Got $max_states."),)
  end
  if !(cutoff_prob >= 0)
    throw(ArgumentError("PEPS solves require cutoff_prob >= 0. Got $cutoff_prob.",),)
  end
  if !(iterations >= 1)
    throw(ArgumentError("PEPS solves require iterations >= 1. Got $iterations."),)
  end
  if !(contraction in (:auto, :svd, :svd_truncate, :zipper))
    throw(ArgumentError("Unsupported PEPS contraction $(repr(contraction)). Use :auto, " *
                        ":svd, :svd_truncate, or :zipper.",),)
  end
  if !isnothing(local_dimension)
    local_dimension >= 1 ||
      throw(ArgumentError("PEPS solves require local_dimension >= 1 when provided. " *
                          "Got $local_dimension.",),)
  end

  return (;
    maxdim = Int(maxdim),
    iterations = Int(iterations),
    onGPU = !(device === TenSolver.cpu),
    beta = Float64(beta),
    max_states = Int(max_states),
    cutoff_prob = Float64(cutoff_prob),
    contraction,
    graduate_truncation,
    transformations,
    local_dimension = isnothing(local_dimension) ? nothing : Int(local_dimension),
    no_cache,
  )
end

function check_spin_domain(domain)
  return domain == [-1, 1] ||
         throw(ArgumentError("PEPSBackend requires domain = [-1, 1]. Got $(repr(domain)).",),)
end

peps_float_type(::Type{T}) where {T} = float(T)

function ising_instance(J::AbstractMatrix{T}, h::AbstractVector{T}) where {T}
  S = peps_float_type(T)
  instance = Dict{Tuple{Int,Int},S}()

  for i in eachindex(h)
    instance[(i, i)] = S(h[i])
  end

  rows, cols, values = findnz(sparse(J))
  for k in eachindex(values)
    i = rows[k]
    j = cols[k]
    if i == j
      continue
    end

    edge = minmax(i, j)
    instance[edge] = get(instance, edge, zero(S)) + S(values[k])
  end

  return instance
end

function check_topology_size(topology, h::AbstractVector)
  expected = TenSolver.topology_size(topology)
  actual = length(h)
  return actual == expected ||
         throw(DimensionMismatch("PEPS topology $(repr(topology)) expects $expected spins, but the " *
                                 "Ising model has $actual spins.",),)
end

edge_supported(::SquareGrid, a::Tuple, b::Tuple) = abs(a[1] - b[1]) + abs(a[2] - b[2]) == 1
edge_supported(::KingGrid, a::Tuple, b::Tuple) = maximum(abs.(a .- b)) == 1

function check_layout_edges(topology, J::AbstractMatrix, lattice)
  rows, cols, values = findnz(sparse(J))
  for k in eachindex(values)
    i = rows[k]
    j = cols[k]
    if i != j && !iszero(values[k])
      source = lattice[i]
      target = lattice[j]
      if source != target && !edge_supported(topology, source, target)
        throw(ArgumentError("Ising coupling ($i, $j) is not compatible with " *
                            "$(repr(topology)). Use a compatible structured topology.",),)
      end
    end
  end
end

function build_potts_hamiltonian(local_dimension, instance, lattice)
  if isnothing(local_dimension)
    return SpinGlassNetworks.potts_hamiltonian(
      instance;
      spectrum = SpinGlassNetworks.full_spectrum,
      cluster_assignment_rule = lattice,
    )
  end

  return SpinGlassNetworks.potts_hamiltonian(
    instance,
    local_dimension;
    spectrum = SpinGlassNetworks.full_spectrum,
    cluster_assignment_rule = lattice,
  )
end

function resolve_transformations(transformations)
  if transformations === :all
    return SpinGlassEngine.all_lattice_transformations
  end
  if transformations === :identity
    return (SpinGlassEngine.rotation(0),)
  end
  if transformations isa Symbol
    throw(ArgumentError("Unsupported PEPS transformations $(repr(transformations)). Use " *
                        ":all, :identity, a transformation, or a collection of transformations.",),)
  end
  if transformations isa Tuple
    return transformations
  end
  if transformations isa AbstractVector
    return Tuple(transformations)
  end
  return (transformations,)
end

function contraction_strategy(contraction::Symbol)
  if contraction in (:auto, :svd, :svd_truncate)
    return SpinGlassEngine.SVDTruncate
  end
  if contraction === :zipper
    return SpinGlassEngine.Zipper
  end
  return throw(ArgumentError("Unsupported PEPS contraction $(repr(contraction)).",),)
end

function peps_network(topology::SquareGrid, potts_h, transform, ::Type{T}) where {T}
  return SpinGlassEngine.PEPSNetwork{
    SpinGlassEngine.SquareSingleNode{SpinGlassEngine.GaugesEnergy},
    SpinGlassEngine.Dense,
    T,
  }(
    topology.m,
    topology.n,
    potts_h,
    transform,
  )
end

function peps_network(topology::KingGrid, potts_h, transform, ::Type{T}) where {T}
  return SpinGlassEngine.PEPSNetwork{
    SpinGlassEngine.KingSingleNode{SpinGlassEngine.GaugesEnergy},
    SpinGlassEngine.Dense,
    T,
  }(
    topology.m,
    topology.n,
    potts_h,
    transform,
  )
end

function solve_transformation(
  topology,
  potts_h,
  transform,
  ::Type{T},
  parameters,
  search_parameters,
  strategy,
  options,
) where {T}
  try
    network = peps_network(topology, potts_h, transform, T)
    contractor = SpinGlassEngine.MpsContractor(
      strategy,
      network,
      parameters;
      onGPU = options.onGPU,
      beta = T(options.beta),
      graduate_truncation = options.graduate_truncation,
    )
    merge_strategy = SpinGlassEngine.merge_branches(contractor; merge_prob = :none)
    solution, info = SpinGlassEngine.low_energy_spectrum(
      contractor,
      search_parameters,
      merge_strategy;
      no_cache = options.no_cache,
    )

    return (; solution, info)
  finally
    SpinGlassEngine.clear_memoize_cache()
  end
end

function decoded_records(J, h, offset, potts_h, solution, transform)
  records = NamedTuple[]
  for i in eachindex(solution.states)
    decoded = SpinGlassNetworks.decode_potts_hamiltonian_state(potts_h, solution.states[i])
    spins = [Int(decoded[j]) for j in eachindex(h)]
    push!(
      records,
      (;
        state = spins,
        energy = TenSolver.ising_energy(J, h, offset, spins),
        probability = solution.probabilities[i],
        transformation = transform,
        raw_energy = solution.energies[i],
      ),
    )
  end
  return records
end

function deduplicated_records(records)
  sort!(records; by = record -> (record.energy, -record.probability))

  deduplicated = NamedTuple[]
  positions = Dict{Tuple{Vararg{Int}},Int}()
  for record in records
    state = Tuple(record.state)
    index = get(positions, state, nothing)
    if isnothing(index)
      push!(deduplicated, record)
      positions[state] = lastindex(deduplicated)
    else
      existing = deduplicated[index]
      deduplicated[index] =
        (; existing..., probability = existing.probability + record.probability)
    end
  end

  return deduplicated
end

function peps_metadata(backend::PEPSBackend, records, raw_results)
  best = first(records)
  selected = raw_results[best.transformation]
  return Dict{String,Any}(
    "backend" => "SpinGlassPEPS",
    "topology" => TenSolver.topology_name(backend.topology),
    "topology_size" => TenSolver.topology_tuple(backend.topology),
    "transformations_tried" => collect(string.(keys(raw_results))),
    "selected_transformation" => string(best.transformation),
    "spin_glass_energies" => collect(selected.solution.energies),
    "spin_glass_probabilities" => collect(selected.solution.probabilities),
    "largest_discarded_probability" => selected.solution.largest_discarded_probability,
    "raw" => raw_results,
  )
end

function quadratic_form(p::AbstractPolynomial{T}) where {T}
  if !(maxdegree(p) <= 2)
    throw(ArgumentError("PEPSBackend supports polynomial inputs only when they are quadratic.",),)
  end

  variables = effective_variables(p)
  indices = Dict(variable => i for (i, variable) in enumerate(variables))
  Q = zeros(T, length(variables), length(variables))
  h = zeros(T, length(variables))
  offset = zero(T)

  for term in terms(p)
    value = coefficient(term)
    if isconstant(term)
      offset += value
      continue
    end

    term_powers = collect(powers(term))
    degree = sum(last, term_powers)
    if degree == 1
      variable, _ = only(term_powers)
      h[indices[variable]] += value
    elseif degree == 2 && length(term_powers) == 1
      variable, _ = only(term_powers)
      Q[indices[variable], indices[variable]] += value
    elseif degree == 2 && length(term_powers) == 2
      first_power, second_power = term_powers
      first_variable, _ = first_power
      second_variable, _ = second_power
      Q[indices[first_variable], indices[second_variable]] += value
    else
      throw(ArgumentError("PEPSBackend supports polynomial inputs only when they are quadratic.",),)
    end
  end

  return Q, h, offset
end

function TenSolver.minimize(
  backend::PEPSBackend,
  p::AbstractPolynomial;
  domain::AbstractVector,
  kwargs...,
)
  check_spin_domain(domain)
  Q, h, offset = quadratic_form(p)
  return TenSolver.minimize(backend, Q, h, offset; domain, kwargs...)
end

function TenSolver.minimize(
  backend::PEPSBackend,
  J::AbstractMatrix{T},
  h::AbstractVector{T},
  offset::T;
  domain::AbstractVector,
  cutoff = nothing,
  preprocess::Bool = false,
  verbosity::Integer = 1,
  kwargs...,
) where {T<:Real}
  check_spin_domain(domain)
  if preprocess
    throw(ArgumentError("PEPSBackend does not support preprocess=true because the topology " *
                        "fixes the variable order.",),)
  end
  if !(size(J, 1) == size(J, 2))
    throw(DimensionMismatch("The Ising coupling matrix must be square. Encountered dimensions " *
                            "$(size(J)).",),)
  end
  if !(size(J, 1) == length(h))
    throw(DimensionMismatch("The Ising field vector length must match the coupling matrix size. " *
                            "Encountered dimensions $(size(J)) and length $(length(h)).",),)
  end

  options = peps_options(; kwargs...)
  check_topology_size(backend.topology, h)

  S = peps_float_type(T)
  instance = ising_instance(J, h)
  ising_graph = SpinGlassNetworks.ising_graph(S, instance)
  lattice =
    SpinGlassNetworks.super_square_lattice(TenSolver.topology_tuple(backend.topology),)
  check_layout_edges(backend.topology, J, lattice)
  potts_h = build_potts_hamiltonian(options.local_dimension, ising_graph, lattice)
  parameters = SpinGlassEngine.MpsParameters{S}(;
    bond_dim = options.maxdim,
    num_sweeps = options.iterations,
  )
  search_parameters = SpinGlassEngine.SearchParameters(;
    max_states = options.max_states,
    cutoff_prob = options.cutoff_prob,
  )
  strategy = contraction_strategy(options.contraction)

  records = NamedTuple[]
  raw_results = Dict{Any,Any}()
  for transform in resolve_transformations(options.transformations)
    result = solve_transformation(
      backend.topology,
      potts_h,
      transform,
      S,
      parameters,
      search_parameters,
      strategy,
      options,
    )
    raw_results[transform] = result
    append!(records, decoded_records(J, h, offset, potts_h, result.solution, transform))
  end

  if isempty(records)
    throw(ArgumentError("SpinGlassPEPS did not return any states."),)
  end

  records = deduplicated_records(records)
  states = [record.state for record in records]
  energies = S[record.energy for record in records]
  probabilities = S[record.probability for record in records]
  metadata = peps_metadata(backend, records, raw_results)

  if verbosity > 0
    @info(
      "SpinGlassPEPS backend finished",
      energy = first(energies),
      states = length(states),
    )
  end

  return first(energies), PEPSSolution{S}(states, energies, probabilities, metadata)
end

end # module
