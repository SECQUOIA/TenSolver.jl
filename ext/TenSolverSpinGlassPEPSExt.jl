module TenSolverSpinGlassPEPSExt

import SparseArrays: findnz

import TenSolver
import TenSolver:
  IsingModel,
  KingGrid,
  PEPSBackend,
  PEPSSolution,
  SquareGrid,
  ising_energy,
  spin_to_bool

import SpinGlassEngine
import SpinGlassNetworks
import SpinGlassTensors

function _peps_type(::IsingModel{T}) where {T}
  return typeof(float(one(T)))
end

function _ising_instance(model::IsingModel{T}) where {T}
  S = _peps_type(model)
  instance = Dict{Tuple{Int, Int}, S}()

  for i in eachindex(model.h)
    instance[(i, i)] = S(model.h[i])
  end

  rows, cols, vals = findnz(model.J)
  for k in eachindex(vals)
    i = rows[k]
    j = cols[k]
    if i < j && !iszero(vals[k])
      instance[(i, j)] = get(instance, (i, j), zero(S)) + S(vals[k])
    end
  end

  return instance
end

function _check_topology_size(topology, model::IsingModel)
  expected = TenSolver._topology_size(topology)
  actual = length(model.h)
  actual == expected ||
    throw(DimensionMismatch("PEPS topology $(repr(topology)) expects $expected spins, but the Ising model has $actual spins."))
end

_edge_supported(::SquareGrid, a::Tuple, b::Tuple) = abs(a[1] - b[1]) + abs(a[2] - b[2]) == 1
_edge_supported(::KingGrid, a::Tuple, b::Tuple) = maximum(abs.(a .- b)) == 1

function _check_layout_edges(topology, model::IsingModel, lattice)
  rows, cols, vals = findnz(model.J)
  for k in eachindex(vals)
    i = rows[k]
    j = cols[k]
    if i < j && !iszero(vals[k])
      ci = lattice[i]
      cj = lattice[j]
      if ci != cj && !_edge_supported(topology, ci, cj)
        throw(ArgumentError(
          "Ising coupling ($i, $j) is not compatible with $(repr(topology)). " *
          "Use a compatible structured topology or backend = :dmrg."
        ))
      end
    end
  end
end

function _potts_hamiltonian(backend::PEPSBackend, ig, lattice)
  if isnothing(backend.local_dimension)
    return SpinGlassNetworks.potts_hamiltonian(
      ig;
      spectrum = SpinGlassNetworks.full_spectrum,
      cluster_assignment_rule = lattice,
    )
  end

  return SpinGlassNetworks.potts_hamiltonian(
    ig,
    backend.local_dimension;
    spectrum = SpinGlassNetworks.full_spectrum,
    cluster_assignment_rule = lattice,
  )
end

function _transformations(transformations)
  transformations === :all && return SpinGlassEngine.all_lattice_transformations
  transformations === :identity && return (SpinGlassEngine.rotation(0),)
  if transformations isa Symbol
    throw(ArgumentError("Unsupported PEPS transformations $(repr(transformations)). Use :all, :identity, a transformation, or a collection of transformations."))
  end
  transformations isa Tuple && return transformations
  transformations isa AbstractVector && return Tuple(transformations)
  return (transformations,)
end

function _strategy(backend::PEPSBackend)
  backend.contraction in (:auto, :svd, :svd_truncate) && return SpinGlassEngine.SVDTruncate
  backend.contraction === :zipper && return SpinGlassEngine.Zipper
  throw(ArgumentError("Unsupported PEPS contraction $(repr(backend.contraction))."))
end

function _network(topology::SquareGrid, potts_h, transform, ::Type{T}) where {T}
  return SpinGlassEngine.PEPSNetwork{
    SpinGlassEngine.SquareSingleNode{SpinGlassEngine.GaugesEnergy},
    SpinGlassEngine.Dense,
    T,
  }(topology.m, topology.n, potts_h, transform)
end

function _network(topology::KingGrid, potts_h, transform, ::Type{T}) where {T}
  return SpinGlassEngine.PEPSNetwork{
    SpinGlassEngine.KingSingleNode{SpinGlassEngine.GaugesEnergy},
    SpinGlassEngine.Dense,
    T,
  }(topology.m, topology.n, potts_h, transform)
end

function _decoded_records(model::IsingModel, potts_h, sol, transform)
  records = NamedTuple[]
  for i in eachindex(sol.states)
    decoded = SpinGlassNetworks.decode_potts_hamiltonian_state(potts_h, sol.states[i])
    spins = [decoded[j] for j in eachindex(model.h)]
    state = spin_to_bool(spins)
    push!(records, (;
      state,
      spins,
      energy = ising_energy(model, spins),
      probability = sol.probabilities[i],
      transformation = transform,
      raw_energy = sol.energies[i],
    ))
  end
  return records
end

function _deduplicated_records(records)
  sort!(records; by = r -> (r.energy, -r.probability))

  deduped = NamedTuple[]
  positions = Dict{Any, Int}()
  for record in records
    key = Tuple(record.state)
    index = get(positions, key, nothing)
    if isnothing(index)
      push!(deduped, record)
      positions[key] = lastindex(deduped)
    else
      existing = deduped[index]
      deduped[index] = (; existing..., probability = existing.probability + record.probability)
    end
  end

  return deduped
end

function _metadata(backend::PEPSBackend, records, raw_results, failures)
  best = first(records)
  raw = raw_results[best.transformation]
  return Dict{String, Any}(
    "backend" => "SpinGlassPEPS",
    "topology" => TenSolver._topology_name(backend.topology),
    "topology_size" => TenSolver._topology_tuple(backend.topology),
    "beta" => backend.beta,
    "bond_dim" => backend.bond_dim,
    "max_states" => backend.max_states,
    "cutoff_prob" => backend.cutoff_prob,
    "onGPU" => backend.onGPU,
    "contraction" => String(backend.contraction),
    "num_sweeps" => backend.num_sweeps,
    "graduate_truncation" => backend.graduate_truncation,
    "local_dimension" => backend.local_dimension,
    "transformations_tried" => collect(string.(keys(raw_results))),
    "transformations_failed" => [string(failure.transformation) for failure in failures],
    "selected_transformation" => string(best.transformation),
    "spin_glass_energies" => collect(raw.solution.energies),
    "spin_glass_probabilities" => collect(raw.solution.probabilities),
    "largest_discarded_probability" => raw.solution.largest_discarded_probability,
  )
end

function TenSolver.solve_ising(backend::PEPSBackend, model::IsingModel; cutoff = nothing, verbosity = 1, kwargs...)
  if !isempty(kwargs)
    names = join(string.(keys(kwargs)), ", ")
    throw(ArgumentError("Unsupported PEPS backend keyword(s): $names. Configure PEPSBackend instead."))
  end

  _check_topology_size(backend.topology, model)

  S = _peps_type(model)
  instance = _ising_instance(model)
  ig = SpinGlassNetworks.ising_graph(S, instance)
  lattice = SpinGlassNetworks.super_square_lattice(TenSolver._topology_tuple(backend.topology))
  _check_layout_edges(backend.topology, model, lattice)
  potts_h = _potts_hamiltonian(backend, ig, lattice)
  params = SpinGlassEngine.MpsParameters{S}(;
    bond_dim = backend.bond_dim,
    num_sweeps = backend.num_sweeps,
  )
  search_params = SpinGlassEngine.SearchParameters(;
    max_states = backend.max_states,
    cutoff_prob = backend.cutoff_prob,
  )
  strategy = _strategy(backend)

  records = NamedTuple[]
  raw_results = Dict{Any, Any}()
  failures = NamedTuple[]
  for transform in _transformations(backend.transformations)
    try
      net = _network(backend.topology, potts_h, transform, S)
      ctr = SpinGlassEngine.MpsContractor(
        strategy,
        net,
        params;
        onGPU = backend.onGPU,
        beta = S(backend.beta),
        graduate_truncation = backend.graduate_truncation,
      )
      merge_strategy = SpinGlassEngine.merge_branches(ctr; merge_prob = :none)
      sol, info = SpinGlassEngine.low_energy_spectrum(
        ctr,
        search_params,
        merge_strategy;
        no_cache = backend.no_cache,
      )

      raw_results[transform] = (; solution = sol, info)
      append!(records, _decoded_records(model, potts_h, sol, transform))
    catch err
      push!(failures, (;
        transformation = transform,
        error = sprint(showerror, err),
      ))
      verbosity > 0 && @warn "SpinGlassPEPS transformation failed" transformation = transform exception = (err, catch_backtrace())
    finally
      SpinGlassEngine.clear_memoize_cache()
    end
  end

  if isempty(records)
    if isempty(failures)
      throw(ArgumentError("SpinGlassPEPS did not return any states."))
    end

    failure_summary = join(("$(failure.transformation): $(failure.error)" for failure in failures), "; ")
    throw(ArgumentError("SpinGlassPEPS did not return any states. Failed transformations: $failure_summary"))
  end
  records = _deduplicated_records(records)
  states = [record.state for record in records]
  energies = S[record.energy for record in records]
  probabilities = S[record.probability for record in records]
  metadata = _metadata(backend, records, raw_results, failures)
  raw = (; results = raw_results, failures)

  verbosity > 0 && @info "SpinGlassPEPS backend finished" energy = first(energies) states = length(states)

  return first(energies), PEPSSolution{S}(states, energies, probabilities, metadata, raw)
end

end # module
