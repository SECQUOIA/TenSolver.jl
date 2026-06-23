# SpinGlassPEPS Integration Architecture

This page records the planned boundary between TenSolver.jl and the
SpinGlassPEPS.jl work described in
[arXiv:2411.16431](https://arxiv.org/abs/2411.16431). It is a design note for
the implementation stack, not a description of behavior available in the
current TenSolver release.

The central decision is that SpinGlassPEPS should be integrated as an optional
structured-graph backend. It should not replace TenSolver's current
ITensor-based DMRG backend, become a hard runtime dependency, or change the
default behavior of [`minimize`](@ref), [`maximize`](@ref), or the JuMP
optimizer.

## Current TenSolver Boundary

TenSolver currently accepts Boolean optimization models:

- QUBO matrices, with optional linear and constant terms.
- PUBO polynomials through MultivariatePolynomials.jl.
- QUBODrivers/JuMP models that are converted to QUBO data before solving.

The current solver path tensorizes the objective into an ITensors.jl MPO over
Boolean variables represented as two-dimensional qudit sites. It then applies
DMRG through ITensorMPS.jl and returns:

- the best sampled objective value; and
- a [`Solution`](@ref), which wraps the MPS and per-iteration convergence
  traces.

This backend is general with respect to the variable ordering and QUBO/PUBO
interaction pattern. It does not require a graph layout such as a grid,
Pegasus, or Zephyr topology.

## Proposed PEPS Backend Boundary

SpinGlassPEPS.jl targets a different solver boundary. The package itself is an
umbrella package that reexports component packages including:

- SpinGlassNetworks.jl for Ising graph construction and lattice/cluster rules.
- SpinGlassEngine.jl for Potts Hamiltonians, branch-and-bound search, low-energy
  spectra, and droplet reconstruction.
- SpinGlassTensors.jl for PEPS construction and boundary-MPS contraction.
- SpinGlassExhaustive.jl for exhaustive utilities.

The algorithm builds a finite-temperature PEPS representation of a Boltzmann
distribution for a clustered Ising/Potts Hamiltonian. It uses approximate
boundary-MPS contractions to estimate conditional probabilities and a
branch-and-bound sweep to retain the most probable low-energy configurations.
It is designed for quasi-two-dimensional graph layouts with useful locality,
including square/king-grid style layouts and, at larger scale, Pegasus and
Zephyr processor topologies.

That shape makes it complementary to TenSolver's current DMRG backend rather
than a drop-in replacement. The PEPS path should be selected only when the
caller can provide, or TenSolver can infer, a supported structured layout.

## Supported Problem Classes

The initial backend split should be:

- `DMRG`: the default backend for general QUBO/PUBO inputs and arbitrary
  variable orderings.
- `PEPS`: an optional backend for structured Ising/Potts graph inputs that can
  be embedded into a supported quasi-two-dimensional layout.

The first PEPS implementation should start with small square-grid and king-grid
layouts. Those are easier to validate, have smaller clusters, and exercise the
same data-model boundary without immediately requiring the memory footprint and
GPU-sensitive execution path needed for large Pegasus or Zephyr instances.

Pegasus and Zephyr should be treated as later targets. The paper and
SpinGlassPEPS documentation describe large unit cells, up to 24 spins for
Pegasus and 16 spins for Zephyr, and emphasize that sparse structures, local
dimensional reduction, contraction settings, and GPU execution can become
important. TenSolver should not expose those as a default path before the bridge
has small-layout correctness tests and clear user-facing configuration.

Dense arbitrary QUBOs should remain on the current DMRG path unless an explicit
embedding or layout mapping is provided. PEPS should not be advertised as a
universal improvement for all QUBOs.

## Dependency Policy

SpinGlassPEPS should not be added as a hard dependency in the first
implementation steps.

There are three reasons for this:

- TenSolver currently supports Julia 1.10, while SpinGlassPEPS 1.5.0 declares
  Julia 1.11 compatibility.
- The SpinGlassPEPS component stack brings a larger numerical dependency
  footprint, including packages used for tensor operations, truncated
  decompositions, CPU/GPU execution, and documentation/examples.
- Most TenSolver users who call the existing DMRG backend should not pay load
  time, installation, or GPU compatibility costs for an optional structured
  backend.

The preferred integration shape is a Julia package extension or a small bridge
package that is loaded only when the SpinGlassPEPS component packages are
available. The core TenSolver package should define the stable data and result
interfaces that the bridge implements. The bridge should depend on the
SpinGlassPEPS components directly when that is cleaner than depending on the
umbrella reexport package.

## Data Model Boundary

TenSolver's public inputs are Boolean optimization models. The PEPS solver
expects structured spin-glass data. The bridge must therefore make the
conversion explicit.

The expected data flow is:

1. Normalize TenSolver input to a Boolean QUBO representation.
2. Convert Boolean variables `x in {0, 1}` to Ising spins `s in {-1, +1}` using
   one documented convention.
3. Build an Ising graph with local fields and pair couplings.
4. Attach a supported layout or cluster assignment rule.
5. Build the corresponding Potts Hamiltonian and PEPS network.
6. Run boundary-MPS contraction plus branch-and-bound search.
7. Decode returned Potts/Ising states back to Boolean vectors.
8. Adapt results to TenSolver and QUBOTools result types.

The QUBO-to-Ising conversion is important enough to land as its own PR before
the backend bridge. It should include exact round-trip and energy-preservation
tests for small instances. The implementation should make sign and constant
offset conventions visible because mistakes there can silently return plausible
but wrong energies.

TenSolver uses the Boolean/spin convention `x_i = (s_i + 1) / 2` and
`s_i = 2x_i - 1`. The conversion utilities are adapters into QUBOTools forms,
not a separate conversion implementation. They preserve TenSolver's current
QUBO truth objective, `dot(x, Q, x) + dot(l, x) + c`; for non-symmetric
matrices the effective pair coefficient is therefore `Q[i, j] + Q[j, i]`.
`qubo_to_ising(Q, l, c)` returns a sparse QUBOTools form in `SpinDomain`, so
`QUBOTools.value(bool_to_spin(x), qubo_to_ising(Q, l, c))` matches the Boolean
objective for every bitstring `x`.

Layout metadata should also be explicit. A PEPS backend call should know whether
the variables are in row-major square-grid order, king-grid order, or a later
Pegasus/Zephyr indexing convention. TenSolver should not guess a topology from a
dense QUBO matrix unless a future API defines that inference clearly.

## Result Model Boundary

The current TenSolver result is intentionally compact: `energy, Solution` from
the direct API and a QUBOTools `SampleSet` from the optimizer API.

SpinGlassPEPS can produce richer output, including ranked energies, states,
probabilities, largest discarded probability during branch-and-bound, lattice
transformation information, retained singular-value diagnostics, and droplet
metadata.

The bridge should preserve this information without leaking SpinGlassPEPS
internal types as the default public result. A practical result boundary is:

- Direct TenSolver calls return the same high-level shape as today for ordinary
  users.
- PEPS-specific metadata is stored in a backend metadata object or result field
  that can be inspected by advanced users.
- QUBODrivers/JuMP calls continue to return a QUBOTools `SampleSet`, with PEPS
  metadata stored under namespaced metadata keys.

The first bridge should preserve at least:

- ranked Boolean states and their objective values;
- PEPS probabilities when available;
- largest discarded probability when available;
- chosen layout and lattice transformation;
- branch-and-bound and contraction parameters; and
- droplet metadata when requested.

## Backend Selection

The default backend must remain the current DMRG implementation.

TenSolver exposes a small backend-selection interface without forcing users to
learn the SpinGlassPEPS API. The current behavior is:

- no backend argument means current DMRG behavior;
- `backend = :dmrg` selects the current path explicitly;
- `backend = DMRGBackend()` selects the current path explicitly through the
  backend-object interface; and
- unavailable backend symbols error clearly without changing default DMRG
  behavior.

This stack step keeps the direct PEPS path as non-public scaffolding. The core
package contains internal backend, topology, and result boundaries, while the
exported `solve_ising` function is the public Ising boundary that optional
structured backends may implement. `TenSolverSpinGlassPEPSExt` owns the
SpinGlass component imports and calls. This keeps ordinary TenSolver installs
on the existing dependency footprint and avoids documenting an activation path
that cannot be tested from registered packages.

The extension remains gated while the upstream dependency stack settles. In
local checks against SpinGlassNetworks 1.4, SpinGlassEngine 1.6, and
SpinGlassTensors 1.3, the current registered component compat bounds do not
resolve with TenSolver's ITensors/QUBOTools environment. The source bridge and
gated tests are kept in this stack step so the TenSolver boundary is concrete,
but the PEPS backend types are not exported or listed in the public API until
CI can exercise the SpinGlass component stack.

Until that activation path is exercised, this stack step is scaffolding rather
than the final completion of the PEPS backend issue. The closing PR should
include a passing small structured-grid CPU solve through the optional
SpinGlass component stack.

The initial internal structured topology scaffolding covers one-spin-per-site
and multi-spin-per-site square/king grids. QUBO inputs are converted through
[`qubo_to_ising`](@ref) before the PEPS extension builds a SpinGlassNetworks
Ising graph, clusters it with `super_square_lattice`, constructs the Potts
Hamiltonian, runs `MpsContractor` plus `low_energy_spectrum`, and decodes
retained states back to TenSolver Boolean vectors.

Later PRs should add QUBODrivers/JuMP raw optimizer attributes for backend and
PEPS parameters.

Any PEPS selection API must validate that the problem includes enough topology
metadata for the structured backend. If the topology is missing or unsupported,
the error should explain whether to provide a layout, use the DMRG backend, or
install/load the optional PEPS bridge.

## Stacked PR Plan

The integration should be implemented as a sequence of stacked PRs:

1. Add this design document and link it from the documentation navigation.
2. Add QUBOTools-backed QUBO/Ising conversion adapters with exact
   energy-preservation tests.
3. Introduce a backend interface while keeping the current DMRG backend as the
   default implementation.
4. Add internal optional SpinGlassPEPS-backed structured solver scaffolding for
   direct structured inputs, without closing the issue until the real extension
   path is exercised.
5. Expose the PEPS backend through QUBODrivers/JuMP attributes, including
   layout and contraction/search parameters.
6. Add user documentation, examples, and benchmark scripts that compare the DMRG
   and PEPS paths on appropriate problem families.

Each PR should remain useful on its own. The early PRs should avoid new runtime
dependencies and should preserve the current public behavior exactly.

## Non-Goals

This integration should not:

- vendor SpinGlassPEPS internals into TenSolver;
- replace the existing `minimize` or `maximize` behavior;
- require GPU packages for installing or loading TenSolver;
- promise that PEPS is better for arbitrary dense QUBOs;
- infer complex hardware layouts without explicit metadata; or
- hide approximate-contraction limitations behind a generic solver name.

## Risks

The main technical risks are:

- Boolean-to-spin convention mistakes, especially sign and constant-offset
  errors.
- Topology mismatch between TenSolver variable order and the PEPS layout.
- Julia-version compatibility between TenSolver and SpinGlassPEPS.
- Increased dependency and load-time footprint if the bridge is not optional.
- Approximate contraction instability at high inverse temperature.
- Overstating sampling diversity from a branch-and-bound method that returns a
  limited retained state set.

The implementation stack should address these risks through exact small-instance
conversion tests, explicit topology metadata, optional dependency loading, and
documentation that describes the PEPS backend as a structured heuristic solver.
