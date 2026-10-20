import ITensors, ITensorMPS

# All matrices and matrix elements in these formulation tests are dimensionless.
function one_site_mpo(site, matrix)
  tensor = ITensors.ITensor(eltype(matrix), ITensors.prime(site), site)
  for row in axes(matrix, 1), column in axes(matrix, 2)
    tensor[ITensors.prime(site) => row, site => column] = matrix[row, column]
  end
  return ITensorMPS.MPO([tensor])
end

function assert_mpo_matrix(H, sites, bitstrings, expected)
  for (row, bra) in enumerate(bitstrings), (column, ket) in enumerate(bitstrings)
    @test mpo_matrix_element(H, sites, bra, ket) ≈ expected[row, column]
  end
end

@testset "Projected Hamiltonian formulations" begin
  sites = ITensors.siteinds("Qudit", 1; dim = 2)
  site = only(sites)
  bitstrings = ((0,), (1,))

  @testset "Commuting formulation remains the default" begin
    H = one_site_mpo(site, [2.0 0.0; 0.0 -1.0])
    P = one_site_mpo(site, [1.0 0.0; 0.0 0.0])

    H_default = TenSolver.project_hamiltonian(H, P; cutoff = 1e-12)
    H_commuting =
      TenSolver.project_hamiltonian(H, P; formulation = :commuting, cutoff = 1e-12)
    H_sandwich =
      TenSolver.project_hamiltonian(H, P; formulation = :sandwich, cutoff = 1e-12)

    expected = [2.0 0.0; 0.0 0.0]
    assert_mpo_matrix(H_default, sites, bitstrings, expected)
    assert_mpo_matrix(H_commuting, sites, bitstrings, expected)
    assert_mpo_matrix(H_sandwich, sites, bitstrings, expected)
  end

  @testset "Sandwich formulation supports a noncommuting projector" begin
    # H = |0><0| and P = |+><+| do not commute:
    # HP = [1/2 1/2; 0 0], while P'HP = [1/4 1/4; 1/4 1/4].
    H = one_site_mpo(site, [1.0 0.0; 0.0 0.0])
    P = one_site_mpo(site, fill(0.5, 2, 2))

    H_commuting =
      TenSolver.project_hamiltonian(H, P; formulation = :commuting, cutoff = 1e-12)
    H_sandwich =
      TenSolver.project_hamiltonian(H, P; formulation = :sandwich, cutoff = 1e-12)

    assert_mpo_matrix(H_commuting, sites, bitstrings, [0.5 0.5; 0.0 0.0])
    assert_mpo_matrix(H_sandwich, sites, bitstrings, fill(0.25, 2, 2))
  end

  @testset "Sandwich formulation preserves projection order" begin
    H = one_site_mpo(site, [1.0 0.0; 0.0 2.0])
    P_zero = one_site_mpo(site, [1.0 0.0; 0.0 0.0])
    P_plus = one_site_mpo(site, fill(0.5, 2, 2))

    zero_then_plus = TenSolver.project_hamiltonian(
      H,
      (P_zero, P_plus);
      formulation = :sandwich,
      cutoff = 1e-12,
    )
    plus_then_zero = TenSolver.project_hamiltonian(
      H,
      (P_plus, P_zero);
      formulation = :sandwich,
      cutoff = 1e-12,
    )

    assert_mpo_matrix(zero_then_plus, sites, bitstrings, fill(0.25, 2, 2))
    assert_mpo_matrix(plus_then_zero, sites, bitstrings, [0.75 0.0; 0.0 0.0])
  end

  @test_throws ArgumentError TenSolver.project_hamiltonian(
    one_site_mpo(site, [1.0 0.0; 0.0 0.0]),
    one_site_mpo(site, [1.0 0.0; 0.0 0.0]);
    formulation = :unknown,
  )
end

@testset "Minimized projectors preserve dense masking" begin
  sites = ITensors.siteinds("Qubit", 3)
  domain = TenSolver.Domains{Float64}(0:1, 3)
  constraints = AbstractConstraint[
    SumConstraint([1, 2, 3], [1, 2, 3], 4; relation = :(<=)),
    AssignmentConstraint([1, 3], [1], :(==), 1),
    SumModConstraint([1, 3], [1, 1], 1; mod = 2),
  ]
  projections = @inferred TenSolver.projection_mpos(constraints, sites; domain)
  terms = ITensorMPS.OpSum()
  terms += 1.0, "X", 1
  terms += 2.0, "Z", 2
  terms += -0.5, "X", 3
  H = ITensorMPS.MPO(terms, sites)
  projected_H = @inferred TenSolver.project_hamiltonian(
    H,
    projections;
    formulation = :sandwich,
    cutoff = 1e-12,
  )
  psi = ITensorMPS.MPS(sites, fill("+", 3))
  projected_psi = TenSolver.project_state(psi, projections; cutoff = 1e-12)

  for bra in all_bitstrings(3), ket in all_bitstrings(3)
    # Independent matrix elements of X₁ + 2Z₂ - X₃/2.
    h = bra == ket ? 2.0 * (ket[2] == 0 ? 1 : -1) : 0.0
    h += bra == (1-ket[1], ket[2], ket[3]) ? 1.0 : 0.0
    h += bra == (ket[1], ket[2], 1-ket[3]) ? -0.5 : 0.0
    expected =
      is_feasible(collect(bra), constraints) && is_feasible(collect(ket), constraints) ? h :
      0.0
    @test mpo_matrix_element(projected_H, sites, bra, ket) ≈ expected atol=1e-10
  end
  for bits in all_bitstrings(3)
    expected = is_feasible(collect(bits), constraints) ? inv(sqrt(8.0)) : 0.0
    @test mps_amplitude(projected_psi, sites, bits) ≈ expected atol=1e-10
  end
end
