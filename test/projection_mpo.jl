import ITensors, ITensorMPS

all_bitstrings(n::Number) = Iterators.product(fill(0:1, n)...)
all_bitstrings(sites::Vector{<:ITensors.Index}) = all_bitstrings(length(sites))

function mpo_diagonal(H, sites, bits)
  psi = ITensorMPS.MPS(sites, string.(bits))
  return real(ITensors.inner(psi', H, psi))
end

function assert_projection_matches_feasibility(
  constraint,
  sites;
  domain = TenSolver.Domains{Float64}(0:1, length(sites)),
)
  dfa = TenSolver.constraint_to_dfa(constraint, domain)
  minimized = @inferred TenSolver.minimize_dfa(dfa)
  arrays = @inferred TenSolver.transition_tensors(Int, dfa)
  H = @inferred TenSolver.projection_mpo(constraint, sites; domain)

  assignments = Iterators.product(domain...)
  for assignment in assignments
    expected = Float64(is_feasible(collect(assignment), constraint))
    basis_values = [findfirst(==(v), domain[k]) - 1 for (k, v) in pairs(assignment)]
    @test dfa_accepts(dfa, assignment) == Bool(expected)
    @test dfa_accepts(minimized, assignment) == Bool(expected)
    @test raw_projection_diagonal(arrays, basis_values) == expected
    @test mpo_diagonal(H, sites, basis_values) ≈ expected atol=sqrt(eps(Float64))
  end

  return H
end

function assert_projection_spot_checks(constraint, sites; domain = TenSolver.Domains{Float64}(0:1, length(sites)))
  H = TenSolver.projection_mpo(constraint, sites; domain)

  for _ in 1:2
    x = rand(domain)
    basis = [findfirst(==(v), domain[k]) - 1 for (k, v) in pairs(x)]
    @test mpo_diagonal(H, sites, basis) ≈ is_feasible(x, constraint) atol=sqrt(eps(Float64))
  end

  return H
end

function dfa_accepts(dfa, input)
  state = dfa.initial

  for (i, a) in enumerate(input)
    next_state = get(dfa.transitions[i], (state, a), nothing)
    isnothing(next_state) && return false
    state = next_state
  end

  return state in dfa.accepting
end

function exactly_one_one_dfa(num_sites)
  transitions = [
    Dict{Tuple{Int,Int},Int}(
      (0, 0) => 0,
      (0, 1) => 1,
      (1, 0) => 1,
      (1, 1) => 2,
      (2, 0) => 2,
      (2, 1) => 2,
    )
    for _ in 1:num_sites
  ]

  return TenSolver.DFA([0, 1, 2], fill(0:1, num_sites), 0, Set([1]), transitions)
end

function divisible_by_three_dfa(num_sites)
  transitions = [
    Dict{Tuple{Int,Int},Int}(
      (r, a) => mod(2r + a, 3)
      for r in 0:2, a in 0:1
    )
    for _ in 1:num_sites
  ]

  return TenSolver.DFA([0, 1, 2], fill(0:1, num_sites), 0, Set([0]), transitions)
end

# Restore only the implicit boundary dimensions, then contract exact integer
# entries. This checks the mask before any floating-point MPO compression.
function raw_projection_diagonal(arrays, basis)
  value = [1]
  for (i, A) in enumerate(arrays)
    left = i == 1 ? 1 : size(A, 1)
    right = i == length(arrays) ? 1 : size(A, i == 1 ? 1 : 2)
    d = size(A, ndims(A))
    tensor = reshape(A, left, right, d, d)
    value = vec(transpose(value) * tensor[:, :, basis[i] + 1, basis[i] + 1])
  end
  return only(value)
end

raw_layer_widths(arrays) = [size(arrays[i], i == 1 ? 1 : 2) for i in 1:(length(arrays) - 1)]

# Count distinct nonempty suffix languages independently of the minimizer.
function suffix_language_widths(dfa)
  prefix_states = Set([dfa.initial])
  widths = Int[]
  for i in 1:(length(dfa.transitions) - 1)
    prefix_states = Set(dfa.transitions[i][(s, a)] for s in prefix_states for
        a in dfa.alphabets[i] if haskey(dfa.transitions[i], (s, a)))
    suffixes = collect(Iterators.product(dfa.alphabets[(i + 1):end]...))
    languages = Set{Vector{Bool}}()
    for state in prefix_states
      language = map(suffixes) do suffix
        s = state
        for (j, a) in enumerate(suffix)
          table = dfa.transitions[i + j]
          if !haskey(table, (s, a))
            return false
          end
          s = table[(s, a)]
        end
        return s in dfa.accepting
      end
      if any(language)
        push!(languages, vec(language))
      end
    end
    push!(widths, max(1, length(languages)))
  end
  return widths
end

@testset "Exact layered DFA minimization" begin
  domain = TenSolver.Domains{Float64}(0:1, 4)
  weighted = SumConstraint(collect(1:4), [1, 2, 3, 4], 5; relation = :(<=))
  dfa = TenSolver.constraint_to_dfa(weighted, domain)
  minimized = @inferred TenSolver.minimize_dfa(dfa)
  arrays = @inferred TenSolver.transition_tensors(Int, dfa)
  @test length.(minimized.states) == [1, 2, 3, 2, 1]
  @test raw_layer_widths(arrays) == [2, 3, 2]
  @test size.(arrays) == [(2, 2, 2), (2, 3, 2, 2), (3, 2, 2, 2), (2, 2, 2)]

  # :unreachable has an accepting suffix but no prefix; :dead has a prefix
  # but no accepting suffix. :a and :b differ only on rejecting transitions.
  synthetic = TenSolver.DFA(
    [:start, :a, :b, :dead, :unreachable, :accept, :reject],
    [[0, 1, 2], [4, 5], [7, 8]],
    :start,
    Set([:accept]),
    [
      Dict((:start, 0)=>:a, (:start, 1)=>:b, (:start, 2)=>:dead),
      Dict(
        (:a, 4)=>:a,
        (:a, 5)=>:reject,
        (:b, 4)=>:b,
        (:dead, 4)=>:dead,
        (:unreachable, 4)=>:a,
      ),
      Dict(
        (:a, 7)=>:accept,
        (:b, 7)=>:accept,
        (:unreachable, 7)=>:accept,
        (:dead, 8)=>:reject,
      ),
    ],
  )
  reduced = @inferred TenSolver.minimize_dfa(synthetic)
  @test length.(reduced.states) == [1, 1, 1, 1]
  tensors = TenSolver.transition_tensors(Int, synthetic)
  @test size.(tensors) == [(1, 3, 3), (1, 1, 2, 2), (1, 2, 2)]
  sites = [
    ITensors.Index(length(a), "Site,Qudit,n=$i") for
    (i, a) in enumerate(synthetic.alphabets)
  ]
  P = TenSolver.dfa_to_mpo(Float64, synthetic, sites)
  for assignment in Iterators.product(synthetic.alphabets...)
    basis = [findfirst(==(a), synthetic.alphabets[i])-1 for (i, a) in enumerate(assignment)]
    expected = assignment[1] in (0, 1) && assignment[2] == 4 && assignment[3] == 7
    @test dfa_accepts(synthetic, assignment) == expected
    @test dfa_accepts(reduced, assignment) == expected
    @test raw_projection_diagonal(tensors, basis) == expected
    @test mpo_diagonal(P, sites, basis) ≈ expected atol=1e-12
  end

  # Exhaustive language comparison of small partial, step-dependent automata,
  # including arbitrary terminal accepting sets and heterogeneous alphabets.
  for _ in 1:20
    alphabets = [[0, 1], [2], [3, 4, 5], [6, 7]]
    tables = [
      Dict((s, a)=>rand(1:4) for s in 1:4 for a in alphabet if rand(Bool)) for
      alphabet in alphabets
    ]
    original =
      TenSolver.DFA(collect(1:4), alphabets, 1, Set(filter(_->rand(Bool), 1:4)), tables)
    reduced = TenSolver.minimize_dfa(original)
    arrays = TenSolver.transition_tensors(Int, original)
    @test raw_layer_widths(arrays) == suffix_language_widths(original)
    for assignment in Iterators.product(alphabets...)
      basis = [findfirst(==(a), alphabets[i])-1 for (i, a) in enumerate(assignment)]
      @test dfa_accepts(reduced, assignment) == dfa_accepts(original, assignment)
      @test raw_projection_diagonal(arrays, basis) == dfa_accepts(original, assignment)
    end
  end

  @testset "Constraint edge cases and permutations" begin
    domain = TenSolver.Domains{Float64}([[0, 2], [1], [0, 1, 3], [0, 2]], 4)
    constraints = AbstractConstraint[
      SumConstraint([1, 3, 4], [0, 1, 5], rhs; relation) for rhs in (0, 2, 9) for
      relation in (:(==), :(!=), :(<=), :(>=))
    ]
    append!(
      constraints,
      [
        SumModConstraint([1, 3, 4], [-1, 2, 3], 1; mod = 4),
        NotEqualsConstraint([1, 3], [2, 1]),
        AssignmentConstraint([1, 3, 4], [1, 2], :(==), 1),
        RelationConstraint(4, :(>=), 1),
      ],
    )
    for permutation in ([1, 2, 3, 4], [3, 1, 4, 2])
      permuted_domain = TenSolver.Domains{Float64}(domain.ds[permutation], 4)
      sites = [
        ITensors.Index(length(d), "Site,Qudit,n=$i") for
        (i, d) in enumerate(permuted_domain)
      ]
      for constraint in constraints
        permuted = TenSolver.permute(constraint, permutation)
        assert_projection_matches_feasibility(permuted, sites; domain = permuted_domain)
        for x in Iterators.product(domain...)
          @test is_feasible(collect(x)[permutation], permuted) ==
                is_feasible(collect(x), constraint)
        end
      end
    end
  end

  @testset "Single site and empty languages" begin
    for values in ([0], [0, 1, 2])
      domain = TenSolver.Domains{Float64}(values, 1)
      sites = ITensors.siteinds("Qudit", 1; dim = length(values))
      for rhs in (0, 1, 3)
        assert_projection_matches_feasibility(
          SumConstraint([1], [1], rhs; relation = :(==)),
          sites;
          domain,
        )
      end
    end
    for n in (1, 2, 4)
      dfa = TenSolver.DFA(
        [0, 1],
        fill([0, 1], n),
        0,
        Set{Int}(),
        [Dict((s, a)=>s for s in 0:1 for a in 0:1) for _ in 1:n],
      )
      reduced = TenSolver.minimize_dfa(dfa)
      @test length.(reduced.states) == ones(Int, n+1)
      @test all(isempty, reduced.transitions)
      arrays = TenSolver.transition_tensors(Int, dfa)
      @test all(A->all(iszero, A), arrays)
      P = TenSolver.dfa_to_mpo(Float64, dfa, ITensors.siteinds("Qudit", n; dim = 2))
      @test norm(P) ≈ 0 atol=1e-12
    end
  end
end

@testset "Constraints as MPO Projection" begin
  TEST_CONSTRAINTS = [
    SumConstraint([1, 3], [2, 1], :(<=), 2),
    SumConstraint([1, 2, 4], [1, 2, 3], :(==), 3),
    SumConstraint([2, 4], [2, 3], :(>=), 3),
    SumConstraint([1, 2, 4], [1, 2, 3], :(!=), 3),
    SumModConstraint([1, 2, 4], [1, 2, 3], 3; mod = 3),
    NotEqualsConstraint([1, 3], [1, 0]),
    NotEqualsConstraint([1, 2], [1.0, 0.0]),
    NotEqualsConstraint([1, 3, 2, 4], Bool[1, 0, 0, 1]),
    AssignmentConstraint([1, 2, 3], [1], :(==), 1),
    AssignmentConstraint([1, 2, 3], [0], :(==), 1),
    AssignmentConstraint([2, 4, 3], [0], :(==), 1),
    RelationConstraint(4, :(<=), 2),
  ]

  @testset "DFA correctness" begin
    exactly_one = exactly_one_one_dfa(3)
    @test dfa_accepts(exactly_one, (0, 0, 0)) == false
    @test dfa_accepts(exactly_one, (1, 0, 0)) == true
    @test dfa_accepts(exactly_one, (0, 1, 0)) == true
    @test dfa_accepts(exactly_one, (1, 1, 0)) == false
    @test dfa_accepts(exactly_one, (1, 1, 1)) == false

    divisible_by_3 = divisible_by_three_dfa(4)
    @test dfa_accepts(divisible_by_3, (0, 0, 0, 0)) == true
    @test dfa_accepts(divisible_by_3, (0, 0, 1, 1)) == true
    @test dfa_accepts(divisible_by_3, (0, 1, 1, 0)) == true
    @test dfa_accepts(divisible_by_3, (1, 0, 1, 0)) == false
    @test dfa_accepts(divisible_by_3, (1, 1, 1, 1)) == true
  end

  @testset "Constraint -> DFA" begin
    sites  = ITensors.siteinds("Qudit", 4; dim=2)
    domain = TenSolver.Domains{Float64}(0:1, length(sites))

    for constraint in TEST_CONSTRAINTS
      dfa = TenSolver.constraint_to_dfa(constraint, domain)

      for bits in all_bitstrings(sites)
        expected = is_feasible(collect(bits), constraint)
        @test dfa_accepts(dfa, bits) ≈ expected atol=1e-8
      end
    end
  end

  @testset "DFA -> MPO" begin
    examples = [
      (exactly_one_one_dfa(3),    ITensors.siteinds("Qudit", 3; dim=2)),
      (divisible_by_three_dfa(4), ITensors.siteinds("Qudit", 4; dim=2)),
    ]

    for (dfa, sites) in examples
      H = TenSolver.dfa_to_mpo(Float64, dfa, sites)

      @testset "MPO Dimensions" begin
        for i in eachindex(sites)
          @test ITensors.dim(ITensors.siteind(H, i)) == ITensors.dim(sites[i])
        end

        for i in 1:length(sites)-1
          @test ITensorMPS.linkdim(H, i) <= length(dfa.states)
        end
      end

      @testset "MPO Diagonal matches acceptance" begin
        for bits in all_bitstrings(sites)
          expected = Float64(dfa_accepts(dfa, bits))
          @test mpo_diagonal(H, sites, bits) ≈ expected atol=1e-8
        end
      end
    end
  end

  @testset "Projection MPO spot checks" begin
    sites = ITensors.siteinds("Qudit", 4; dim=2)

    cases = [
      SumConstraint([1, 3], [2, 1], :(<=), 2),
      NotEqualsConstraint([1, 3], [1, 0]),
      NotEqualsConstraint([1, 2], [1.0, 0.0]),
      NotEqualsConstraint([1, 3, 2, 4], Bool[1, 0, 0, 1]),
      AssignmentConstraint([1, 3], [1], :(==), 1),
      RelationConstraint(4, :(<=), 2),
    ]

    for constraint in cases
      assert_projection_spot_checks(constraint, sites)
    end
  end

  @testset "Projected Hamiltonian and state utilities" begin
    Q = [
       1.0   0.25 -0.50
       0.25 -2.00  0.75
      -0.50  0.75  3.00
    ]
    domain = TenSolver.Domains{Float64}(0:1, 3)
    H = TenSolver.tensorize(Q; domain)
    sites = ITensorMPS.siteinds(first, H; plev=0)

    constraints = AbstractConstraint[
      NotEqualsConstraint([1, 2], [1, 1]),
      AssignmentConstraint([1, 3], [0], :(==), 1),
    ]
    projections = TenSolver.projection_mpos(constraints, sites; domain)

    @test ITensorMPS.maxlinkdim(projections[1]) <= 2

    H_eff = TenSolver.project_hamiltonian(H, projections; cutoff=1e-12)
    expected_maxlink = ITensorMPS.maxlinkdim(H) * prod(ITensorMPS.maxlinkdim, projections)
    @test ITensorMPS.maxlinkdim(H_eff) <= expected_maxlink

    psi = ITensorMPS.MPS(sites, fill("full", length(sites)))
    projected_psi = TenSolver.project_state(psi, projections; cutoff=1e-12)

    @test ITensorMPS.siteinds(first, H_eff; plev=0) == sites
    @test ITensorMPS.siteinds(first, H_eff; plev=1) == ITensors.prime.(sites)
    @test all(isempty, ITensorMPS.siteinds(all, H_eff; plev=2))
    @test ITensorMPS.siteinds(projected_psi) == sites

    for bits in all_bitstrings(sites)
      feasible = is_feasible(collect(bits), constraints)
      expected = feasible ?
        mpo_diagonal(H, sites, bits) :
        0.0

      @test mpo_diagonal(H_eff, sites, bits) ≈ expected atol=1e-10
    end

    # Both the objective and projections are diagonal. Representative
    # off-diagonal checks guard against index/prime mistakes without repeating
    # the same zero contraction for every pair of basis states.
    for (bra_bits, ket_bits) in (
      ((0, 0, 0), (1, 0, 0)),
      ((0, 0, 1), (0, 1, 1)),
      ((0, 0, 0), (0, 0, 1)),
      ((0, 0, 0), (1, 1, 0)),
    )
      @test mpo_matrix_element(H_eff, sites, bra_bits, ket_bits) ≈ 0.0 atol=1e-10
    end

    for bits in all_bitstrings(sites)
      expected = is_feasible(collect(bits), constraints) ?
        mps_amplitude(psi, sites, bits) :
        0.0

      @test mps_amplitude(projected_psi, sites, bits) ≈ expected atol=1e-10
    end
  end

  @testset "Infeasible projections remain zero" begin
    domain = TenSolver.Domains{Float64}(0:1, 2)
    H = TenSolver.tensorize([1.0 0.5; 0.5 2.0]; domain)
    sites = ITensorMPS.siteinds(first, H; plev=0)
    impossible = SumConstraint([1, 2], [1, 1], :(==), 3)
    P = TenSolver.projection_mpo(impossible, sites; domain)

    H_eff = TenSolver.project_hamiltonian(H, P; cutoff=1e-12)
    psi = ITensorMPS.MPS(sites, fill("full", length(sites)))
    projected_psi = TenSolver.project_state(psi, P; cutoff=1e-12)

    @test norm(H_eff) ≈ 0.0 atol=1e-12
    @test norm(projected_psi) ≈ 0.0 atol=1e-12
  end

  @testset "NotEqualsConstraint projection" begin
    sites = ITensors.siteinds("Qudit", 4; dim=2)
    domain = TenSolver.Domains{Float64}(0:1, length(sites))


    forbidden_tuple = NotEqualsConstraint([1, 3, 4], [1, 0, 1])
    H = TenSolver.projection_mpo(forbidden_tuple, sites; domain)

    for bits in all_bitstrings(sites)
      forbidden = bits[1] == 1 && bits[3] == 0 && bits[4] == 1
      @test mpo_diagonal(H, sites, bits) ≈ !forbidden atol=1e-8
    end
    @test ITensorMPS.maxlinkdim(H) <= 2

    local_exclusions = [
      NotEqualsConstraint([i, i + 1], [1, 1])
      for i in 1:3
    ]
    Hs = TenSolver.projection_mpos(local_exclusions, sites; domain)

    for (constraint, local_H) in zip(local_exclusions, Hs)
      left, right = TenSolver.constraint_sites(constraint)

      for bits in all_bitstrings(sites)
        forbidden = bits[left] == 1 && bits[right] == 1
        @test mpo_diagonal(local_H, sites, bits) ≈ !forbidden atol=1e-8
      end
      @test ITensorMPS.maxlinkdim(local_H) <= 2
    end

    for bits in all_bitstrings(sites)
      has_adjacent_ones = any(i -> bits[i] == 1 && bits[i + 1] == 1, 1:3)
      @test is_feasible(collect(bits), local_exclusions) == !has_adjacent_ones
    end
  end

  @testset "AssignmentConstraint projection" begin
    generalized_cases = [
      AssignmentConstraint([1, 3], [1, 2], relation, rhs)
      for relation in (:(==), :(!=), :(<=), :(>=))
      for rhs in (0, 1, 2, 3)
    ]
    domain = TenSolver.Domains{Float64}(0:2, 3)

    for constraint in generalized_cases
      dfa = TenSolver.constraint_to_dfa(constraint, domain)

      for assignment in Iterators.product(domain...)
        @test dfa_accepts(dfa, assignment) == is_feasible(collect(assignment), constraint)
      end
    end

    projection_sites = ITensors.siteinds("Qudit", 3; dim=3)
    for relation in (:(==), :(!=), :(<=), :(>=))
      constraint = AssignmentConstraint([1, 3], [1, 2], relation, 1)
      assert_projection_matches_feasibility(constraint, projection_sites; domain)
    end

    @test_throws BoundsError TenSolver.constraint_to_dfa(
      AssignmentConstraint([4], [1], :(==), 2),
      domain,
    )

    domain = TenSolver.Domains{Float64}(0:2, 4)
    bool_assignment = AssignmentConstraint([1, 3, 4], Bool[true], :(==), 2)
    bool_dfa = @inferred TenSolver.constraint_to_dfa(bool_assignment, domain)
    for bits in all_bitstrings(4)
      @test dfa_accepts(bool_dfa, bits) == is_feasible(collect(bits), bool_assignment)
    end


    exact_one_sites = ITensors.siteinds("Qudit", 5; dim=2)
    domain = TenSolver.Domains{Float64}(0:1, 5)
    for relation in (:(==), :(<=), :(>=))
      exact_one = AssignmentConstraint(1:5, [1], relation, 1)
      dfa = @inferred TenSolver.constraint_to_dfa(exact_one, domain)
      H = assert_projection_matches_feasibility(exact_one, exact_one_sites)

      @test ITensorMPS.maxlinkdim(H) <= 2
    end

    not_exactly_one = AssignmentConstraint(1:5, [1], :(!=), 1)
    not_equal_dfa = @inferred TenSolver.constraint_to_dfa(not_exactly_one, domain)
    @test length(not_equal_dfa.states) <= 3
  end

  @testset "RelationConstraint projection" begin
    sites = ITensors.siteinds("Qudit", 5; dim=2)

    relation_cases = [
      RelationConstraint(left, relation, right)
      for relation in (:(==), :(!=), :(<=), :(>=))
      for (left, right) in ((1, 4), (4, 1), (2, 5), (5, 2))
    ]
    domain = TenSolver.Domains{Float64}(0:1, length(sites))

    for constraint in relation_cases
      dfa = TenSolver.constraint_to_dfa(constraint, domain)
      @test length(dfa.states) <= 2

      for bits in all_bitstrings(sites)
        @test dfa_accepts(dfa, bits) == is_feasible(collect(bits), constraint)
      end

      H = TenSolver.projection_mpo(constraint, sites; domain)
      @test ITensorMPS.maxlinkdim(H) <= 2
    end
  end

  @testset "SumModConstraint projection" begin
    constraint = SumModConstraint([1, 3], [-1, 2], -1; mod = 3)

    @testset let sites   = ITensors.siteinds("Qudit", 4; dim=3),
                 domains = TenSolver.Domains{Float64}(-1:1, length(sites))
      dfa = TenSolver.constraint_to_dfa(constraint, domains)
      H = assert_projection_matches_feasibility(constraint, sites; domain = domains)

      @test length(dfa.states) <= 3
      @test ITensorMPS.maxlinkdim(H) <= constraint.mod
    end

    @testset let domains = TenSolver.Domains{Float64}([0, 0.5], 3)
      @test_throws ArgumentError TenSolver.constraint_to_dfa(constraint, domains)
    end
  end

  @testset "SumConstraint floating-point lowering" begin
    sites = ITensors.siteinds("Qudit", 3; dim=2)
    domain = TenSolver.Domains{Float64}(0:1, length(sites))

    constraints = [
      SumConstraint([1, 2], [1.0, 1.0], 1.0; relation=:(==)),
      SumConstraint([1, 3], [2.0, 1.0], 2.0; relation=:(<=)),
      SumConstraint([2, 3], [2.0, 3.0], 3.0; relation=:(>=)),
      SumConstraint([1, 2], [1.0, 2.0], 1.0; relation=:(!=)),
    ]
    for constraint in constraints
      dfa = TenSolver.constraint_to_dfa(constraint, domain)
      assert_projection_spot_checks(constraint, sites)

      for bits in all_bitstrings(sites)
        expected = is_feasible(collect(bits), constraint)
        @test dfa_accepts(dfa, bits) == expected
      end
    end

    @test_throws ArgumentError TenSolver.projection_mpo(
      SumConstraint([1], [1], 1; relation=:(==)),
      ITensors.siteinds("Qudit", 1; dim=2);
      domain = TenSolver.Domains{Float64}([-1, 1], 1),
    )
  end

  @testset "SumConstraint bond dimension scales with capacity, not size" begin
    # The exact partial-sum automaton for `sum(w .* x) <= rhs` has rhs + 2
    # states, so the projection MPO bond dimension is bounded by rhs + 2 and,
    # crucially, is independent of the number of constrained variables and of
    # the weight magnitudes. This is the structural advantage over a
    # penalty-QUBO encoding, whose DMRG bond dimension grows with problem size.
    function sumbond(sitelist, weights, rhs)
      sites = ITensors.siteinds("Qudit", maximum(sitelist); dim=2)
      domain = TenSolver.Domains{Float64}(0:1, length(sites))

      return ITensorMPS.maxlinkdim(
        TenSolver.projection_mpo(
          SumConstraint(sitelist, weights, rhs; relation=:(<=)),
          sites;
          domain = domain,
        ),
      )
    end

    for rhs in (1, 2, 4)
      # The bound holds and is reached for a moderate number of unit-weight items.
      @test sumbond(collect(1:8), ones(Int, 8), rhs) <= rhs + 2
      # Growing the item count does not grow the bond dimension ...
      @test sumbond(collect(1:16), ones(Int, 16), rhs) == sumbond(collect(1:8), ones(Int, 8), rhs)
    end

    # ... nor does inflating the weights (same rhs, arbitrary large weights).
    @test sumbond([1, 2, 3, 4, 5, 6], [5, 7, 3, 9, 2, 8], 2) <= 2 + 2

    # The projected Hamiltonian bond stays within the generic product bound.
    Q = [
       1.0  0.25 -0.5  0.0  0.0
       0.0 -2.0   0.75 0.0  0.0
       0.0  0.0   3.0  0.5  0.0
       0.0  0.0   0.0 -1.0  0.25
       0.0  0.0   0.0  0.0  2.0
    ]
    domain         = TenSolver.Domains{Float64}(0:1, 5)
    H              = TenSolver.tensorize(Q, diag(Q); domain)
    sites          = ITensorMPS.siteinds(first, H; plev=0)
    sum_constraint = SumConstraint([1, 2, 3, 4, 5], ones(Int, 5), 2; relation=:(<=))
    projections    = TenSolver.projection_mpos([sum_constraint], sites; domain)
    H_eff          = TenSolver.project_hamiltonian(H, projections; cutoff=1e-12)

    @test ITensorMPS.maxlinkdim(projections[1]) <= 2 + 2
    @test ITensorMPS.maxlinkdim(H_eff) <= ITensorMPS.maxlinkdim(H) * (2 + 2)
  end

  @testset "Integrality checks for SumConstraint and SumModConstraint" begin
    domains = TenSolver.Domains{Float64}([[0, 1], [0.1, 1.5], [-1, 1], [1, 5]], 4)
    constraints = [
      SumConstraint([1, 4], [1, 1], 1; relation = :(==)),
      SumModConstraint([1, 3, 4], [1, 1, 1], 1; mod = 3),
     ]

    for c in constraints
      @test TenSolver.constraint_to_dfa(c, domains) isa TenSolver.DFA
    end

    bad_constraints = [
      SumConstraint([2, 4], [1, 1], 1; relation = :(==)),
      SumConstraint([3, 4], [1, 1], 1; relation = :(==)),
      SumModConstraint([1, 2, 3], [1, 1, 1], 1; mod = 3),
     ]

    for c in bad_constraints
      @test_throws ArgumentError TenSolver.constraint_to_dfa(c, domains)
    end
  end
end
