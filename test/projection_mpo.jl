import ITensors, ITensorMPS

all_bitstrings(n::Number) = Iterators.product(fill(0:1, n)...)
all_bitstrings(sites::Vector{<:ITensors.Index}) = all_bitstrings(length(sites))

function mpo_diagonal(H, sites, bits)
  psi = ITensorMPS.MPS(sites, string.(bits))
  return real(ITensors.inner(psi', H, psi))
end

function assert_projection_matches_feasibility(constraint, sites)
  H = TenSolver.projection_mpo(constraint, sites)

  for bits in all_bitstrings(sites)
    expected = Float64(is_feasible(collect(bits), constraint))
    @test mpo_diagonal(H, sites, bits) == expected
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

  return TenSolver.DFA([0, 1, 2], [0, 1], 0, Set([1]), transitions)
end

function divisible_by_three_dfa(num_sites)
  transitions = [
    Dict{Tuple{Int,Int},Int}(
      (r, a) => mod(2r + a, 3)
      for r in 0:2, a in 0:1
    )
    for _ in 1:num_sites
  ]

  return TenSolver.DFA([0, 1, 2], [0, 1], 0, Set([0]), transitions)
end

@testset "Constraints as MPO Projection" begin
  TEST_CONSTRAINTS = [
    SumConstraint([1, 3], [2, 1], :(<=), 2),
    SumConstraint([1, 2, 4], [1, 2, 3], :(==), 3),
    SumConstraint([2, 4], [2, 3], :(>=), 3),
    SumConstraint([1, 2, 4], [1, 2, 3], :(!=), 3),
    NotEqualsConstraint([1, 3], [1, 0]),
    NotEqualsConstraint([1, 2], [1.0, 0.0]),
    NotEqualsConstraint([1, 3, 2, 4], Bool[1, 0, 0, 1]),
    ExactlyOneConstraint([1, 2, 3], 1),
    ExactlyOneConstraint([1, 2, 3], 0),
    ExactlyOneConstraint([2, 4, 3], 0),
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
    sites = ITensors.siteinds("Qudit", 4; dim=2)

    for constraint in TEST_CONSTRAINTS
      dfa = TenSolver.constraint_to_dfa(constraint, sites)

      for bits in all_bitstrings(sites)
        expected = is_feasible(collect(bits), constraint)
        @test dfa_accepts(dfa, bits) == expected
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
          @test ITensorMPS.linkdim(H, i) == length(dfa.states)
        end
      end

      @testset "MPO Diagonal matches acceptance" begin
        for bits in all_bitstrings(sites)
          expected = Float64(dfa_accepts(dfa, bits))
          @test mpo_diagonal(H, sites, bits) ≈ expected
        end
      end
    end
  end

  @testset "Projection MPO" begin
    sites = ITensors.siteinds("Qudit", 4; dim=2)

    for constraint in TEST_CONSTRAINTS
      assert_projection_matches_feasibility(constraint, sites)
    end
  end

  @testset "NotEqualsConstraint projection" begin
    sites = ITensors.siteinds("Qudit", 4; dim=2)

    forbidden_tuple = NotEqualsConstraint([1, 3, 4], [1, 0, 1])
    H = assert_projection_matches_feasibility(forbidden_tuple, sites)

    for bits in all_bitstrings(sites)
      forbidden = bits[1] == 1 && bits[3] == 0 && bits[4] == 1
      @test mpo_diagonal(H, sites, bits) == Float64(!forbidden)
    end
    @test ITensorMPS.maxlinkdim(H) <= 2

    local_exclusions = [
      NotEqualsConstraint([i, i + 1], [1, 1])
      for i in 1:3
    ]
    Hs = TenSolver.projection_mpos(local_exclusions, sites)

    for (constraint, local_H) in zip(local_exclusions, Hs)
      left, right = TenSolver.constraint_sites(constraint)

      for bits in all_bitstrings(sites)
        forbidden = bits[left] == 1 && bits[right] == 1
        @test mpo_diagonal(local_H, sites, bits) == Float64(!forbidden)
      end
      @test ITensorMPS.maxlinkdim(local_H) <= 2
    end

    for bits in all_bitstrings(sites)
      has_adjacent_ones = any(i -> bits[i] == 1 && bits[i + 1] == 1, 1:3)
      @test is_feasible(collect(bits), local_exclusions) == !has_adjacent_ones
    end
  end

  @testset "ExactlyOneConstraint projection" begin
    cases = [
      (ExactlyOneConstraint(1:2, 1), ITensors.siteinds("Qudit", 2; dim=2)),
      (ExactlyOneConstraint(1:3, 0), ITensors.siteinds("Qudit", 3; dim=2)),
      (ExactlyOneConstraint(1:5, 1), ITensors.siteinds("Qudit", 5; dim=2)),
    ]

    for (constraint, sites) in cases
      dfa = @inferred TenSolver.constraint_to_dfa(constraint, sites)
      @test length(dfa.states) == 2

      for bits in all_bitstrings(sites)
        target_hits = count(==(constraint.value), bits)
        @test dfa_accepts(dfa, bits) == (target_hits == 1)
      end

      H = assert_projection_matches_feasibility(constraint, sites)
      @test ITensorMPS.maxlinkdim(H) <= 2
    end
  end

  @testset "RelationConstraint projection" begin
    sites = ITensors.siteinds("Qudit", 5; dim=2)

    relation_cases = [
      RelationConstraint(left, relation, right)
      for relation in (:(==), :(!=), :(<=), :(>=))
      for (left, right) in ((1, 4), (4, 1), (2, 5), (5, 2))
    ]

    for constraint in relation_cases
      dfa = TenSolver.constraint_to_dfa(constraint, sites)
      @test length(dfa.states) <= 2

      for bits in all_bitstrings(sites)
        @test dfa_accepts(dfa, bits) == is_feasible(collect(bits), constraint)
      end

      H = assert_projection_matches_feasibility(constraint, sites)
      @test ITensorMPS.maxlinkdim(H) <= 2
    end
  end

  @testset "ExactlyOneConstraint projection" begin
    sites = ITensors.siteinds("Qudit", 4; dim=2)

    exact_one = ExactlyOneConstraint([1, 3], 1)
    dfa = TenSolver.constraint_to_dfa(exact_one, sites)
    H = assert_projection_matches_feasibility(exact_one, sites)

    @test length(dfa.states) == 2
    @test ITensorMPS.maxlinkdim(H) <= 2
  end

  @testset "SumConstraint floating-point lowering" begin
    sites = ITensors.siteinds("Qudit", 3; dim=2)

    constraints = [
      SumConstraint([1, 2], [1.0, 1.0], 1.0; relation=:(==)),
      SumConstraint([1, 3], [2.0, 1.0], 2.0; relation=:(<=)),
      SumConstraint([2, 3], [2.0, 3.0], 3.0; relation=:(>=)),
      SumConstraint([1, 2], [1.0, 2.0], 1.0; relation=:(!=)),
    ]

    for constraint in constraints
      dfa = TenSolver.constraint_to_dfa(constraint, sites)
      H = TenSolver.projection_mpo(constraint, sites)

      for bits in all_bitstrings(sites)
        expected = is_feasible(collect(bits), constraint)
        @test dfa_accepts(dfa, bits) == expected
      end

      for bits in all_bitstrings(sites)
        expected = Float64(is_feasible(collect(bits), constraint))
        @test mpo_diagonal(H, sites, bits) ≈ expected
      end
    end
  end
end
