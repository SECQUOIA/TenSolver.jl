import ITensors, ITensorMPS

all_bitstrings(n::Number) = Iterators.product(fill(0:1, n)...)
all_bitstrings(sites::Vector{<:ITensors.Index}) = all_bitstrings(length(sites))

function mpo_diagonal(H, sites, bits)
  psi = ITensorMPS.MPS(sites, string.(bits))
  return real(ITensors.inner(psi', H, psi))
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
      H = TenSolver.projection_mpo(constraint, sites)

      for bits in all_bitstrings(sites)
        expected = Float64(is_feasible(collect(bits), constraint))
        @test mpo_diagonal(H, sites, bits) ≈ expected
      end
    end
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
