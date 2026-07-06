import ITensors, ITensorMPS

all_bitstrings(n) = Iterators.product(fill(0:1, n)...)

function projection_diagonal(H, sites)
  diagonal = Dict{Tuple,Float64}()

  for assignment in all_bitstrings(length(sites))
    bits = Tuple(assignment)
    psi  = ITensorMPS.MPS(sites, string.(bits))
    diagonal[bits] = real(ITensors.inner(psi', H, psi))
  end

  return diagonal
end

function dfa_accepts(dfa, input)
  state = dfa.initial_state

  for (i, a) in enumerate(input)
    next_state = get(dfa.transitions[i], (state, a), nothing)
    isnothing(next_state) && return false
    state = next_state
  end

  return state in dfa.accepting_states
end


function assert_mpo_matches_dfa(H, dfa, sites)
  diagonal = projection_diagonal(H, sites)

  for bits in keys(diagonal)
    expected = Float64(dfa_accepts(dfa, bits))
    @test diagonal[bits] ≈ expected
  end

  return H
end

function assert_mpo_dimensions(H, dfa, sites)
  for i in eachindex(sites)
    @test ITensors.dim(ITensors.siteind(H, i)) == ITensors.dim(sites[i])
  end

  for i in 1:length(sites) - 1
    @test ITensors.dim(ITensorMPS.linkind(H, i)) == length(dfa.states)
  end

  return H
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

    constraints = [
      SumConstraint([1, 3], [2, 1], Symbol("<="), 2),
      SumConstraint([1, 2, 4], [1, 2, 3], Symbol("=="), 3),
      SumConstraint([2, 4], [2, 3], Symbol(">="), 3),
      SumConstraint([1, 2, 4], [1, 2, 3], Symbol("!="), 3),
    ]

    for constraint in constraints
      dfa = TenSolver.constraint_to_dfa(constraint, sites)

      for bits in all_bitstrings(length(sites))
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

      assert_mpo_dimensions(H, dfa, sites)
      assert_mpo_matches_dfa(H, dfa, sites)
    end
  end

  @testset "Projection MPO" begin
    sites = ITensors.siteinds("Qudit", 4; dim=2)

    constraints = [
      SumConstraint([1, 3], [2, 1], Symbol("<="), 2),
      SumConstraint([1, 2, 4], [1, 2, 3], Symbol("=="), 3),
      SumConstraint([2, 4], [2, 3], Symbol(">="), 3),
      SumConstraint([1, 2, 4], [1, 2, 3], Symbol("!="), 3),
    ]

    for constraint in constraints
      H = TenSolver.projection_mpo(constraint, sites)
      diagonal = projection_diagonal(H, sites)

      for bits in keys(diagonal)
        expected = is_feasible(collect(bits), constraint) ? 1.0 : 0.0
        @test diagonal[bits] ≈ expected
      end
    end
  end
end
