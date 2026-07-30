@testset "Constraints" begin
  @testset "Constructors" begin
    sum = SumConstraint([1, 3], [2, 1], Symbol("<="), 2)
    @test sum isa SumConstraint
    @test Set(TenSolver.constraint_sites(sum)) == Set([1, 3])
    @test sum.weights == Dict(1 => 2, 3 => 1)
    @test sum.relation == Symbol("<=")
    @test sum.rhs == 2

    keyword_sum = SumConstraint([1, 2], [1.0, 5.0], 2; relation=Symbol("=="))
    @test keyword_sum.relation == Symbol("==")
    @test keyword_sum.weights == Dict(1 => 1.0, 2 => 5.0)
    @test keyword_sum.rhs == 2.0

    not_equals = NotEqualsConstraint([1, 2], [1, 0])
    @test not_equals isa NotEqualsConstraint{Int}
    @test Set(TenSolver.constraint_sites(not_equals)) == Set([1, 2])
    @test not_equals.values == Dict(1 => 1, 2 => 0)

    float_not_equals = NotEqualsConstraint([1, 2], [1.0, 0.0])
    @test float_not_equals isa NotEqualsConstraint{Float64}
    @test Set(TenSolver.constraint_sites(float_not_equals)) == Set([1, 2])
    @test float_not_equals.values == Dict(1 => 1.0, 2 => 0.0)

    exactly_one = AssignmentConstraint(2:4, [1], :(==), 1)
    @test exactly_one isa AssignmentConstraint{Int}
    @test exactly_one.sites == [2, 3, 4]
    @test exactly_one.values == Set([1])

    assign_two = AssignmentConstraint(2:4, [1, 2, 3], :(<=), 2)
    @test assign_two isa AssignmentConstraint{Int}
    @test assign_two.sites == [2, 3, 4]
    @test assign_two.values == Set([1, 2, 3])
    @test assign_two.rhs == 2

    bool_assignment = AssignmentConstraint(1:3, Bool[true], :(==), 2)
    @test bool_assignment isa AssignmentConstraint{Bool}
    @test bool_assignment.values == Set(Bool[true])
    @test bool_assignment.rhs == 2

    narrow_assignment = AssignmentConstraint([1], Int8[1], :(==), 128)
    @test narrow_assignment isa AssignmentConstraint{Int8}
    @test narrow_assignment.values == Set(Int8[1])
    @test narrow_assignment.rhs == 128

    relation = RelationConstraint(1, Symbol(">="), 2)
    @test relation isa RelationConstraint
    @test relation isa AbstractConstraint
    @test relation.left_site == 1
    @test relation.relation == Symbol(">=")
    @test relation.right_site == 2
  end

  @testset "Constructor validation" begin
    @test_throws ArgumentError AssignmentConstraint(Int[], 1, :(==), 1)
    @test_throws ArgumentError AssignmentConstraint([0], 1, :(==), 1)
    @test_throws ArgumentError AssignmentConstraint([1, 1], 1, :(==), 1)
    @test_throws ArgumentError AssignmentConstraint([1.0], 1, :(==), 1)
    @test_throws ArgumentError AssignmentConstraint([1], [1], :(==), -1)
    @test_throws ArgumentError AssignmentConstraint([1], [1], :(==), 1.5)
    @test_throws ArgumentError AssignmentConstraint([1], [1], :(<), 1)

    @test_throws DimensionMismatch SumConstraint([1, 2], [1], Symbol("=="), 1)
    @test_throws ArgumentError SumConstraint([1], [-1], Symbol("=="), 0)
    @test_throws ArgumentError SumConstraint([1], [1], Symbol("~"), 0)
    @test_throws ArgumentError SumConstraint([1], [1], Symbol("<"), 0)
    @test_throws ArgumentError SumConstraint([1], [1], "==", 0)
    @test_throws UndefKeywordError SumConstraint([1, 2], [1, 1], 1)

    @testset "SumConstraint floating-point validation" begin
      @test SumConstraint([1, 2], [1.0, 2.0], 2.0; relation=:(<=)) isa SumConstraint
      @test SumConstraint([1, 2], [1.0, 1.0], 1.0; relation=:(==)) isa SumConstraint

      @test_throws ArgumentError SumConstraint([1, 2], [1.5, 1.0], 2.0; relation=:(<=))
      @test_throws ArgumentError SumConstraint([1, 2], [1.0, 1.0], 1.5; relation=:(<=))
      @test_throws ArgumentError SumConstraint([1, 2], [1.0, -1.0], 0.0; relation=:(<=))
    end

    @test_throws DimensionMismatch NotEqualsConstraint([1, 2], [1])

    @test_throws ArgumentError RelationConstraint(0, Symbol("=="), 1)
    @test_throws ArgumentError RelationConstraint(1, Symbol("=="), 1)
    @test_throws ArgumentError RelationConstraint(1, Symbol("~"), 2)
    @test_throws ArgumentError RelationConstraint(1, Symbol(">"), 2)
    @test_throws ArgumentError RelationConstraint(1, "==", 2)
  end

  @testset "Feasibility" begin
    sum_le = SumConstraint([1, 2, 3], [2, 1, 1], Symbol("<="), 3)
    @test is_feasible([1, 0, 1], sum_le)
    @test !is_feasible([1, 1, 1], sum_le)

    sum_eq = SumConstraint([1, 2], [1, 1], 1; relation=Symbol("=="))
    @test is_feasible([1, 0], sum_eq)
    @test !is_feasible([1, 1], sum_eq)

    sum_mod = SumModConstraint([1, 2], [1, 1], 1; mod=2)
    @test  is_feasible([1, 0], sum_mod)
    @test !is_feasible([1, 1], sum_mod)

    not_equals = NotEqualsConstraint([1, 3], [1, 0])
    @test !is_feasible([1, 1, 0], not_equals)
    @test is_feasible([1, 1, 1], not_equals)

    exactly_one = AssignmentConstraint(2:4, [1], :(==), 1)
    @test is_feasible([0, 1, 0, 0], exactly_one)
    @test !is_feasible([0, 1, 1, 0], exactly_one)

    assign_two = AssignmentConstraint(2:4, [1, 2, 3], :(<=), 2)
    @test is_feasible([0, 2, 0, 0], assign_two)
    @test !is_feasible([0, 2, 1, 2], assign_two)

    relation = RelationConstraint(1, Symbol("<="), 2)
    @test is_feasible([0, 1], relation)
    @test !is_feasible([1, 0], relation)

    constraints = AbstractConstraint[
      SumConstraint([1, 2], [1, 1], 1; relation=Symbol("==")),
      RelationConstraint(1, Symbol(">="), 2),
    ]
    @test is_feasible([1, 0], constraints)
    @test !is_feasible([0, 1], constraints)
    @test is_feasible([1, 0], AbstractConstraint[])

    mixed_constraints = AbstractConstraint[
      AssignmentConstraint([1, 2], 0, :(==), 1),
      RelationConstraint(1, Symbol(">="), 2),
    ]
    @test is_feasible([1, 0], mixed_constraints)
    @test !is_feasible([0, 1], mixed_constraints)
    @test !is_feasible([1, 1], mixed_constraints)

    @test_throws BoundsError is_feasible([0, 2], exactly_one)
    @test_throws BoundsError is_feasible([1], RelationConstraint(1, Symbol("=="), 2))
  end
end
