@testset "Constraints" begin
  @testset "Constructors" begin
    sum = sum_constraint([1, 3], [2, 1], Symbol("<="), 2)
    @test sum isa SumConstraint
    @test sum isa AbstractConstraint
    @test sum.sites == [1, 3]
    @test sum.weights == [2, 1]
    @test sum.relation == Symbol("<=")
    @test sum.rhs == 2

    default_sum = sum_constraint([1, 2], [1.5, 0.5], 2.0)
    @test default_sum.relation == Symbol("==")
    @test default_sum.weights == [1.5, 0.5]
    @test default_sum.rhs == 2.0

    not_equals = not_equals_constraint([1, 2], [1, 0])
    @test not_equals isa NotEqualsConstraint
    @test not_equals isa AbstractConstraint
    @test not_equals.sites == [1, 2]
    @test not_equals.values == [1, 0]

    exactly_one = exactly_one_constraint(2:4)
    @test exactly_one isa ExactlyOneConstraint
    @test exactly_one isa AbstractConstraint
    @test exactly_one.sites == [2, 3, 4]

    relation = relation_constraint(1, Symbol(">="), 2)
    @test relation isa RelationConstraint
    @test relation isa AbstractConstraint
    @test relation.left_site == 1
    @test relation.relation == Symbol(">=")
    @test relation.right_site == 2
  end

  @testset "Constructor validation" begin
    @test_throws ArgumentError exactly_one_constraint(Int[])
    @test_throws ArgumentError exactly_one_constraint([0])
    @test_throws ArgumentError exactly_one_constraint([1, 1])
    @test_throws ArgumentError exactly_one_constraint([1.0])

    @test_throws DimensionMismatch sum_constraint([1, 2], [1], Symbol("=="), 1)
    @test_throws ArgumentError sum_constraint([1], [-1], Symbol("=="), 0)
    @test_throws ArgumentError sum_constraint([1], [1], Symbol("~"), 0)
    @test_throws ArgumentError sum_constraint([1], [1], "==", 0)
    # A relation passed positionally to the three-argument form is rejected
    # rather than silently treated as `rhs`.
    @test_throws ArgumentError sum_constraint([1, 2], [1, 1], Symbol("<="))

    @test_throws DimensionMismatch not_equals_constraint([1, 2], [1])
    @test_throws ArgumentError not_equals_constraint([1], [2])

    @test_throws ArgumentError relation_constraint(0, Symbol("=="), 1)
    @test_throws ArgumentError relation_constraint(1, Symbol("=="), 1)
    @test_throws ArgumentError relation_constraint(1, Symbol("~"), 2)
    @test_throws ArgumentError relation_constraint(1, "==", 2)
  end

  @testset "Feasibility" begin
    sum_le = sum_constraint([1, 2, 3], [2, 1, 1], Symbol("<="), 3)
    @test is_feasible([1, 0, 1], sum_le)
    @test !is_feasible([1, 1, 1], sum_le)

    sum_eq = sum_constraint([1, 2], [1, 1], 1)
    @test is_feasible([1, 0], sum_eq)
    @test !is_feasible([1, 1], sum_eq)

    not_equals = not_equals_constraint([1, 3], [1, 0])
    @test !is_feasible([1, 1, 0], not_equals)
    @test is_feasible([1, 1, 1], not_equals)

    exactly_one = exactly_one_constraint([2, 3, 4])
    @test is_feasible([0, 1, 0, 0], exactly_one)
    @test !is_feasible([0, 1, 1, 0], exactly_one)

    relation = relation_constraint(1, Symbol("<="), 2)
    @test is_feasible([0, 1], relation)
    @test !is_feasible([1, 0], relation)

    constraints = AbstractConstraint[
      sum_constraint([1, 2], [1, 1], 1),
      relation_constraint(1, Symbol(">="), 2),
    ]
    @test is_feasible([1, 0], constraints)
    @test !is_feasible([0, 1], constraints)
    @test is_feasible([1, 0], AbstractConstraint[])

    @test_throws ArgumentError is_feasible([0, 2], exactly_one)
    @test_throws BoundsError is_feasible([1], relation_constraint(1, Symbol("=="), 2))
  end
end
