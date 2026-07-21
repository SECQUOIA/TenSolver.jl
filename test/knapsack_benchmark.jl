include(joinpath(@__DIR__, "..", "benchmarks", "knapsack", "KnapsackBenchmark.jl"))
isdefined(@__MODULE__, :brute_force) || include(joinpath(@__DIR__, "utils.jl"))

using .KnapsackBenchmark

function exact_penalty_minimizer(instance, model)
  best_assignment = Int[]
  best_objective = Inf

  for assignment in Iterators.product(fill(0:1, size(model.Q, 1))...)
    x = collect(assignment)
    objective = penalty_value(model, x)
    if objective < best_objective
      best_assignment = x
      best_objective = objective
    end
  end

  return best_assignment, best_objective
end

@testset "Knapsack penalty/projection benchmark" begin
  instance = reference_instance()

  @testset "exact reference and seeded family" begin
    exact = brute_force_optimum(instance)
    @test exact.value == 13.0
    @test exact.weight == 6
    @test exact.items == [1, 0, 1, 0]

    constraint = TenSolver.SumConstraint(
      collect(eachindex(instance.weights)),
      instance.weights,
      instance.capacity;
      relation=:(<=),
    )
    expected_energy, expected_items = brute_force(
      items -> -item_value(instance, items),
      length(instance.weights),
      [constraint],
    )
    @test exact.value == -expected_energy
    @test is_capacity_feasible(instance, expected_items)

    family_a = seeded_family((5, 7); seed=123)
    family_b = seeded_family((5, 7); seed=123)
    @test getfield.(family_a, :weights) == getfield.(family_b, :weights)
    @test getfield.(family_a, :values) == getfield.(family_b, :values)
    @test getfield.(family_a, :capacity) == getfield.(family_b, :capacity)
    @test getfield.(family_a, :weights) != getfield.(seeded_family((5, 7); seed=124), :weights)
  end

  @testset "bounded-binary slack covers exactly the capacity range" begin
    for capacity in 0:12
      encoded = slack_weights(capacity)
      represented = Set(
        sum((bit * weight for (bit, weight) in zip(bits, encoded)); init=0)
        for bits in Iterators.product(fill(0:1, length(encoded))...)
      )
      @test represented == Set(0:capacity)
    end
  end

  @testset "penalty QUBO matches its defining expression" begin
    penalty = 0.2
    model = penalty_qubo(instance, penalty)

    for assignment in Iterators.product(fill(0:1, size(model.Q, 1))...)
      x = collect(assignment)
      items = x[1:model.nitems]
      slack = x[model.nitems + 1:end]
      residual = item_weight(instance, items) + sum(slack .* model.slack_weights) - instance.capacity
      expected = -item_value(instance, items) + penalty * residual^2
      @test penalty_value(model, x) ≈ expected atol=1e-12
    end
  end

  @testset "penalty sensitivity is visible on the reference instance" begin
    weak = penalty_qubo(instance, 0.001sum(instance.values))
    strong = penalty_qubo(instance, 1.1sum(instance.values))
    weak_assignment, _ = exact_penalty_minimizer(instance, weak)
    strong_assignment, _ = exact_penalty_minimizer(instance, strong)
    weak_items = weak_assignment[1:weak.nitems]
    strong_items = strong_assignment[1:strong.nitems]

    @test !is_capacity_feasible(instance, weak_items)
    @test is_capacity_feasible(instance, strong_items)
    @test item_value(instance, strong_items) == 13.0
  end

  @testset "projection sample selection prefers feasibility" begin
    infeasible_high_value = [1, 1, 1, 1]
    feasible_optimum = [1, 0, 1, 0]
    selected = KnapsackBenchmark.best_projection_sample(
      instance,
      [infeasible_high_value, feasible_optimum],
    )
    @test selected == feasible_optimum
  end

  @testset "projection resource metrics expose both projector and effective Hamiltonian" begin
    metrics = projection_resource_metrics(instance; cutoff=1e-12)
    @test metrics.objective_mpo_bond >= 1
    @test metrics.projection_mpo_bond <= instance.capacity + 2
    @test metrics.effective_hamiltonian_bond >= 1
  end

  @testset "controlled projection scaling stays within the capacity-state bound" begin
    rows = projection_scaling_rows(; cutoff=1e-12)
    @test all(row -> row.projection_mpo_bond <= row.capacity_state_bound, rows)

    item_count_bonds = [row.projection_mpo_bond for row in rows if row.sweep == "item_count"]
    weight_magnitude_bonds = [
      row.projection_mpo_bond for row in rows if row.sweep == "weight_magnitude"
    ]
    @test length(unique(item_count_bonds)) == 1
    @test length(unique(weight_magnitude_bonds)) == 1
  end
end
