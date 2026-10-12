using Test, TenSolver
import ITensors

function test_device_solve(device, on_device; atol = 1e-8)
  Q = [0.0 2.0 0.0; 0.0 0.0 -1.0; 0.0 0.0 0.0]
  l = [3.0, 1.0, 2.0]
  exactly_two = SumConstraint([1, 2, 3], [1, 1, 1], 2; relation = :(==))
  force_one = SumConstraint([1], [1], 1; relation = :(==))
  # With exactly two bits set, the costs of 110, 101, 011 are 6, 5, 2.
  # The unconstrained optimum 000 has cost zero and is forbidden.
  cases = [
    ("unconstrained", Q, l, AbstractConstraint[], 0.0, [0, 0, 0]),
    ("native constraints", Q, l, [exactly_two], 2.0, [0, 1, 1]),
    ("noise and variance", Q, l, [exactly_two], 2.0, [0, 1, 1]),
    (
      "multiple constraints",
      Q,
      l,
      [exactly_two, SumConstraint([1], [1], 0; relation = :(==))],
      2.0,
      [0, 1, 1],
    ),
    ("zero objective", zeros(3, 3), zeros(3), [exactly_two], 0.0, nothing),
    (
      "infeasible",
      Q,
      l,
      [SumConstraint([1, 2, 3], [1, 1, 1], 4; relation = :(==))],
      Inf,
      nothing,
    ),
    ("single site", zeros(1, 1), [3.0], [force_one], 3.0, [1]),
    ("single-site zero objective", zeros(1, 1), [0.0], [force_one], 0.0, [1]),
    (
      "single-site infeasible",
      zeros(1, 1),
      [3.0],
      [SumConstraint([1], [1], 2; relation = :(==))],
      Inf,
      nothing,
    ),
    (
      "single-site unconstrained tie",
      zeros(1, 1),
      [0.0],
      AbstractConstraint[],
      0.0,
      nothing,
    ),
  ]

  @testset "$name" for (name, Q, l, constraints, optimum, expected) in cases
    callbacks = Ref(0)
    callback(psi; kwargs...) = begin
      @test all(on_device, psi)
      callbacks[] += 1
    end
    kwargs = (;
      domain = 0:1,
      constraints,
      device,
      iterations = 3,
      inidim = 8,
      maxdim = 8,
      cutoff = 1e-12,
      verbosity = 0,
      on_iteration = callback,
    )
    kwargs = if name == "noise and variance"
      # Use the backend's default noise schedule and evaluate variance each sweep.
      (; kwargs..., check_variance_every_iteration = 1, vtol = -Inf)
    else
      (; kwargs..., noise = [0.0])
    end
    if isinf(optimum)
      energy, solution =
        @test_logs (:warn, r"empty feasible subspace") minimize(Q, l; kwargs...)
      @test energy == Inf
      @test !is_feasible(solution)
      @test isnothing(solution.tensor)
      @test isempty(solution.stats.energies)
      @test callbacks[] == 0
      @test_throws DomainError TenSolver.sample(solution)
    else
      energy, solution = minimize(Q, l; kwargs...)
      @test energy ≈ optimum atol = atol
      @test is_feasible(solution)
      @test all(on_device, solution.tensor)
      @test callbacks[] == (length(l) == 1 ? 1 : 3)
      if name == "noise and variance"
        @test length(solution.stats.variances) == 3
        @test all(v -> !isnothing(v) && isfinite(v), solution.stats.variances)
      end
      for x in TenSolver.sample(solution, 8)
        @test is_feasible(x, constraints)
        @test isnothing(expected) || x == expected
      end
      if name == "single-site unconstrained tie"
        amplitudes = Array(ITensors.array(only(solution.tensor)))
        @test abs2.(amplitudes) ≈ [0.5, 0.5] atol = atol
      end
    end
  end
end
