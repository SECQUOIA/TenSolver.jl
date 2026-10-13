using Test
using TenSolver
import DynamicPolynomials as DP
import TypedPolynomials as TP
import MultivariatePolynomials as MP

function assert_preserved_polynomial_solution(
  p,
  vars,
  domains,
  expected_energy,
  objective;
  constraints = TenSolver.AbstractConstraint[],
  expected_sample = nothing,
  solve_domain = domains,
)
  # Only variables effective in the input belong to this regression's contract.
  @test collect(MP.effective_variables(p)) == collect(vars)

  # Check the symbolic fixture against an independently written objective on
  # its small domain box; the expected minimum is derived analytically below.
  for values in Iterators.product(domains...)
    a = collect(values)
    @test p(vars => a) ≈ objective(a)
  end

  E, psi = TenSolver.minimize(
    p;
    domain = solve_domain,
    constraints,
    backend = :dmrg,
    iterations = 4,
    inidim = 2,
    maxdim = 4,
    cutoff = 1e-12,
    noise = 0.0,
    eigsolve_krylovdim = 8,
    verbosity = 0,
  )

  @test E ≈ expected_energy atol = 1e-10
  @test TenSolver.is_feasible(psi)
  for a in TenSolver.sample(psi, 8)
    @test length(a) == length(vars)
    # Guard evaluation after recording a truncated sample as a test failure.
    if !(length(a) == length(vars))
      continue
    end
    @test all(a[i] in domains[i] for i in eachindex(vars))
    @test objective(a) ≈ expected_energy atol = 1e-10
    @test p(vars => a) ≈ E atol = 1e-10
    @test TenSolver.is_feasible(a, constraints)
    if !isnothing(expected_sample)
      @test a == expected_sample
    end
  end
end

function test_polynomial_variable_preservation(x)
  @testset "Objective bounds retain original domain positions" begin
    # The binary root polynomial vanishes; the remaining term is bounded by
    # x2's maximum 4, plus the strict one-unit margin, rather than x1's maximum 1.
    p = 1.0x[1]^2 - x[1] + x[2]
    domains = TenSolver.Domains{Float64}([0:1, [3, 4]], 2)
    reduced = TenSolver.domain_residue(p, domains)
    @test TenSolver.objective_box_bound(reduced, domains, collect(x)) == 5.0
  end

  @testset "Binary variable cancellation" begin
    # x1^2 - x1 vanishes at both binary values; x2 = 1 gives min -2.
    # Either x1 value is valid: samples need not enumerate both optima.
    p = x[1]^2 - x[1] - 2x[2]
    assert_preserved_polynomial_solution(
      p,
      x,
      [0:1, 0:1],
      -2.0,
      a -> a[1]^2 - a[1] - 2a[2];
      solve_domain = 0:1,
    )
  end

  @testset "Heterogeneous domains keep original positions" begin
    # x1 is binary and disappears; x2 is in {2, 4}, so min -8 at x2 = 4.
    # Disjoint local domains detect a shortened or shifted variable mapping.
    p = x[1]^2 - x[1] - 2x[2]
    assert_preserved_polynomial_solution(
      p,
      x,
      [0:1, [2, 4]],
      -8.0,
      a -> a[1]^2 - a[1] - 2a[2],
    )
  end

  @testset "All objective dependence disappears" begin
    # Both binary root polynomials vanish, leaving constant 7. Every original
    # two-variable assignment is optimal; only sample soundness is required.
    p = 7 + x[1]^2 - x[1] + x[2]^2 - x[2]
    assert_preserved_polynomial_solution(
      p,
      x,
      [0:1, 0:1],
      7.0,
      a -> 7 + a[1]^2 - a[1] + a[2]^2 - a[2],
    )
  end

  @testset "Nonbinary elimination retains its generated constant" begin
    # (x1 - 1)(x1 - 2) vanishes on {1, 2}, so x1^2 - 3x1 = -2.
    # The generated constant survives reduction, and x2 = 1 gives min -4.
    p = x[1]^2 - 3x[1] - 2x[2]
    assert_preserved_polynomial_solution(
      p,
      x,
      [[1, 2], 0:1],
      -4.0,
      a -> a[1]^2 - 3a[1] - 2a[2],
    )
  end

  @testset "Constraint retains an objective-independent variable" begin
    # Constraint index 1 still means original x1. Fixing x1 = 0 leaves
    # x2 = 1 optimal at -2, uniquely [0, 1] in the original input order.
    p = x[1]^2 - x[1] - 2x[2]
    constraints = [TenSolver.SumConstraint([1], [1], 0; relation = :(==))]
    assert_preserved_polynomial_solution(
      p,
      x,
      [0:1, 0:1],
      -2.0,
      a -> a[1]^2 - a[1] - 2a[2];
      constraints,
      expected_sample = [0, 1],
    )
  end
end

@testset "Polynomial variables survive domain reduction" begin
  @testset "DynamicPolynomials" begin
    DP.@polyvar x[1:2]
    test_polynomial_variable_preservation(x)
  end
  @testset "TypedPolynomials" begin
    TP.@polyvar x[1:2]
    test_polynomial_variable_preservation(x)
  end
end
