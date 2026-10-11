import DynamicPolynomials as DP
import TypedPolynomials as TP

@testset "Explicit domain selection" begin
  Q = [2.0 0.0; 0.0 -1.0]
  l = [-3.0, 2.0]
  c = 4.0
  DP.@polyvar x[1:2]
  TP.@polyvar y[1:2]
  p = x[1]^2 - x[2]
  typed_p = y[1]^2 - y[2]

  @testset "Omitted domain fails across native overloads" begin
    for solve in (minimize, maximize)
      for args in ((Q,), (Q, l), (Q, l, c), (Q, c), (l,), (l, c), (p,), (typed_p,))
        err = try
          solve(args...; verbosity = 0)
        catch exception
          exception
        end
        @test err isa UndefKeywordError
        if err isa UndefKeywordError
          @test err.var == :domain
        end
      end
    end
  end

  @testset "Explicit binary domain across forwarding overloads" begin
    Q1 = reshape([2.0], 1, 1)
    l1 = [-3.0]
    # Evaluate each one-variable objective at 0 and 1. These values include
    # the constant term and are independent of domain reduction in the solver.
    fixtures = (
      ((Q1,), 0.0, [0], 2.0, [1]),
      ((Q1, l1), -1.0, [1], 0.0, [0]),
      ((Q1, l1, c), 3.0, [1], 4.0, [0]),
      ((Q1, c), 4.0, [0], 6.0, [1]),
      ((l1,), -3.0, [1], 0.0, [0]),
      ((l1, c), 1.0, [1], 4.0, [0]),
    )
    for (args, emin, xmin, emax, xmax) in fixtures
      for (solve, expected, assignment) in ((minimize, emin, xmin), (maximize, emax, xmax))
        E, psi = solve(args...; domain = 0:1, verbosity = 0)
        @test E ≈ expected
        @test assignment in psi
        @test sample(psi) == assignment
      end
    end
  end

  @testset "Explicit polynomial domains retain reduction and extrema" begin
    DP.@polyvar z
    TP.@polyvar w
    # 2z^3 - 3z + 4 takes values (4, 3) at (0, 1), (5, 3) at
    # (-1, 1), and (-6, 4, 49) at (-2, 0, 3).
    for polynomial in (2z^3 - 3z + 4, 2w^3 - 3w + 4)
      for (domain, emin, xmin, emax, xmax) in (
        (0:1, 3.0, [1], 4.0, [0]),
        ([-1, 1], 3.0, [1], 5.0, [-1]),
        ([-2, 0, 3], -6.0, [-2], 49.0, [3]),
      )
        for (solve, expected, assignment) in ((minimize, emin, xmin), (maximize, emax, xmax))
          E, psi = solve(polynomial; domain, verbosity = 0)
          @test E ≈ expected
          @test assignment in psi
          @test sample(psi) == assignment
        end
      end
    end
  end

  @testset "Explicit heterogeneous domains retain extrema" begin
    # Independent linear terms: 2x1 is least at -1 and greatest at 2;
    # -3x2 is least at 1.5 and greatest at 0. Add the offset 4.
    domain = [[2, -1, 2], [1.5, 0.0, 0.5]]
    for (solve, expected, assignment) in (
      (minimize, -2.5, [-1.0, 1.5]),
      (maximize, 8.0, [2.0, 0.0]),
    )
      E, psi = solve([2.0, -3.0], c; domain, verbosity = 0, noise = 0.0)
      @test E ≈ expected
      @test assignment in psi
      @test sample(psi) == assignment
      @test all(x -> all(x[i] in domain[i] for i in eachindex(x)), sample(psi, 5))
    end
  end
end
