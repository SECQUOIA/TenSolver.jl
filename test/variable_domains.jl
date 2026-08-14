import DynamicPolynomials as DP
import TenSolver as TS

@testset "Variable Domains" begin
  @testset "Domains Array Interface" begin
    dom = TS.Domains{Float64}([[0, 1], [-1, 1], [-3, 4, 5.7]], 3)

    @test eltype(dom) === Float64

    @test [0, -1, 5.7] in dom
    @test_throws DimensionMismatch [0, 1] in dom
  end

  @testset "Domain Simplification" begin
    domains = map(d -> TS.Domains{Float64}(d, 3), [
      [0, 1],
      [-1, 1],
      [-3, 5.7, 4],
      [[0, 1], [-1, 1], [-3, 4, 5.7]],
     ])

    @testset "Quadratic Objectives" begin
      Q, l, c = TS.domain_residue([1 0; 0 1.0], [0.0, 0.0], 0.0, TS.Domains{Float64}([-1, 1], 2))
      @test iszero(Q)
      @test iszero(l)
      @test c ≈ 2

      for domain in domains
        Q, l, c = randn(3, 3), randn(3), randn()
        Qr, lr, cr = TS.domain_residue(Q, l, c, domain)
        obj(x) = dot(x, Q, x) + dot(l, x) + c
        obj_r(x) = dot(x, Qr, x) + dot(lr, x) + cr

        if all(length.(domain) .== 2)
          @test iszero(Diagonal(Qr))
        end

        for x in domain[1], y in domain[2], z in domain[3]
          @test obj([x, y, z]) ≈ obj_r([x, y, z])
        end
      end
    end

    @testset "Polynomial Objectives" begin
      DP.@polyvar x[1:3]

      for (deg, domain) in Iterators.product([2, 3, 5], domains)
        p = randpoly(x, deg)
        q = TS.domain_residue(p, domain)

        for v in DP.effective_variables(q)
          @test DP.maxdegree(q, v) <= maximum(length, domain)
        end

        for a in domain[1], b in domain[2], d in domain[3]
          @test p(a, b, d) ≈ q(a, b, d)
        end
      end
    end
  end
end
