@testset "Fractional domains" begin
  @testset "Unconstrained objective" begin
    domain = [0.0, 0.5, 1.0]
    l = [-2.0, 3.0]

    E, psi = minimize(l; domain, verbosity=0)
    x = sample(psi)

    @test E ≈ -2.0
    @test x == [1.0, 0.0]
    @test psi.domain == domain
    @test x in psi
  end

  @testset "SumConstraint fails fast" begin
    capacity = SumConstraint([1], [1], 1; relation=:(<=))
    message = try
      minimize([0.0]; domain=[0.0, 0.5, 1.0], constraints=[capacity], verbosity=0)
      ""
    catch error
      @test error isa ArgumentError
      sprint(showerror, error)
    end

    @test message ==
      "ArgumentError: SumConstraint only supports nonnegative integer domains."
  end
end
