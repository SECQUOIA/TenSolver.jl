@testset "Backend interface" begin
  @testset "DMRG is the default backend" begin
    Q = reshape([-2.0], 1, 1)

    default_energy, default_solution = minimize(Q; verbosity=0)
    object_energy, object_solution = minimize(Q; backend=DMRGBackend(), verbosity=0)
    symbol_energy, symbol_solution = minimize(Q; backend=:dmrg, verbosity=0)

    @test default_energy ≈ -2.0
    @test object_energy ≈ default_energy
    @test symbol_energy ≈ default_energy
    @test TenSolver.sample(default_solution) == [1]
    @test TenSolver.sample(object_solution) == [1]
    @test TenSolver.sample(symbol_solution) == [1]
  end

  @testset "Maximize forwards backend selection" begin
    Q = reshape([2.0], 1, 1)
    E, psi = maximize(Q; backend=:dmrg, verbosity=0)

    @test E ≈ 2.0
    @test TenSolver.sample(psi) == [1]
  end

  @testset "Unavailable backends error clearly" begin
    Q = reshape([1.0], 1, 1)

    peps_error = try
      minimize(Q; backend=:peps, verbosity=0)
    catch err
      err
    end
    @test peps_error isa ArgumentError
    @test occursin("backend :peps is not available", sprint(showerror, peps_error))
    @test occursin("backend = :dmrg", sprint(showerror, peps_error))

    unsupported_error = try
      minimize(Q; backend="dmrg", verbosity=0)
    catch err
      err
    end
    @test unsupported_error isa ArgumentError
    @test occursin("Unsupported backend", sprint(showerror, unsupported_error))
  end
end
