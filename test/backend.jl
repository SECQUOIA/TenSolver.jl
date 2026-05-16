struct TestSymbolBackend <: TenSolver.AbstractTenSolverBackend end

TenSolver._normalize_backend(::Val{:test_symbol_backend}) = TestSymbolBackend()
function TenSolver._minimize(::TestSymbolBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); kwargs...) where {T}
  return T(42), (; Q, l, c, kwargs)
end

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

  @testset "Symbol backends can be provided by extensions" begin
    Q = reshape([1.0], 1, 1)
    E, payload = minimize(Q; backend=:test_symbol_backend, verbosity=0)

    @test E == 42.0
    @test payload.Q === Q
    @test isnothing(payload.l)
    @test payload.c == 0.0
    @test payload.kwargs[:cutoff] == 1e-8
    @test payload.kwargs[:verbosity] == 0
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
