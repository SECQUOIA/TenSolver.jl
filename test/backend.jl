struct TestSymbolBackend <: TenSolver.AbstractTenSolverBackend end
struct MissingMethodBackend <: TenSolver.AbstractTenSolverBackend end

TenSolver.normalize_backend(::Val{:test_symbol_backend}) = TestSymbolBackend()
function TenSolver.minimize(::TestSymbolBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); kwargs...) where {T}
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

  @test TenSolver.GTNBackend() isa TenSolver.AbstractTenSolverBackend
  @test :GTNBackend ∉ names(TenSolver)
  @test !isdefined(TenSolver, :solution_space)

  @testset "Maximize forwards backend selection" begin
    Q = reshape([2.0], 1, 1)
    E, psi = maximize(Q; backend=:dmrg, verbosity=0)

    @test E ≈ 2.0
    @test TenSolver.sample(psi) == [1]
  end

  @testset "Symbol backends can be provided by extensions" begin
    Q = reshape([1.0], 1, 1)
    E, payload = minimize(Q; backend=:test_symbol_backend, verbosity=0, cutoff=1e-6)

    @test E == 42.0
    @test payload.Q === Q
    @test iszero(payload.l)
    @test iszero(payload.c)
    @test payload.kwargs[:cutoff] == 1e-6
    @test payload.kwargs[:verbosity] == 0
    @test !haskey(payload.kwargs, :preprocess)
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
    @test occursin("PEPS extension", sprint(showerror, peps_error))
    @test occursin("backend = :dmrg", sprint(showerror, peps_error))

    gtn_error = try
      minimize(Q; backend=:gtn)
    catch err
      err
    end
    @test gtn_error isa ArgumentError
    @test occursin("requires GenericTensorNetworks and ProblemReductions", sprint(showerror, gtn_error))

    unknown_symbol_error = try
      minimize(Q; backend=:foo, verbosity=0)
    catch err
      err
    end
    @test unknown_symbol_error isa ArgumentError
    @test occursin("No backend-specific `minimize` method is available for backend :foo", sprint(showerror, unknown_symbol_error))
    @test !occursin("PEPS extension", sprint(showerror, unknown_symbol_error))

    unsupported_error = try
      minimize(Q; backend="dmrg", verbosity=0)
    catch err
      err
    end
    @test unsupported_error isa ArgumentError
    @test occursin("No backend-specific `minimize` method is available for backend \"dmrg\"", sprint(showerror, unsupported_error))

    missing_method_error = try
      minimize(Q; backend=MissingMethodBackend(), verbosity=0)
    catch err
      err
    end
    @test missing_method_error isa ArgumentError
    @test occursin("No backend-specific `minimize` method is available for backend", sprint(showerror, missing_method_error))
    @test occursin("MissingMethodBackend", sprint(showerror, missing_method_error))
  end
end
