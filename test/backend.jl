struct TestSymbolBackend <: TenSolver.AbstractTenSolverBackend end
struct MissingMethodBackend <: TenSolver.AbstractTenSolverBackend end
struct TestPolynomialBackend <: TenSolver.AbstractTenSolverBackend end

import DynamicPolynomials as DP
import MultivariatePolynomials as MP

function TenSolver.minimize(
  ::TestPolynomialBackend,
  p::MP.AbstractPolynomial;
  domain,
  constraints,
  verbosity,
)
  return 42.0, (; p, domain, constraints, verbosity)
end

TenSolver.normalize_backend(::Val{:test_symbol_backend}) = TestSymbolBackend()
function TenSolver.minimize(::TestSymbolBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); kwargs...) where {T}
  return T(42), (; Q, l, c, kwargs)
end

@testset "Backend interface" begin
  @testset "Existing polynomial methods retain their keyword contract" begin
    DP.@polyvar x[1:2]
    p = x[1]^2 + x[2]
    E, payload = minimize(p; backend = TestPolynomialBackend(), domain = 0:1, verbosity = 0)
    @test E == 42.0
    @test payload.p == x[1] + x[2]
    @test length(payload.domain) == 2
    @test isempty(payload.constraints)
    @test payload.verbosity == 0

    # A backend must explicitly support the original register before receiving
    # an objective whose reduced dependency set would truncate assignments.
    err = try
      minimize(
        x[1]^2 - x[1] - 2x[2];
        backend = TestPolynomialBackend(),
        domain = 0:1,
        verbosity = 0,
      )
    catch caught
      caught
    end
    @test err isa ArgumentError
    @test occursin("domain reduction that removes variables", sprint(showerror, err))
  end

  @testset "Maximize forwards backend selection" begin
    # The three-valued domain keeps the diagonal quadratic term in Q instead
    # of simplifying it into the linear term before it reaches the backend.
    Q = reshape([2.0], 1, 1)
    E, payload = maximize(
      Q;
      backend=:test_symbol_backend,
      domain=0:2,
      verbosity=0,
    )

    @test E == -42.0
    @test payload.Q == reshape([-2.0], 1, 1)
    @test payload.l == [0.0]
    @test payload.c == 0.0
    @test payload.kwargs[:verbosity] == 0
  end

  @testset "Symbol backends can be provided by extensions" begin
    Q = reshape([1.0], 1, 1)
    E, payload = minimize(Q; domain = 0:1, backend=:test_symbol_backend, verbosity=0, cutoff=1e-6)

    @test E == 42.0
    @test payload.kwargs[:cutoff] == 1e-6
    @test payload.kwargs[:verbosity] == 0
    @test !haskey(payload.kwargs, :preprocess)
  end

  @testset "Unavailable backends error clearly" begin
    Q = reshape([1.0], 1, 1)

    peps_error = try
      minimize(Q; domain = 0:1, backend=:peps, verbosity=0)
    catch err
      err
    end
    @test peps_error isa ArgumentError
    @test occursin("backend :peps is not available", sprint(showerror, peps_error))
    @test occursin("PEPS extension", sprint(showerror, peps_error))
    @test occursin("backend = :dmrg", sprint(showerror, peps_error))

    unknown_symbol_error = try
      minimize(Q; domain = 0:1, backend=:foo, verbosity=0)
    catch err
      err
    end
    @test unknown_symbol_error isa ArgumentError
    @test occursin("No backend-specific `minimize` method is available for backend :foo", sprint(showerror, unknown_symbol_error))
    @test !occursin("PEPS extension", sprint(showerror, unknown_symbol_error))

    unsupported_error = try
      minimize(Q; domain = 0:1, backend="dmrg", verbosity=0)
    catch err
      err
    end
    @test unsupported_error isa ArgumentError
    @test occursin("No backend-specific `minimize` method is available for backend \"dmrg\"", sprint(showerror, unsupported_error))

    missing_method_error = try
      minimize(Q; domain = 0:1, backend=MissingMethodBackend(), verbosity=0)
    catch err
      err
    end
    @test missing_method_error isa ArgumentError
    @test occursin("No backend-specific `minimize` method is available for backend", sprint(showerror, missing_method_error))
    @test occursin("MissingMethodBackend", sprint(showerror, missing_method_error))
  end
end
