using Test, Random, LinearAlgebra

import JuMP

using  TenSolver

# Makes ITensor slower but catches more errors. Good for development.
import ITensors
ITensors.enable_debug_checks()

filepath(x) = joinpath(dirname(@__FILE__), x)

"""
  brute_force(f, n)

A version of `solve` that uses a brute force approach instead of Tensor networks.
Despite being painfully slow, this is useful as a sanity check.
"""
function brute_force(f::Function, T, n::Int64)
  min_now  = +Inf
  solution = Vector{T}[]

  for i in 0:(2^n-1)
    # The Boolean vector corresponding to the natural i
    bs = [convert(T, parse(Bool, d)) for d in last(bitstring(i), n)]
    z  = f(bs)

    if z < min_now
      solution = bs
      min_now  = z
    end

  end

  return min_now, solution
end

@testset "QUBO Correctness" begin
  dim = 5

  @testset "Pure quadratic" begin
    Q = randn(dim, dim)

    # TenSolver solution
    e, psi = TenSolver.solve(Q; cutoff = 1e-16)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test dot(x, Q, x) ≈ e

    for i in 1:10
      y = rand(Bool, dim)
      @test dot(y, Q, y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #

    e0, x0 = brute_force(x -> dot(x, Q, x), Float64, dim)
    # Same minimum value
    @test e ≈ e0
    # Same solution
    @test x == x0
  end

  @testset "Quad+Lin" begin
    Q = 2*randn(dim, dim)
    l = 2*randn(dim)
    c = randn()

    # Objective function
    obj(x) = dot(x, Q, x) + dot(l, x) + c

    # TenSolver solution
    e, psi = TenSolver.solve(Q, l, c; cutoff = 1e-16)
    x = TenSolver.sample(psi)

    # Does the ground energy match solution?
    @test obj(x) ≈ e

    for i in 1:10
      y = rand(Bool, dim)
      @test obj(y) >= e - 1e-8 # A small gap to amount for floating errors
    end

    # ~:~ Exact solution ~:~ #
    e0, x0 = brute_force(obj, Float64, dim)
    # Same minimum value
    @test e ≈ e0
  end
end


@testset "JuMP interface" begin
  dim = 5
  Q = 2*randn(dim, dim)
  l = 2*randn(dim)
  c = randn()
  obj(x) = dot(x, Q, x) + dot(l, x) + c

  m = JuMP.Model(TenSolver.Optimizer)
  @JuMP.variable(m, x[1:dim], Bin)
  @JuMP.objective(m, Min, dot(x, Q, x) + dot(l, x) + c)

  JuMP.optimize!(m)

  e = JuMP.objective_value(m)

  # ~:~ Exact solution ~:~ #
  e0, x0 = brute_force(obj, Float64, dim)
  # Same minimum value
  @test e ≈ e0
end


@testset "Ill-formed input" begin
  dim = 4

  # Should throw when the matrix is not square
  Q = randn(dim, dim - 2)
  @test_throws BoundsError solve(Q)
end

