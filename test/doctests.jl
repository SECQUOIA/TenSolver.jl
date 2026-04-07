using Test
using Documenter
using TenSolver

@testset "Doctests" begin
    docspath = joinpath(dirname(@__FILE__), "..", "docs", "src")
    doctest(docspath, [TenSolver])
end
