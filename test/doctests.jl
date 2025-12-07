using Test
using Documenter
using TenSolver

@testset "Doctests" begin
    docspath = joinpath(dirname(@__FILE__), "..", "docs", "src")
    doctest(TenSolver; manual = [
        joinpath(docspath, "index.md"),
        joinpath(docspath, "examples.md")
    ])
end
