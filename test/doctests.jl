using Test
using Documenter
using TenSolver

@testset "Doctests" begin
    doctest(TenSolver; manual = ["../docs/src/index.md", "../docs/src/examples.md"])
end
