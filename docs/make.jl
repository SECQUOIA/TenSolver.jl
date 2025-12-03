using Documenter
using TenSolver

makedocs(
    sitename = "TenSolver.jl",
    format = Documenter.HTML(
        mathengine = Documenter.KaTeX(),
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://SECQUOIA.github.io/TenSolver.jl",
        assets = String[],
    ),
    modules = [TenSolver],
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    repo = "https://github.com/SECQUOIA/TenSolver.jl/blob/{commit}{path}#{line}",
    authors = "Iago Leal de Freitas, David E. Bernal Neira",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
# Deploy site to Github Pages
if !("local" in ARGS)
    deploydocs(
        repo = "github.com/SECQUOIA/TenSolver.jl.git"
    )
end
