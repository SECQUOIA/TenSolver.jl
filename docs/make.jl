using Documenter
using TenSolver

makedocs(
    sitename = "TenSolver.jl",
    format = Documenter.HTML(
        mathengine = Documenter.KaTeX(),
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://SECQUOIA.github.io/TenSolver.jl",
        repolink = "https://github.com/SECQUOIA/TenSolver.jl",
        assets = String[],
    ),
    modules = [TenSolver],
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "SpinGlassPEPS Integration" => "spinglasspeps_integration.md",
        "API Reference" => "api.md",
    ],
    repo = Documenter.Remotes.GitHub("SECQUOIA", "TenSolver.jl"),
    authors = "Iago Leal de Freitas, David E. Bernal Neira",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
# Deploy site to Github Pages
if !("local" in ARGS)
    deploydocs(
        repo = "github.com/SECQUOIA/TenSolver.jl.git",
        devbranch = "main",
        push_preview = true,
    )
end
