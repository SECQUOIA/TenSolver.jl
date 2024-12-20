using Documenter
using TenSolver

makedocs(
    sitename = "TenSolver.jl",
    format = Documenter.HTML(
      mathengine = Documenter.KaTeX(),
    ),
    modules = [TenSolver]
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
