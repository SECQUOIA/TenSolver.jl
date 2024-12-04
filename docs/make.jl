using Documenter
using TenSolver

makedocs(
    sitename = "TenSolver",
    format = Documenter.HTML(),
    modules = [TenSolver]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
