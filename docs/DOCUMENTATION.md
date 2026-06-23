# Documentation Guide for TenSolver.jl

This document explains how the documentation for TenSolver.jl is set up and how to work with it.

## Structure

The documentation is built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and consists of the following pages:

- **Home** (`docs/src/index.md`): Overview, installation, quick start, and features
- **Examples** (`docs/src/examples.md`): Practical examples showing how to use TenSolver.jl
- **SpinGlassPEPS Integration** (`docs/src/spinglasspeps_integration.md`): Planned architecture for the optional structured PEPS backend
- **API Reference** (`docs/src/api.md`): Complete API documentation with docstrings

Internal developer notes live outside the generated user documentation:

- **SpinGlassPEPS Integration** (`docs/internal/spinglasspeps_integration.md`): Planned architecture for the optional structured PEPS backend

## Building Documentation Locally

To build the documentation locally:

```bash
# Install documentation dependencies
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

# Build documentation (without deployment)
julia --project=docs/ docs/make.jl local
```

The built documentation will be in `docs/build/index.html`.

## GitHub Actions Workflow

The documentation is automatically built and deployed via GitHub Actions:

### On Pull Requests
- Documentation is built and tested to ensure it compiles without errors
- No deployment occurs (documentation is only built for validation)
- Helps catch documentation errors early in the development process

### On Merges to Main
- Documentation is built and deployed to the `gh-pages` branch
- Published at: https://SECQUOIA.github.io/TenSolver.jl/dev/

### On Tags
- Documentation is built and deployed as a stable version
- Published at: https://SECQUOIA.github.io/TenSolver.jl/stable/

## Adding Documentation

### Adding Docstrings

Add docstrings to functions in the source code using Julia's docstring syntax:

```julia
"""
    function_name(args)

Brief description.

## Arguments
- `arg1`: Description of arg1
- `arg2`: Description of arg2

## Returns
- Description of return value

## Examples
```jldoctest
julia> function_name(1, 2)
3
```
"""
function function_name(arg1, arg2)
    # implementation
end
```

### Adding New Documentation Pages

1. Create a new `.md` file in `docs/src/`
2. Add the page to the `pages` array in `docs/make.jl`
3. Update the table of contents in `docs/src/index.md` if needed

### Adding Internal Design Notes

Use `docs/internal/` for implementation plans, architecture notes, and staged PR
coordination that are useful to maintainers but should not appear in the
published user documentation. Promote an internal note to `docs/src/` only when
it describes behavior that is available to users.

## Deployment Configuration

The deployment is configured via `deploydocs()` in `docs/make.jl`:

- Uses SSH deploy key (set via `DOCUMENTER_KEY` secret in GitHub)
- Deploys to the `gh-pages` branch
- Creates versioned documentation for tags
- Creates dev documentation from the main branch
- Supports preview deployments for PRs (when enabled)

## Troubleshooting

### Documentation Build Fails in CI

1. Check the GitHub Actions logs for error messages
2. Try building locally with the same Julia version
3. Ensure all docstrings are properly formatted
4. Verify that all cross-references are valid

### Documentation Not Deploying

1. Verify `DOCUMENTER_KEY` secret is set correctly in repository settings
2. Check that the `gh-pages` branch exists
3. Ensure GitHub Pages is enabled in repository settings
4. Check the GitHub Actions logs for deployment errors
