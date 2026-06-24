"""
    DMRGBackend()

Select TenSolver's default ITensorMPS DMRG backend.
"""
struct DMRGBackend <: AbstractTenSolverBackend end

const default_backend = DMRGBackend()

normalize_backend(::Val{:dmrg}) = default_backend

function minimize(::DMRGBackend, Q::AbstractMatrix{T}, l::Union{AbstractVector{T}, Nothing}=nothing, c::T=zero(T); cutoff=1e-8, preprocess::Bool=false, kwargs...) where T
  Qp, lp, permutation = preprocess ? preprocess_qubo(Q, l, cutoff) : (Q, l, collect(1:size(Q, 1)))
  H      = tensorize(Qp, isnothing(lp) ? diag(Qp) : diag(Qp) + lp; cutoff)
  obj(x) = dot(x, Q, x) + c + maybe(l -> dot(l,x), l; default=zero(T))

  return _minimize_mpo(H, c, obj; cutoff, permutation, kwargs...)
end

function minimize(::DMRGBackend, p::AbstractPolynomial{T}; cutoff=1e-8, kwargs...) where T
  H   = tensorize(p)
  cte = constant(p)
  vs = variables(p)
  return _minimize_mpo(H, cte, a -> p(vs => a); cutoff, kwargs...)
end
