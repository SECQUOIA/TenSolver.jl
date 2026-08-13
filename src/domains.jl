"""
    Domains{T}

Represent a finite domain for each variable in an optimization problem.
"""
struct Domains{T<:Real} <: AbstractVector{Vector{T}}
  ds :: Vector{Vector{T}}

  function Domains(dom::AbstractVector{<:Real}, nvariables::Integer)
    dom = canonicalize_variable_domain(dom)
    T = float(eltype(dom))
    return new{T}(fill(dom, nvariables))
  end

  function Domains(ds::AbstractVector{<:AbstractVector}, nvariables::Integer)
    @argcheck length(ds) == nvariables
    ds = canonicalize_variable_domain.(ds)
    T  = float(promote_type(map(eltype, ds)...))
    return new{T}(ds)
  end
end

function Domains(dom::Domains, nvariables::Integer)
  @argcheck length(dom) == nvariables
  return dom
end

function canonicalize_variable_domain(vdomain)
  @argcheck applicable(length, vdomain)
  @argcheck (!isempty)(vdomain)
  @argcheck eltype(vdomain) <: Real

  # Preprocessing to dedeplicate domain values
  return (ismutable(vdomain) ? unique! : unique)(sort(vdomain))
end


#=====================================================================#
# Abstract Array Interface                                            #
#=====================================================================#

Base.size(domains::Domains) = (length(domains.ds),)

Base.IndexStyle(::Type{<:Domains}) = IndexLinear()

Base.getindex(domains::Domains, i::Int) = domains.ds[i]

#=====================================================================#
# Other Interfaces                                                    #
#=====================================================================#

Base.in(x, dom::Domains) = all(insorted.(x, dom))

Base.eltype(dom::Domains{T})  where T = T
Base.valtype(dom::Domains{T}) where T = T

function Base.permute!(dom::Domains, permutation::AbstractVector)
  @argcheck length(dom) == length(permutation)
  dom.ds .= dom.ds[permutation]
  return dom
end

Random.Sampler(::Type{<:AbstractRNG}, dom::Domains, ::Repetition) = SamplerTrivial(dom)

function Random.rand(rng::AbstractRNG, sp::SamplerTrivial{<:Domains})
  return [rand(rng, d) for d in sp[]]
end
