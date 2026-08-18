"""
    Domains{T}

Represent a finite domain for each variable in an optimization problem.

Indexing or Iterating over it yields the local variable domains,
while "wholesale" operations such as `in` or `rand`
treat it as a collection of possible values for a variable.
"""
struct Domains{T<:Real}
  ds :: Vector{Vector{T}}

  function Domains{T}(dom::AbstractVector{<:Number}, nvariables::Integer) where T
    dom = canonicalize_variable_domain(T, dom)
    return new{T}(fill(dom, nvariables))
  end

  function Domains{T}(ds::AbstractVector{<:AbstractVector}, nvariables::Integer) where T
    @argcheck length(ds) == nvariables
    ds = canonicalize_variable_domain.(T, ds)
    return new{T}(ds)
  end
end

function Domains{T}(dom::Domains, nvariables::Integer) where T
  return Domains{T}(dom.ds, nvariables)
end

function canonicalize_variable_domain(T::Type, vdomain)
  vdomain = convert(Vector{T}, vdomain)

  @argcheck (!isempty)(vdomain)
  @argcheck eltype(vdomain) <: Real

  # Preprocessing to dedeplicate domain values
  return unique!(sort!(vdomain))
end


#=====================================================================#
# Abstract Array Interface                                            #
#=====================================================================#

Base.length(dom::Domains) = length(dom.ds)
Base.getindex(dom::Domains, i::Int) = dom.ds[i]

Base.size(dom::Domains) = (length(dom),)
Base.axes(dom::Domains) = map(Base.OneTo, size(dom))

Base.broadcastable(dom::Domains) = dom.ds
Base.BroadcastStyle(::Type{<:Domains}) = Base.Broadcast.DefaultArrayStyle{1}()

Base.IndexStyle(::Type{<:Domains}) = IndexLinear()

Base.eltype(dom::Domains{T})  where T = T
Base.valtype(dom::Domains{T}) where T = T

function Base.iterate(dom::Domains, state::Int=1)
  if state <= length(dom)
    return dom[state], state + 1
  else
    return nothing
  end
end


#=====================================================================#
# Other Interfaces                                                    #
#=====================================================================#

Base.in(x, dom::Domains) = all(insorted.(x, dom))

Random.Sampler(::Type{<:AbstractRNG}, dom::Domains, ::Repetition) = SamplerTrivial(dom)

function Random.rand(rng::AbstractRNG, sp::SamplerTrivial{<:Domains})
  return [rand(rng, d) for d in sp[]]
end

function Base.permute!(dom::Domains, permutation::AbstractVector)
  @argcheck length(dom) == length(permutation)
  dom.ds .= dom.ds[permutation]
  return dom
end

function permute(dom::Domains{T}, permutation::AbstractVector) where T
  return Domains{T}(dom.ds[permutation], length(dom))
end
