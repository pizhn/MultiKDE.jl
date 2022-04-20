# default kernel function of difference types
const KERNEL_TYPE = Dict(
    ContinuousDim => gaussian_kernel,
    CategoricalDim => wang_ryzin_kernel,
    UnorderedCategoricalDim => aitchison_aitken_kernel
)

function map_dim(candidates::Vector)
    unordered_to_index, index_to_unordered = Dict(), Dict()
    for (i, elem) in enumerate(candidates)
        unordered_to_index[elem] = i
        index_to_unordered[i] = elem
    end
    (unordered_to_index, index_to_unordered)
end

# Candidate need to be mapped from UnorderedCategoricalDim
function candidate_need_map(dim_type, candidate)
    if dim_type isa UnorderedCategoricalDim
        for _candidate in candidate
            if !(_candidate isa Real)
                return true
            end
        end
    end
    return false
end

# Observation need to be mapped from UnorderedCategoricalDim
function observation_need_map(dim_type, observation)
    if dim_type isa UnorderedCategoricalDim
        for _observation in observation
            if !(_observation isa Real)
                return true
            end
        end
    end
    return false
end

# Lazy evaluation univariate KDE
Base.@kwdef mutable struct KDEUniv
    type::DimensionType
    bandwidth::Real
    data::Vector
    kernel::Dict
end
# Using default kernel for dim_type
function KDEUniv(dim_type::DimensionType, bandwidth, observation)
    KDEUniv(dim_type, bandwidth, observation, KERNEL_TYPE[typeof(dim_type)])
end

function pdf(kde::KDEUniv, x::Real; pdf_type::PDF_TYPE=PDF, keep_all=true)
    densities = kde.kernel[pdf_type](kde.bandwidth, kde.data, x)
    if keep_all
        return densities
    else
        return mean(densities) / kde.bandwidth
    end
end

# Multivariate KDE based on KDEUniv
Base.@kwdef mutable struct KDEMulti
    # Type of every dimension, continuous or discrete
    dims::Vector
    # KDE from different dimensions
    KDEs::Vector
    # observations: An k*n matrix, where k is dimension of KDEs and n is number of observations
    observations::Vector
    mat_observations::Matrix
    # In UnorderedContinuous case, we assign an index 1:N to every value, where N=|D_i|, then we pass the 1:N to KDE
    # dimension i is number or not
    mapped::Vector
    unordered_to_index::Dict{KDEUniv, Dict{Any, Real}}
    index_to_unordered::Dict{KDEUniv, Dict{Real, Any}}
end

# Constructor without candidates
function KDEMulti(dims::Vector, observations::Vector)
    KDEMulti(dims, nothing, observations)
end
function KDEMulti(dims::Vector, bws::Union{Vector, Nothing}, observations::Vector)
    for (dim_type, observation) in zip(dims, observations)
        if observation_need_map(dim_type, observation)
            error("If there is Uncatogorical dimension and not a number, should specify its candidate value.")
        end
    end
    KDEMulti(dims, bws, observations, nothing)
end

# Constructor with candidates
## Convert tuple candidates to dict candidates
function get_dict_candidates(dim_types::Vector{DimensionType}, candidates::Tuple)
    dict_candidates = Dict{Int, Vector}()
    for (i, dim_type, candidate) in zip(1:length(dim_types), dim_types, candidates)
        if candidate_need_map(dim_type, candidate)
            dict_candidates[i] = candidate
        end
    end
    dict_candidates
end

## Tuple candidates
function KDEMulti(dims::Vector, observations::Vector, candidates::Tuple)
    KDEMulti(dims, nothing, observations, get_dict_candidates(dims, candidates))
end
## Dict candidates
function KDEMulti(dims::Vector, observations::Vector, candidates::Union{Dict{Int, Vector}, Nothing})
    KDEMulti(dims, nothing, observations, candidates)
end
function KDEMulti(dims::Vector, bws::Union{Vector, Nothing}, observations::Vector, candidates::Union{Dict{Int, Vector}, Nothing})
    mat_observations = hcat(observations...)
    KDEMulti(dims, bws, mat_observations, candidates)
end

# Main constructor
function KDEMulti(dims::Vector, bws::Union{Vector, Nothing}, mat_observations::Matrix, candidates::Union{Dict{Int, Vector}, Nothing})
    _KDEs, _observations, _unordered_to_index, _index_to_unordered = Vector(), Vector(), Dict{Int, Dict{Any, Real}}(), 
                                                                            Dict{Int, Dict{Real, Any}}()
    mapped = Vector{Bool}()
    for (i, dim_type_i) in zip(1:size(mat_observations)[1], dims)
        _observations_i = mat_observations[i, :]
        if observation_need_map(dim_type_i, _observations_i) && ((candidates isa Nothing) || !haskey(candidates, i))
            error("No corresponding candidate at dimension. ")
        end
        if !(candidates isa Nothing) && haskey(candidates, i) && candidate_need_map(dim_type_i, candidates[i])
            _unordered_to_index_i, _index_to_unordered_i = map_dim(candidates[i])
            _unordered_to_index[i], _index_to_unordered[i] = _unordered_to_index_i, _index_to_unordered_i
            _observations_i = [_unordered_to_index_i[elem] for elem in _observations_i]
            push!(mapped, true)
        else
            push!(mapped, false)
        end
        push!(_observations, _observations_i)
    end
    if bws isa Nothing
        bws = default_bandwidth(_observations)
    end
    for (_observations_i, bandwidth_i, dim_type_i) in zip(_observations, bws, dims)
        # kde_i = KDEUniv(_observations_i, bandwidth_i, dim_type_i, type_kernel[typeof(dim_type_i)], is_number)
        kde_i = KDEUniv(dim_type_i, bandwidth_i, _observations_i)
        push!(_KDEs, kde_i)
    end
    unordered_to_index, index_to_unordered = Dict{KDEUniv, Dict{Any, Real}}(), Dict{KDEUniv, Dict{Real, Any}}()
    for i in 1:size(mat_observations)[1]
        if haskey(_index_to_unordered, i)
            unordered_to_index[_KDEs[i]] = _unordered_to_index[i]
            index_to_unordered[_KDEs[i]] = _index_to_unordered[i]
        end
    end
    KDEMulti(dims, _KDEs, _observations, mat_observations, mapped, unordered_to_index, index_to_unordered)
end

# GPKE and kernel code refers to Nonparametric part of ['statsmodels'](https://github.com/statsmodels/statsmodels)
# 
# Copyright of statsmodels: 
# 
#     Copyright (C) 2006, Jonathan E. Taylor
#     All rights reserved.
#     
#     Copyright (c) 2006-2008 Scipy Developers.
#     All rights reserved.
#     
#     Copyright (c) 2009-2018 statsmodels Developers.
#     All rights reserved.

# pdf of KDEMulti, using (unnormalized) GPKE(Generalized Product Kernel Estimator)
function gpke(multi_kde::KDEMulti, x::Vector, pdf_type::PDF_TYPE=PDF)
    Kval = Array{Real}(undef, (length(multi_kde.observations[1]), length(multi_kde.observations)))
    for (i, _kde, _x) in zip(1:length(x), multi_kde.KDEs, x)
        if multi_kde.mapped[i]
            _x = multi_kde.unordered_to_index[_kde][_x]
        end
        Kval[:, i] = pdf(_kde, _x; pdf_type=pdf_type)
    end
    iscontinuous = [_dim isa ContinuousDim ? true : false for _dim in multi_kde.dims]
    dens = prod(Kval, dims=2) / prod([kde.bandwidth for kde in multi_kde.KDEs][iscontinuous])
    sum(dens)
end

# Alias of gpke
function pdf(multi_kde::KDEMulti, x::Vector, pdf_type::PDF_TYPE=PDF)
    gpke(multi_kde, x, pdf_type)
end

# Alias of gpke
function cdf(multi_kde::KDEMulti, x::Vector)
    gpke(multi_kde, x, CDF)
end

# Scott's normal reference rule of thumb bandwidth parameter
function default_bandwidth(observations::Vector)
    X = std.(observations)
    1.06 * X * length(observations).^(-1 / (4 + length(observations[1])))
end
