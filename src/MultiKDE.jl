# Multivariate Lazy Kernel Density Estimator ====================================================================

module MultiKDE

export KDEMulti, KDEUniv, pdf, gpke
export DimensionType, ContinuousDim, CategoricalDim, UnorderedCategoricalDim

using Distributions, SpecialFunctions

abstract type  DimensionType end
struct ContinuousDim <: DimensionType end 
struct CategoricalDim <: DimensionType
    level::Int
end
struct UnorderedCategoricalDim <: DimensionType
    level::Int
end

include("kernel.jl")
include("kde.jl")

end
