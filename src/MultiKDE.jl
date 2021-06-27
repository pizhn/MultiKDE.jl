# Multivariate Lazy Kernel Density Estimator ====================================================================

module MultiKDE

export KDEMulti, KDEUniva, pdf, gpke
export DimensionType, ContinuousDim, CategoricalDim, UnorderedCategoricalDim

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
