# KDE kernels
# type of distribution
@enum PDF_TYPE begin
    PDF  # Probability Density Function
    CDF  # Cumulative Density Function
end

# unordered categorical default kernel: aitchison-aitken
function aitchison_aitken(bandwidth::Real, observations::Vector, x::Real; num_levels=nothing)
    if isnothing(num_levels)
        num_levels = length(unique(observations))
    end
    kernel_value = ones(length(observations)) * bandwidth / (num_levels - 1)
    idx = observations .== x
    kernel_value[idx] .= (idx * (1 - bandwidth))[idx]
    return kernel_value
end

# unordered categorical default cumulative kernel: aitchison-aitken
function aitchison_aitken_cdf(bandwidth::Real, observations::Vector, x::Real)
    x = round(Int, x)
    obs_unique = unique(observations)
    ordered = zeros(length(observations))
    num_levels = length(obs_unique)
    for x_it in obs_unique
        if x_it <= x  #FIXME: why a comparison for unordered variables?
            ordered .+= aitchison_aitken(bandwidth, observations, x_it, num_levels=num_levels)
        end
    end

    return ordered
end

# dictionary of aitchison aitken kernels
const aitchison_aitken_kernel = Dict(PDF=>aitchison_aitken, CDF=>aitchison_aitken_cdf)


# ordered categorical default kernel: wang-ryzin
function wang_ryzin(bandwidth::Real, observations::Vector, x::Real)
    kernel_value = 0.5 * (1 - bandwidth) * (bandwidth .^ abs.(observations .- x))
    idx = observations .== x
    kernel_value[idx] = (idx * (1-bandwidth))[idx]
    return kernel_value
end


# ordered categorical default cumulative kernel: wang-ryzin
function wang_ryzin_cdf(bandwidth::Real, observations::Vector, x::Real)
    ordered = zeros(length(observations))
    for x_it in unique(observations)
        if x_it <= x
            ordered .+= wang_ryzin(bandwidth, observations, x_it)
        end
    end

    return ordered
end

# dictionary of wang ryzin kernels
const wang_ryzin_kernel = Dict(PDF=>wang_ryzin, CDF=>wang_ryzin_cdf)


# continuous default kernel: gaussian
function gaussian(bandwidth::Real, observations::Vector, x::Real)
    (1 / sqrt(2*π)) * exp.(-(observations.-x).^2 / (bandwidth^2*2))
end

# continuous default cumulative kernel: gaussian
function gaussian_cdf(bandwidth::Real, observations::Vector, x::Real)
    0.5 .* bandwidth .* (1 .+ erf.((x .- observations) ./ (bandwidth * sqrt(2))))
end

# dictionary of gaussian kernels
const gaussian_kernel = Dict(PDF=>gaussian, CDF=>gaussian_cdf)