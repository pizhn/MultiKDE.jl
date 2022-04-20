# KDE kernels
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
    obs_unique = unique(observations)
    ordered = zeros(length(observations))
    num_levels = length(obs_unique)
    for x_it in obs_unique
        if x_it <= x  #FIXME: why a comparison for unordered variables?
            ordered += aitchison_aitken(bandwidth, observations, x, num_levels=num_levels)
        end
    end

    return ordered
end

# ordered categorical default kernel: wang-ryzin
function wang_ryzin(bandwidth::Real, observations::Vector, x::Real)
    kernel_value = 0.5 * (1 - bandwidth) * (bandwidth .^ abs.(observations .- x))
    idx = observations .== x
    kernel_value[idx] = (idx * (1-bandwidth))[idx]
    return kernel_value
end


# ordered categorical default cumulative kernel: wang-ryzin
function wang_ryzin_cdf(bandwidth::Real, observations::Vector, x::Real) #(h, Xi, x_u):
    ordered = zeros(length(observations))
    for x_it in unique(observations)
        if x_it <= x
            ordered += wang_ryzin(bandwidth, observations, x_it)
        end
    end

    return ordered
end

# continuous default kernel: gaussian
function gaussian(bandwidth::Real, observations::Vector, x::Real)
    (1 / sqrt(2*π)) * exp.(-(observations.-x).^2 / (bandwidth^2*2))
end

# continuous default cumulative kernel: gaussian
function gaussian_cdf(bandwidth::Real, observations::Vector, x::Real)
    0.5 * bandwidth * (1 + erf.((x .- observations) ./ (bandwidth * sqrt(2))))
end
