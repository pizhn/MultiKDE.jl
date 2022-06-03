# KDE kernels
# unordered categorical default kernel: aitchison-aitken
function aitchison_aitken(bandwidth::Real, observations::Vector, x::Real)
    num_levels = length(unique(observations))
    kernel_value = ones(length(observations)) * bandwidth / (num_levels - 1)
    idx = observations .== x
    kernel_value[idx] .= (idx * (1 - bandwidth))[idx]
    return kernel_value
end

# ordered categorical default kernel: wang-ryzin
function wang_ryzin(bandwidth::Real, observations::Vector, x::Real)
    kernel_value = 0.5 * (1 - bandwidth) * (bandwidth .^ abs.(observations .- x))
    idx = observations .== x
    kernel_value[idx] = (idx * (1-bandwidth))[idx]
    return kernel_value
end

# continuous default kernel: gaussian
function gaussian(bandwidth::Real, observations::Vector, x::Real)
    (1 / sqrt(2*π)) * exp.(-(observations.-x).^2 / (bandwidth^2*2))
end
