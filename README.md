# MultiKDE

[![Actions Status](https://github.com/noil-reed/MultiKDE.jl/workflows/CI/badge.svg)](https://github.com/noil-reed/MultiKDE.jl/actions)
[![codecov](https://codecov.io/gh/noil-reed/MultiKDE.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/noil-reed/MultiKDE.jl)


A kernel density estimation library, what make this one different from other Julia KDE libraries are:
1. Multidimension: Using [product kernel](https://csyue.nccu.edu.tw/ch/Kernel%20Estimation(Ref).pdf) to estimate multi-dimensional kernel density. 
2. Lazy evaluation: Doesn't pre-initialize a KDE, only evaluate points when necessary. 
3. Categorical distribution: This library supports categorical KDE using two specific kernel functions [Wang-Ryzin](https://academic.oup.com/biomet/article-abstract/68/1/301/237752?redirectedFrom=fulltext) and [Aitchson-Aitken](https://academic.oup.com/biomet/article-abstract/63/3/413/270829?redirectedFrom=fulltext), in which the former one is for categorical distribution that is ordered (age, amount...), the latter is for categorical distribution that is unordered (sex, the face of the coin...). When using unordered categorical distribution, non-numeric objects are also supported. 


## Use
### Example <sub><sup>[[notebook]](https://github.com/noil-reed/notebooks/blob/main/MultiKDE_demo/demo.ipynb)</sup></sub>


#### One-dimension KDE

```julia

using MultiKDE
using Distributions, Random, Plots

# Simulation
bws = [0.05 0.1 0.5]
d = Normal(0, 1)
observations = rand(d, 50)
granularity_1d = 100
x = Vector(LinRange(minimum(observations), maximum(observations), granularity_1d))
ys = []
for bw in bws
    kde = KDEUniv(ContinuousDim(), bw, observations, MultiKDE.gaussian)
    y = [MultiKDE.pdf(kde, _x, keep_all=false) for _x in x]
    push!(ys, y)
end

# Plot
highest = maximum([maximum(y) for y in ys])
plot(x, ys, label=bws, fmt=:svg)
plot!(observations, [highest+0.05 for _ in 1:length(ys)], seriestype=:scatter, label="observations", size=(900, 450), legend=:outertopright)

```

![1d KDE visualization](https://raw.githubusercontent.com/noil-reed/notebooks/842a60e81bad431dd70c6e04eb93f82ff10c1cda/MultiKDE_demo/dim1.svg)

#### Multi-dimension KDE


```julia

using MultiKDE
using Distributions, Random, Plots

# Simulation
dims = [ContinuousDim(), ContinuousDim()]
bws = [[0.3, 0.3], [0.5, 0.5], [1, 1]]
mn = MvNormal([0, 0], [1, 1])
observations = rand(mn, 50)
observations = [observations[:, i] for i in 1:size(observations, 2)]
observations_x1 = [_obs[1] for _obs in observations]
observations_x2 = [_obs[2] for _obs in observations]
granularity_2d = 100
x1_range = LinRange(minimum(observations_x1), maximum(observations_x1), granularity_2d)
x2_range = LinRange(minimum(observations_x2), maximum(observations_x2), granularity_2d)
x_grid = [[_x1, _x2] for _x1 in x1_range for _x2 in x2_range]
y_grid = []
for bw in bws
    kde = KDEMulti(dims, bw, observations)
    y = [MultiKDE.pdf(kde, _x) for _x in x_grid]
    push!(y_grid, y)
end

# Plot
highest = maximum([maximum(y) for y in y_grid])
plot([_x[1] for _x in x_grid], [_x[2] for _x in x_grid], y_grid, label=[bw[1] for bw in bws][:, :]', size=(900, 450), legend=:outertopright)
plot!(observations_x1, observations_x2, [highest for _ in 1:length(observations)], seriestype=:scatter, label="observations")

```

![2d KDE visualization](https://raw.githubusercontent.com/noil-reed/notebooks/842a60e81bad431dd70c6e04eb93f82ff10c1cda/MultiKDE_demo/dim2.svg)

<!-- ### Categorical (TBA) -->

## Post
[MultiKDE.jl: A Lazy Evaluation Multivariate Kernel Density Estimator](https://noilreed.github.io/2021/MultiKDE.jl-a-lazy-evaluation-multivariate-kernel-density-estimator/)

## Liscense
Licensed under [MIT Liscense](https://github.com/noil-reed/MultiKDE.jl/blob/main/LICENSE).

## Contact
ping69852@gmail.com
