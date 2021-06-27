using Test, MultiKDE, Distributions

# Hard-code test cases and expected results
hardcode_data = [[0.75762386,6.0,8.0,],
[0.40709311,2.0,8.0,],
[0.73019489,1.0,7.0,],
[0.9350646,1.0,8.0,],
[0.20622478,7.0,3.0,],
[0.82868765,4.0,9.0,],
[0.83275865,6.0,5.0,],
[0.06151981,6.0,5.0,],
[0.47064985,2.0,0.0,],
[0.81312999,5.0,4.0,],
[0.60976384,5.0,0.0,],
[0.58597128,0.0,9.0,],
[0.97580821,5.0,9.0,],
[0.21986135,2.0,4.0,],
[0.33835384,6.0,6.0,],
[0.52131305,9.0,8.0,],
[0.0651185,6.0,9.0,],
[0.75884609,9.0,3.0,],
[0.59935732,5.0,4.0,],
[0.88377118,6.0,3.0,],]
hardcode_data_pred = [[0.11249392,4.0,6.0,],
[0.28704927,5.0,3.0,],
[0.06442488,1.0,6.0,],
[0.88971842,6.0,9.0,],
[0.58269063,5.0,2.0,],
[0.04070221,3.0,4.0,],
[0.33534097,2.0,3.0,],
[0.25188701,3.0,9.0,],
[0.54533024,5.0,9.0,],
[0.51731902,2.0,2.0,],
[0.40469525,5.0,1.0,],
[0.34114321,8.0,4.0,],
[0.39484506,7.0,9.0,],
[0.45614159,6.0,3.0,],
[0.01384118,2.0,2.0,],
[0.09847895,3.0,5.0,],
[0.95260732,8.0,3.0,],
[0.92190954,2.0,1.0,],
[0.10851023,7.0,7.0,],
[0.41038077,2.0,3.0,],]
# output of statsmodels given above input
hardcode_expected_result = [
    [4.014556252605015e-07,4.93587849306923e-07,5.017127433753062e-05,4.297840888944418,1.4103686336015966,0.0004239071556898233,0.0002450830674101923,0.001520097546290904,1.6695504130319276e-05,9.585990125997658e-05,0.0024919490954259764,0.024667731573921547,1.2143707817645687e-05,8.952814734339356e-05,3.470292679120249e-09,9.825400944075402e-06,0.00017389979047319838,0.10795426189791625,2.5029651369084925e-05,4.859412084689039,],
    [0.006731618223323871,0.034852108047552656,0.0008987511262355018,0.26494502607810944,0.002093798643830388,0.04134463379287447,0.001734435606041325,0.034237993437472795,0.34814274656951194,0.0014581092899801078,0.002014698187056216,0.002549489235012434,0.03711991361931681,0.32252782622899556,0.0013659916119672653,0.0024460021110439176,0.06131450492717148,0.0012846078414541185,0.000750247341719304,0.0017450935407583167,],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],
    [0.07012047583564995,0.16682618301528643,0.04595196932831633,0.26088694468743184,0.15452574048888829,0.09336064500373596,0.11310195315610574,0.09485512130103643,0.3361451517102332,0.10902096724118503,0.13788547837516862,0.05885101643100109,0.1329887267194227,0.3387753430435409,0.0787994864890616,0.052985136882964974,0.10988250614008524,0.07668631403187032,0.06941533271304429,0.11576170881780576,],
]
hardcode_dims = [ContinuousDim(), CategoricalDim(10), UnorderedCategoricalDim(10)]
hardcode_bws = [[0.01, 0.1, 1], [1, 0.2, 0.01], [0.02, 1, 1], [0.5, 0.5, 0.5]]

# test data for sanity tests
categorical_level = 10
unordered_extra_level = 10
unordered_objs = Vector{Any}([Vector{Int},  Vector{Float64}, Vector{Array}, Vector{Vector}, Vector{Pair}, Vector{ErrorException}, Vector{Tuple}])
append!(unordered_objs, 1:unordered_extra_level)
unordered_level = length(unordered_objs)

function randint(N, n_mx)
    [ceil(Int,rand()*n_mx) for _ in 1:N]
end

function gen_batch_data(dim_types, n)
    dt = []
    for _ in 1:n
        push!(dt, [gen_data(dim_types[i]) for i in 1:length(dim_types)])
    end
    dt
end

function gen_data(dim_type)
    local _dt
    if dim_type isa CategoricalDim
        _dt = rand(1:categorical_level)
    elseif dim_type isa UnorderedCategoricalDim
        _dt = unordered_objs[rand(1:length(unordered_objs))]
    elseif dim_type isa ContinuousDim
        _dt = rand(Uniform(0, 100))
    else
        error("Nonsupported dim_type")
    end
    _dt
end

# Calculating gradient using difference differential
function kde_gradient(multi_kde, x, delta=1e-10)
    x1 = [_kde.type isa ContinuousDim ? _x-delta : 
                            (_kde.type isa UnorderedCategoricalDim ? _x : _x-delta) 
                            for (_kde, _x) in zip(multi_kde.KDEs, x)]
    x2 = [_kde.type isa ContinuousDim ? _x+delta : 
                            (_kde.type isa UnorderedCategoricalDim ? _x : _x+delta) 
                            for (_kde, _x) in zip(multi_kde.KDEs, x)]
    (MultiKDE.pdf(multi_kde, x2) - MultiKDE.pdf(multi_kde, x1)) / delta
end

# Same as above, but calculating the gradient with direction that x points to y
function kde_gradient_to_point(multi_kde, x, y, delta=1e-10)
    vec = [_kde.type isa ContinuousDim ? _x-_y : 
                            (_kde.type isa UnorderedCategoricalDim ? nothing : _x-_y) 
                            for (_kde, _x, _y) in zip(multi_kde.KDEs, x, y)]
    x1 = [_vec===nothing ? _x : _x-(_vec*delta) for (_x, _vec) in zip(x, vec)]
    x2 = [_vec===nothing ? _x : _x+(_vec*delta) for (_x, _vec) in zip(x, vec)]
    (MultiKDE.pdf(multi_kde, x2) - MultiKDE.pdf(multi_kde, x1)) / delta

end

@testset "multi_kde" begin
    # 1. Hard-code cases check
    # result = RealVectorVector()
    result = Vector()
    for bandwidth in hardcode_bws
        multi_kde = KDEMulti(hardcode_dims, bandwidth, hardcode_data)
        _result = [MultiKDE.pdf(multi_kde, _data) for _data in hardcode_data_pred]
        push!(result, _result)
    end
    @test sum(result .≈ hardcode_expected_result) == length(hardcode_expected_result)
    # Variables needed for Sanity-check
    dim_types = [ContinuousDim, CategoricalDim, UnorderedCategoricalDim]
    # Try different combinations
    num_obs_test = 100
    num_dims = [1, 2, 3, 4, 5]
    bw2 = 1e-5
    bw3 = 0.1
    for num_dim in num_dims
        println("Number of dimensions = ", num_dim)
        # Prepare Random KDEMulti
        _dims = dim_types[randint(num_dim, length(dim_types))]
        _dims = [_dim==ContinuousDim ? _dim() : (_dim==CategoricalDim ? _dim(categorical_level) : _dim(unordered_level)) for _dim in _dims]
        _bw2 = [bw2 for _ in 1:num_dim]
        _bw3 = [bw3 for _ in 1:num_dim]
        dt = gen_batch_data(_dims, num_obs_test)
        candidates = Dict{Int, Vector}()
        for (i, __dims) in zip(1:length(_dims), _dims)
            if __dims isa UnorderedCategoricalDim
                candidates[i] = unordered_objs
            end
        end
        multi_kde2 = KDEMulti(_dims, _bw2, dt, candidates)
        multi_kde3 = KDEMulti(_dims, _bw3, dt, candidates)
        # 2. Sanity-check 1: When bandwidth is small enough, the gradient of PDF of observations shoule be close to zero.
        println("Performing sanity check 1...")
        ## Gradient function
        grad2 = [kde_gradient(multi_kde2, _dt) for _dt in dt]
        println(grad2)
        @test sum(grad2 .< 1e-3) == length(grad2)
        # 3. Sanity-check 2: Using a hyperbox to include all observations, for every vector out of box that points opposite to box center, the gradient should be negative. 
        println("Performing sanity check 2...")
        num_s2_test = 100
        mins = [minimum(kde.data) for kde in multi_kde3.KDEs]
        maxs = [maximum(kde.data) for kde in multi_kde3.KDEs]
        middle = [_dim isa ContinuousDim ? (_max-_min) / 2 : 
                    (_dim isa UnorderedCategoricalDim ? (multi_kde3.index_to_unordered[_kde][round(Int, (_max+_min)/2)]) : round(Int, (_max+_min)/2))
                    for (i, _dim, _kde, _min, _max) in zip(1:length(multi_kde3.dims), multi_kde3.dims, multi_kde3.KDEs, mins, maxs)]
        ranges = [min((maxs[i]-mins[i]), 1) for i in 1:length(mins)]
        grad3 = []
        for i in 1:num_s2_test
            point_out = [_dim isa ContinuousDim ? rand([mins[i]-rand(Uniform(0, ranges[i])), maxs[i]+rand(Uniform(0, ranges[i]))]) : 
                            (_dim isa UnorderedCategoricalDim ? rand([multi_kde3.index_to_unordered[_kde][1], multi_kde3.index_to_unordered[_kde][length(multi_kde3.index_to_unordered[_kde])]]) : 
                                                                         rand([mins[i]-rand(1:ranges[i]), maxs[i]+rand(1:ranges[i])])) 
                            for (i, _dim, _kde) in zip(1:length(multi_kde3.dims), multi_kde3.dims, multi_kde3.KDEs)]
            _grad = kde_gradient_to_point(multi_kde3, point_out, middle)
            push!(grad3, _grad)
        end
        println(grad3)
        @test sum((grad3 .≈ 0) .| (grad3 .< 0)) == length(grad3)
    end
end