using Test, MultiKDE, Distributions  # , Plots

# Hard-code test cases and expected results
hardcode_data = [
    [0.75762386,6.0,8.0,],
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
    [0.88377118,6.0,3.0,],
]
hardcode_data_pred = [
    [0.11249392,4.0,6.0,],
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
    [0.41038077,2.0,3.0,],
]

# Expected result of above input
hardcode_expected_result = [
    [4.014556252605015e-07,4.93587849306923e-07,5.017127433753062e-05,4.297840888944418,1.4103686336015966,0.0004239071556898233,0.0002450830674101923,0.001520097546290904,1.6695504130319276e-05,9.585990125997658e-05,0.0024919490954259764,0.024667731573921547,1.2143707817645687e-05,8.952814734339356e-05,3.470292679120249e-09,9.825400944075402e-06,0.00017389979047319838,0.10795426189791625,2.5029651369084925e-05,4.859412084689039,],
    [0.006731618223323871,0.034852108047552656,0.0008987511262355018,0.26494502607810944,0.002093798643830388,0.04134463379287447,0.001734435606041325,0.034237993437472795,0.34814274656951194,0.0014581092899801078,0.002014698187056216,0.002549489235012434,0.03711991361931681,0.32252782622899556,0.0013659916119672653,0.0024460021110439176,0.06131450492717148,0.0012846078414541185,0.000750247341719304,0.0017450935407583167,],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],
    [0.07012047583564995,0.16682618301528643,0.04595196932831633,0.26088694468743184,0.15452574048888829,0.09336064500373596,0.11310195315610574,0.09485512130103643,0.3361451517102332,0.10902096724118503,0.13788547837516862,0.05885101643100109,0.1329887267194227,0.3387753430435409,0.0787994864890616,0.052985136882964974,0.10988250614008524,0.07668631403187032,0.06941533271304429,0.11576170881780576,],
]
hardcode_dims = [ContinuousDim(), CategoricalDim(10), UnorderedCategoricalDim(10)]
hardcode_bws = [[0.01, 0.1, 1], [1, 0.2, 0.01], [0.02, 1, 1], [0.5, 0.5, 0.5]]

# Arguments for sanity tests
continuous_range = (0, 200)
categorical_level = 10
unordered_extra_level = 10
# zero_val_tolerance = 1e-2
s1_tol = 1
univ_sanity_trial, multi_sanity_trial = 5, 1
num_s2_test = 100
num_obs = 20
dim_types = [ContinuousDim, CategoricalDim, UnorderedCategoricalDim]
bw_s1 = 0.1
bw_s2 = 0.1
diff_delta = 2e-1
multi_test_num_dims = [1, 2, 3, 4, 5]
min_range = 1
# Objects for sanity tests
unordered_objs = Vector{Any}([Vector{Int},  Vector{Float64}, Vector{Array}, Vector{Vector}, Vector{Pair}, Vector{ErrorException}, Vector{Tuple}])
append!(unordered_objs, 1:unordered_extra_level)
unordered_level = length(unordered_objs)

function randint(n_mx)
    ceil(Int,rand()*n_mx)
end

function randint(N, n_mx)
    [ceil(Int,rand()*n_mx) for _ in 1:N]
end

function gen_batch_data(dim_type::DimensionType, n)
    dt = []
    for _ in 1:n
        _dt = gen_data(dim_type)
        if !(_dt isa Real)
            _dt = findall(x->x===_dt, unordered_objs)[1]
        end
        push!(dt, _dt)
    end
    dt
end

function gen_batch_data(dim_types::Vector, n)
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
        _dt = rand(Uniform(continuous_range[1], continuous_range[2]))
    else
        error("Nonsupported dim_type")
    end
    _dt
end

# Check if the point is local maximum(gradient=0) using difference differential
function is_maximum(multi_kde::KDEMulti, x; delta=1e-10)
    x1 = [_kde.type isa ContinuousDim ? _x-delta : 
                            (_kde.type isa UnorderedCategoricalDim ? _x : _x-delta) 
                            for (_kde, _x) in zip(multi_kde.KDEs, x)]
    x2 = [_kde.type isa ContinuousDim ? _x+delta : 
                            (_kde.type isa UnorderedCategoricalDim ? _x : _x+delta) 
                            for (_kde, _x) in zip(multi_kde.KDEs, x)]
    val, val_x1, val_x2 = MultiKDE.pdf(multi_kde, x), MultiKDE.pdf(multi_kde, x1), MultiKDE.pdf(multi_kde, x2)
    (val-val_x1 >= 0) && (val-val_x2 >= 0)
end

function is_maximum(kde::KDEUniv, x; delta=1e-10)
    x1 = kde.type isa ContinuousDim ? x-delta : (kde.type isa UnorderedCategoricalDim ? x-delta : x-delta)
    x2 = kde.type isa ContinuousDim ? x+delta : (kde.type isa UnorderedCategoricalDim ? x+delta : x+delta)
    val, val_x1, val_x2 = MultiKDE.pdf(kde, x, keep_all=false), MultiKDE.pdf(kde, x1, keep_all=false), MultiKDE.pdf(kde, x2, keep_all=false)
    (val-val_x1 >= 0) && (val-val_x2 >= 0)
end

# Same as above, but calculating the gradient with direction that x points to y
function kde_gradient_to_point(multi_kde::KDEMulti, x, y; delta=1e-10)
    vec = [_kde.type isa ContinuousDim ? _x-_y : 
                            (_kde.type isa UnorderedCategoricalDim ? nothing : _x-_y) 
                            for (_kde, _x, _y) in zip(multi_kde.KDEs, x, y)]
    x1 = [_vec===nothing ? _x : _x-(_vec*delta) for (_x, _vec) in zip(x, vec)]
    x2 = [_vec===nothing ? _x : _x+(_vec*delta) for (_x, _vec) in zip(x, vec)]
    (MultiKDE.pdf(multi_kde, x2) - MultiKDE.pdf(multi_kde, x1)) / delta
end

function kde_gradient_to_point(kde::KDEUniv, x, y; delta=1e-10)
    vec = kde.type isa ContinuousDim ? x-y : (kde.type isa UnorderedCategoricalDim ? nothing : x-y)
    x1 = vec===nothing ? x : x-(vec*delta)
    x2 = vec===nothing ? x : x+(vec*delta)
    (MultiKDE.pdf(kde, x2; keep_all=false) - MultiKDE.pdf(kde, x1; keep_all=false)) / delta
end

@testset "KDEUniv" begin
    println("Test set for KDEUniv")
    for it in 1:univ_sanity_trial
        _dim = dim_types[randint(length(dim_types))]
        _dim = _dim==ContinuousDim ? _dim() : (_dim==CategoricalDim ? _dim(categorical_level) : _dim(unordered_level))
        dt = gen_batch_data(_dim, num_obs)
        # Filter observations that are too close
        local dt_s1
        if _dim isa ContinuousDim
            dt_s1 = filter(e->all([abs(e-_dt)>2 for _dt in dt if _dt≠e]), dt)
        else
            dt_s1 = dt
        end
        kde_s1 = KDEUniv(_dim, bw_s1, dt_s1)
        kde_s2 = KDEUniv(_dim, bw_s2, dt)
        # 1. Sanity-check 1: When bandwidth is small enough, the gradient of PDF of observations shoule be close to zero.
        println("Performing sanity check 1...", string(_dim))
        ## Gradient function
        is_maximum_s1 = [is_maximum(kde_s2, _dt, delta=diff_delta) for _dt in kde_s1.data]
        # println("------------------------------")
        # println(sort(kde_s1.data))
        # println([MultiKDE.pdf(kde_s1, _dt, keep_all=false) for _dt in sort(kde_s1.data)])
        # println(is_maximum_s1)
        # println("------------------------------")
        # # Debug plotting
        # if true  # sum(is_maximum_s1) < (length(is_maximum_s1)-s1_tol)
        #     x = LinRange(minimum(kde_s1.data), maximum(kde_s1.data), 20000)
        #     y = [MultiKDE.pdf(kde_s1, _x, keep_all=false) for _x in x]
        #     y_cdf = [MultiKDE.cdf(kde_s1, _x, keep_all=false) for _x in x]
        #     highest = maximum(y)
        #     plot(x, y)
        #     plot!(kde_s1.data, [highest+0.05 for _ in 1:length(y)], seriestype=:scatter, size=(900, 450))
        #     println(sort(kde_s1.data))
        #     println([MultiKDE.pdf(kde_s1, _dt, keep_all=false) for _dt in sort(kde_s1.data)])
        #     println(kde_s1.kernel)
        #     println(sort(kde_s1.data))
        #     println("Plotted")
        #     savefig(string(sum(is_maximum_s1), it, "_", string(_dim), ".png"))
        #     # Add plotting cdf distribution
        #     plot(x, y_cdf)
        #     plot!(kde_s1.data, [highest+0.05 for _ in 1:length(y)], seriestype=:scatter, size=(900, 450))
        #     savefig(string("cdf_", sum(is_maximum_s1), it, ".png"))
        # end
        @test sum(is_maximum_s1) >= (length(is_maximum_s1)-s1_tol)
        # 2. Sanity-check 2: Using a hyperbox to include all observations, for every vector out of box that points opposite to box center, the gradient should be negative. 
        println("Performing sanity check 2...")
        mn, mx = minimum(kde_s2.data), maximum(kde_s2.data)
        middle = (mx-mn)/2
        range = min(mx-mn, min_range)
        grad_s2 = []
        for i in 1:num_s2_test
            point_out = rand([mn-range, mx+range])
            _grad = kde_gradient_to_point(kde_s2, point_out, middle, delta=diff_delta)
            push!(grad_s2, _grad)
        end
        println(grad_s2)
        @test sum((grad_s2 .≈ 0) .| (grad_s2 .< 0)) == length(grad_s2)
    end
end

@testset "KDEMulti" begin
    println("Test set for KDEMulti")
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
    # Try different combinations
    for num_dim in multi_test_num_dims
        for _ in 1:multi_sanity_trial
            println("Number of dimensions = ", num_dim)
            # Prepare Random KDEMulti
            _dims = dim_types[randint(num_dim, length(dim_types))]
            _dims = [_dim==ContinuousDim ? _dim() : (_dim==CategoricalDim ? _dim(categorical_level) : _dim(unordered_level)) for _dim in _dims]
            _bw_s1 = [bw_s1 for _ in 1:num_dim]
            _bw_s2 = [bw_s2 for _ in 1:num_dim]
            dt = gen_batch_data(_dims, num_obs)
            candidates = Dict{Int, Vector}()
            for (i, __dims) in zip(1:length(_dims), _dims)
                if __dims isa UnorderedCategoricalDim
                    candidates[i] = unordered_objs
                end
            end
            local dt_s1
            if any([_dim isa ContinuousDim for _dim in _dims])
                valid_idx = []
                for (_k, _dim) in zip(1:length(_dims), _dims)
                    if _dim isa ContinuousDim
                        _dim_dt = [_dt[_k] for _dt in dt]
                        _valid_idx = filter(_i->all([abs(_dim_dt[_i]-_dim_dt[_j])>2 for _j in 1:length(_dim_dt) if _dim_dt[_i]≠_dim_dt[_j]]), 1:length(_dim_dt))
                        push!(valid_idx, _valid_idx)
                    end
                end
                valid_idx = intersect(valid_idx...)
                dt_s1 = dt[valid_idx]
            else
                dt_s1 = dt
            end
            # println("------------------------------")
            # println(_dims)
            # println(_bw_s1)
            # println(_bw_s2)
            # println(dt_s1)
            # println(candidates)
            # println("------------------------------")
            multi_kde_s1 = KDEMulti(_dims, _bw_s1, dt_s1, candidates)
            multi_kde_s2 = KDEMulti(_dims, _bw_s2, dt, candidates)
            # 2. Sanity-check 1: When bandwidth is small enough, the gradient of PDF of observations shoule be close to zero.
            println("Performing sanity check 1...")
            ## Gradient function
            is_maximum_s1 = [is_maximum(multi_kde_s1, _dt, delta=diff_delta) for _dt in dt_s1]
            println(is_maximum_s1)
            # @test sum(grad_s1 .< zero_val_tolerance) >= (length(grad_s1)-zero_num_tolerance)
            @test sum(is_maximum_s1) >= (length(is_maximum_s1))
            # 3. Sanity-check 2: Using a hyperbox to include all observations, for every vector out of box that points opposite to box center, the gradient should be negative. 
            println("Performing sanity check 2...")
            mins = [minimum(kde.data) for kde in multi_kde_s2.KDEs]
            maxs = [maximum(kde.data) for kde in multi_kde_s2.KDEs]
            middle = [_dim isa ContinuousDim ? (_max-_min) / 2 : 
                        (_dim isa UnorderedCategoricalDim ? (multi_kde_s2.index_to_unordered[_kde][round(Int, (_max+_min)/2)]) : round(Int, (_max+_min)/2))
                        for (i, _dim, _kde, _min, _max) in zip(1:length(multi_kde_s2.dims), multi_kde_s2.dims, multi_kde_s2.KDEs, mins, maxs)]
            ranges = [min((maxs[i]-mins[i]), min_range) for i in 1:length(mins)]
            grad_s2 = []
            for i in 1:num_s2_test
                point_out = [_dim isa ContinuousDim ? rand([mins[i]-rand(Uniform(0, ranges[i])), maxs[i]+rand(Uniform(0, ranges[i]))]) : 
                                (_dim isa UnorderedCategoricalDim ? rand([multi_kde_s2.index_to_unordered[_kde][1], multi_kde_s2.index_to_unordered[_kde][length(multi_kde_s2.index_to_unordered[_kde])]]) : 
                                                                            rand([mins[i]-rand(1:ranges[i]), maxs[i]+rand(1:ranges[i])])) 
                                for (i, _dim, _kde) in zip(1:length(multi_kde_s2.dims), multi_kde_s2.dims, multi_kde_s2.KDEs)]
                _grad = kde_gradient_to_point(multi_kde_s2, point_out, middle, delta=diff_delta)
                push!(grad_s2, _grad)
            end
            println(grad_s2)
            @test sum((grad_s2 .≈ 0) .| (grad_s2 .< 0)) == length(grad_s2)
        end
    end
end
