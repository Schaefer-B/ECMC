using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using BenchmarkTools
using Serialization


include("../ecmc.jl")



function get_lift_vectors(nvectors, variation_type, step_var)

    result_vectors = []

    @showprogress 1 "Getting lift vectors for step_var = $step_var" for i in 1:nvectors
        lift_vector = refresh_lift_vector(2)
        delta = refresh_delta(1., step_var, 1., variation_type)

        push!(result_vectors, lift_vector*delta)
    end

    return result_vectors
end



function calculate_and_plot(variation_type, step_vars, nvectors=10^3, bins=10)

    
    title = string(variation_type)

    var_vectors = []
    for var_value in step_vars
        v = get_lift_vectors(nvectors, variation_type, var_value)
        push!(var_vectors, v)
    end

    n = length(step_vars)
    gr(size=(1.3*900*n, 1.3*800), thickness_scaling = 1.5)
    final_plot = plot(layout=(1, n))
    for i in 1:n
        v = var_vectors[i]
        x = [v[k][1] for k=eachindex(v)]
        y = [v[k][2] for k=eachindex(v)]
        final_plot = plot!(x, y, st=:histogram2d, subplot=i, bins=(bins, bins), normalized=:pdf, title=string("    ", title, ", variation value = ", step_vars[i]))
    end


    return final_plot, var_vectors
end


function mean_length(all_vectors)

    k = length(all_vectors)

    means = []
    stds = []
    for var_index in 1:k
        length_arr = []
        v_arr = all_vectors[var_index]
        for v in v_arr
            l = sqrt(v[1]^2 + v[2]^2)
            push!(length_arr, l)
        end
        push!(means, mean(length_arr))
        push!(stds, std(length_arr))
    end
    return means, stds
end


#-------------------------
variation_types = [NoVariation(), UniformVariation(), NormalVariation(), ExponentialVariation()]
step_vars = [0.1, 0.3]
nvectors = 1.2*10^7
bins = 150
nvectors/bins^2


v_type = 3
final_plot, all_vectors = calculate_and_plot(variation_types[v_type], step_vars, nvectors, bins);

display(final_plot)

png(string(variation_types[v_type],"_visualization"))

means, stds = mean_length(all_vectors)

