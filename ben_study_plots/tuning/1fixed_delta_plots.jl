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


include("../../ecmc.jl")
include("../test_distributions.jl")







#----------------functions for sampling------------------
function create_delta_array(delta_start, delta_end, steps)
    a = log(delta_start)
    b = log(delta_end)
    log_arr = collect(range(a, b, steps))
    arr = exp.(log_arr)
    return arr
end

function create_algorithm(delta, nsamples, Direction, step_var=0., var_type = NoVariation(), jb_refresh=50)
    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= nsamples,
        nburnin = 0,
        nchains = 1,
        chain_length=20, 
        remaining_jumps_before_refresh = jb_refresh,
        step_amplitude = delta,
        factorized = false,
        step_var = step_var,
        variation_type = var_type,
        direction_change = Direction,
        tuning = MFPSTuner(adaption_scheme=GoogleAdaption(automatic_adjusting=true), max_n_steps = 2*10^4),
    )
    return algorithm
end


function create_ecmc_states(density, algorithm, nchains)
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:nchains]
    

    delta = []
    for i in 1:nchains
        push!(delta, algorithm.step_amplitude)
    end
    #delta = algorithm.step_amplitude

    
    params = [0.98,
    0.207210542888343,
    0.0732514723260891,
    0.4934509024569294,
    17.587673168668637,
    2.069505973296211,
    0.6136869940758715,
    163.53188824455017]

    ecmc_states = [ECMCState(
        C = initial_samples[i],
        current_energy = -logdensityof(density, initial_samples[i]),
        lift_vector = lift_vectors[i], 
        delta = delta[i], 
        step_amplitude = delta[i], 
        step_var = algorithm.step_var,
        
        variation_type = algorithm.variation_type,
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh,
        )  for i in 1:nchains]

    return ecmc_states
end




function acc_run(posterior, algorithm, nchains)
    posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
    density, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
    shape = varshape(density)


    ecmc_states = create_ecmc_states(density, algorithm, nchains)


    acc_per_chain = Vector{Float64}(undef, nchains)
    ess_per_chain = Vector{Float64}(undef, nchains)
    diff_to_mean_per_chain = Vector{Float64}(undef, nchains)

    for c in 1:nchains
        samples = [_ecmc_sample(density, algorithm, ecmc_state=ecmc_states[c], chainid=c)]
        samples_trafo = shape.(convert_to_BAT_samples(samples, density))
        samples_per_chain = inverse(trafo).(samples_trafo)
        ess_per_chain[c] = mean(bat_eff_sample_size(samples_per_chain).result.a)
        acc_per_chain[c] = (ecmc_states[c].n_acc/ecmc_states[c].n_steps)
        diff_to_mean_per_chain[c] = mean(abs.(mean(samples_per_chain).a))
    end

    
    mean_acc = mean(acc_per_chain)
    std_acc = std(acc_per_chain)
    mean_ess = mean(ess_per_chain)
    mean_diff = mean(diff_to_mean_per_chain)

    return mean_acc, std_acc, mean_ess, mean_diff
end



function run_one_algo(delta_tests_arr, nsamples, runs, distribution, dimension, dir_algo, step_var=0.1, var_type = NoVariation(), jb_refresh=50)

    likelihood, prior = distribution(dimension);
    posterior = PosteriorMeasure(likelihood, prior);

    algorithms = [create_algorithm(delta_tests_arr[d], nsamples, dir_algo, step_var, var_type, jb_refresh) for d=eachindex(delta_tests_arr)]



    k = length(delta_tests_arr)
    mean_acc = Vector{Float64}(undef, k)
    std_acc = Vector{Float64}(undef, k)
    mean_ess_arr = Vector{Float64}(undef, k)
    mean_diff = Vector{Float64}(undef, k)

    for delta_index in 1:k
        mean_acc[delta_index], std_acc[delta_index], mean_ess_arr[delta_index], mean_diff[delta_index] = acc_run(posterior, algorithms[delta_index], runs) 
    end



    return mean_acc, std_acc, mean_ess_arr, mean_diff
end


function run_all_algos(delta_tests_arr, nsamples, runs, distribution, dimension, dir_algos, step_var=0.1, var_type = NoVariation(), jb_refresh=50)

    algo_count = length(dir_algos)
    mean_accs = Vector{Vector{Float64}}(undef, algo_count)
    std_accs = Vector{Vector{Float64}}(undef, algo_count)
    ess_arrays = Vector{Vector{Float64}}(undef, algo_count)
    mean_diffs = Vector{Vector{Float64}}(undef, algo_count)
    Threads.@threads for i in 1:algo_count
        mean_accs[i], std_accs[i], ess_arrays[i], mean_diffs[i] = run_one_algo(delta_tests_arr, nsamples, runs, distribution, dimension, dir_algos[i], step_var, var_type, jb_refresh)
    end

    return mean_accs, std_accs, ess_arrays, mean_diffs
end


function find_mean(ess_arrays, delta_tests_arr, mean_accs, direction_algos)
    
    best_acc = []
    mfp_arr = []
    for algo_index in eachindex(direction_algos)
        ess_sum = sum(ess_arrays[algo_index])
        p_ess = ess_arrays[algo_index]/ess_sum

        mean_delta = sum(delta_tests_arr .* p_ess)

        lower_index = findlast(x -> x <= mean_delta ? true : false, delta_tests_arr)
        if lower_index != length(delta_tests_arr)
            higher_index = lower_index + 1

            lower_delta = delta_tests_arr[lower_index]
            higher_delta = delta_tests_arr[higher_index]

            lower_acc = mean_accs[algo_index][lower_index]
            higher_acc = mean_accs[algo_index][higher_index]

            linear_gradient = (higher_acc - lower_acc) / (higher_delta - lower_delta)
            diff = mean_delta - lower_delta

            acc = linear_gradient * diff + lower_acc
            push!(best_acc, acc)

            mfp = acc/(1-acc)

            push!(mfp_arr, mfp)
        else
            acc = mean_accs[algo_index][lower_index]
            push!(best_acc, acc)

            mfp = acc/(1-acc)

            push!(mfp_arr, mfp)
        end
    end

    return best_acc, mfp_arr
end


#----------------functions for plotting------------------
function plot_accs(delta_tests_arr, mean_accs, std_accs, ess_arrays, direction_algos)


    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    
    max_layout_prim = 5
    if algo_count > 1
        for i in 2:max_layout_prim
            if algo_count%i == 0
                final_plot = plot(layout=(Int(algo_count/i), i))
                break
            end
        end
    else
        final_plot = plot(layout=(1,1))
    end


    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)

        m = maximum(ess_arrays[algo_index])
        m_index = findall(x->x == m ? true : false, ess_arrays[algo_index])
        m_delta = delta_tests_arr[m_index]

        ribbon_lower = - std_accs[algo_index]
        ribbon_upper = std_accs[algo_index]
        final_plot = plot!(delta_tests_arr, mean_accs[algo_index], lw=2, title=label, xlims=(delta_tests_arr[1],Inf), ylims=(0. , 1.), subplot=algo_index, ribbon=(ribbon_lower, ribbon_upper))
        final_plot = plot!(xlabel="Delta", ylabel="Acceptance rate", subplot=algo_index)
        

        x_diff = maximum(delta_tests_arr) - minimum(delta_tests_arr)
        x_translation = 0.94#- x_diff*0.01
        for delta_index in eachindex(m_delta)
            final_plot = plot!([m_delta[delta_index]], st=:vline, lw=2, subplot=algo_index, color=:red)
            annotate!((m_delta[delta_index]*x_translation), 0.12, text(string("Delta = ", round(m_delta[delta_index], digits=5)), :red, :right, 8), subplot=algo_index)
            annotate!((m_delta[delta_index]*x_translation), 0.06, text(string("Acc. rate = ", round(mean_accs[algo_index][m_index[delta_index]], digits=5)), :red, :right, 8), subplot=algo_index)
        end

    end
    final_plot = plot!(legend = false)
    final_plot = plot!(xaxis=:log)
    #final_plot = plot!(xlims=(delta_tests_arr[1],Inf))

    return final_plot
end


function plot_mean_ess(delta_tests_arr, mean_accs, ess_arrays, direction_algos)


    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    
    max_layout_prim = 5

    if algo_count > 1
        for i in 2:max_layout_prim
            if algo_count%i == 0
                final_plot = plot(layout=(Int(algo_count/i), i))
                break
            end
        end
    else
        final_plot = plot(layout=(1,1))
    end

    
    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)

        m = maximum(ess_arrays[algo_index])
        m_index = findall(x->x == m ? true : false, ess_arrays[algo_index])
        m_delta = delta_tests_arr[m_index]

        final_plot = plot!(delta_tests_arr, ess_arrays[algo_index], title=label, subplot=algo_index, lw=2, xlims=(delta_tests_arr[1],Inf), ylims=(0. , Inf))
        final_plot = plot!(xlabel="Delta", ylabel="Mean ESS per dimension", subplot=algo_index)

        diff = maximum(delta_tests_arr) - minimum(delta_tests_arr)
        x_translation = 0.94#- diff*0.01
        for delta_index in eachindex(m_delta)
            final_plot = plot!([m_delta[delta_index]], st=:vline, lw=2, subplot=algo_index, color=:red)
            annotate!((m_delta[delta_index]*x_translation), 0.1*maximum(ess_arrays[algo_index]), text(string("Delta = ", round(m_delta[delta_index], digits=5)), :red, :right, 8), subplot=algo_index)
            annotate!((m_delta[delta_index]*x_translation), 0.05*maximum(ess_arrays[algo_index]), text(string("Acc. rate = ", round(mean_accs[algo_index][m_index[delta_index]], digits=5)), :red, :right, 8), subplot=algo_index)
        end
    end
    final_plot = plot!(legend = false)
    final_plot = plot!(xaxis=:log)
    #final_plot = plot!(xlims=(delta_tests_arr[1],Inf))

    return final_plot
end


function plot_mean_diffs(delta_tests_arr, mean_accs, mean_diffs, ess_arrays, direction_algos)


    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    
    max_layout_prim = 5
    if algo_count > 1
        for i in 2:max_layout_prim
            if algo_count%i == 0
                final_plot = plot(layout=(Int(algo_count/i), i))
                break
            end
        end
    else
        final_plot = plot(layout=(1,1))
    end


    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)

        m = maximum(ess_arrays[algo_index])
        m_index = findall(x->x == m ? true : false, ess_arrays[algo_index])
        m_delta = delta_tests_arr[m_index]

        final_plot = plot!(delta_tests_arr, mean_diffs[algo_index], lw=2, title=label, xlims=(delta_tests_arr[1],Inf), ylims=(0. , Inf), subplot=algo_index)
        final_plot = plot!(xlabel="Delta", ylabel="Mean difference to true mean", subplot=algo_index)
        

        x_diff = maximum(delta_tests_arr) - minimum(delta_tests_arr)
        x_translation = 0.94#- x_diff*0.01
        for delta_index in eachindex(m_delta)
            final_plot = plot!([m_delta[delta_index]], st=:vline, lw=2, subplot=algo_index, color=:red)
            annotate!((m_delta[delta_index]*x_translation), 0.5*maximum(mean_diffs[algo_index]), text(string("Delta = ", round(m_delta[delta_index], digits=5)), :red, :right, 8), subplot=algo_index)
            #annotate!((m_delta[delta_index]*x_translation), 0.06, text(string("Acc. ratio = ", round(mean_accs[algo_index][m_index[delta_index]], digits=5)), :red, :right, 8), subplot=algo_index)
            annotate!((m_delta[delta_index]*x_translation), 0.45*maximum(mean_diffs[algo_index]), text(string("Acc. rate = ", round(mean_accs[algo_index][m_index[delta_index]], digits=5)), :red, :right, 8), subplot=algo_index)
        end

    end
    final_plot = plot!(legend = false)
    final_plot = plot!(xaxis=:log)
    #final_plot = plot!(xlims=(delta_tests_arr[1],Inf))

    return final_plot
end




#------------------initializing, running and plotting----------------
distribution = mvnormal
dimension = 16
nsamples = 5*10^5
runs = 4

direction_algos = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]
direction_algos = [StochasticReflectDirection()]
delta_tests_arr = create_delta_array(0.001, 0.1, 20)


#---
mean_accs, std_accs, ess_arrays, mean_diffs = run_all_algos(delta_tests_arr, nsamples, runs, distribution, dimension, direction_algos)

best_accs, best_mfps = find_mean(ess_arrays, delta_tests_arr, mean_accs, direction_algos)

# result: best_accs = refresh: 0.3452185204245507  reverse: 0.6312359472313176  reflect: 0.6497585289605811  stochasticreflect: 0.6156345388184439

#---
p = plot_accs(delta_tests_arr, mean_accs, std_accs, ess_arrays, direction_algos)

p_ess = plot_mean_ess(delta_tests_arr, mean_accs, ess_arrays, direction_algos)

p_diff = plot_mean_diffs(delta_tests_arr, mean_accs, mean_diffs, ess_arrays, direction_algos)

#

png("mean_of_diff_to_mean_all_directions")



acc = 0.61
mfp = acc/(1-acc)




#---------saved----------

to_save = [mean_accs, std_accs, ess_arrays, mean_diffs];
serialize("ben_study_plots/tuning/saves/saved_fixed_delta_plots.jls", to_save);
#used values:
#distribution = mvnormal
#dimension = 16
#nsamples = 5*10^5
#runs = 4
#direction_algos = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]
#delta_tests_arr = create_delta_array(0.001, 0.1, 20)

saved = deserialize("ben_study_plots/tuning/saves/saved_fixed_delta_plots.jls");
mean_accs = saved[1]
std_accs = saved[2]
ess_arrays = saved[3]
mean_diffs = saved[4]

# result: best_accs = refresh: 0.3452185204245507  reverse: 0.6312359472313176  reflect: 0.6497585289605811  stochasticreflect: 0.6156345388184439