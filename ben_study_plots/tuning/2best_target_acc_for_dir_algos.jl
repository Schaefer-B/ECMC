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



#----------------strcuts-------------------
@with_kw mutable struct DeltaState
    boundary_deltas::Vector{Float64}
    boundary_ess::Vector{Float64}

    all_deltas::Vector{Float64}
    all_ess::Vector{Float64}
    all_ess_stds::Vector{Float64}

    direction_algorithm
    distribution
    dimension::Int64
    nsamples::Int64
    nchains::Int64
    depth::Int64
end



#----------------functions for sampling------------------
function mid_delta(deltas)
    logdelta_1 = log(deltas[1])
    logdelta_2 = log(deltas[2])

    logmiddle = (logdelta_1 + logdelta_2)/2

    return exp(logmiddle)
end


function new_boundaries(old_boundary_deltas, new_delta, old_boundary_ess, new_ess)
    index = 0
    old_boundary_ess[1] >= old_boundary_ess[2] ? index = 1 : index = 2
    if index == 1
        new_boundary_deltas = [old_boundary_deltas[1], new_delta]
        new_boundary_ess = [old_boundary_ess[1], new_ess]
    else
        new_boundary_deltas = [new_delta, old_boundary_deltas[2]]
        new_boundary_ess = [new_ess, old_boundary_ess[2]]
    end

    return new_boundary_deltas, new_boundary_ess
end


function create_algorithm(delta, nsamples, Direction, nchains, step_var=0., var_type = NoVariation(), ch_length=10, jb_refresh=50)
    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= nsamples,
        nburnin = Int(floor(0.01*nsamples)),
        nchains = nchains,
        chain_length=ch_length, 
        remaining_jumps_before_refresh = jb_refresh,
        step_amplitude = delta,
        factorized = false,
        step_var = step_var,
        variation_type = var_type,
        direction_change = Direction,
        tuning = ECMCNoTuner(),
    )
    return algorithm
end


function get_mean_ess(posterior, algorithm)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    return mean(ess), std(ess)
end



function one_step_deeper!(d_state)

    @unpack boundary_deltas, boundary_ess, depth = d_state

    old_boundary_deltas = boundary_deltas
    old_boundary_ess = boundary_ess

    likelihood, prior = d_state.distribution(d_state.dimension)
    posterior = PosteriorMeasure(likelihood, prior)


    new_delta = mid_delta(boundary_deltas)
    algorithm = create_algorithm(new_delta, d_state.nsamples, d_state.direction_algo, d_state.nchains)

    new_ess, new_ess_std = get_mean_ess(posterior, algorithm)

    boundary_deltas, boundary_ess = new_boundaries(old_boundary_deltas, new_delta, old_boundary_ess, new_ess)

    depth += 1

    @pack! d_state = boundary_deltas, boundary_ess, depth

    push!(d_state.all_deltas, new_delta)
    push!(d_state.all_ess, new_ess)
    push!(d_state.all_ess_stds, new_ess_std)

    return d_state
end


function initialize_delta_state(deltas, direction_algo, distribution, dimension, nsamples, nchains)


    algorithms = [create_algorithm(deltas[i], nsamples, direction_algo, nchains) for i in 1:2]

    likelihood, prior = distribution(dimension)
    posterior = PosteriorMeasure(likelihood, prior)

    lower_b_ess, lower_b_std = get_mean_ess(posterior, algorithms[1])
    higher_b_ess, higher_b_std = get_mean_ess(posterior, algorithms[2])

    d_state  = DeltaState(
        boundary_deltas = deltas,
        boundary_ess = [lower_b_ess, higher_b_ess],
        all_deltas = [deltas[1], deltas[2]],
        all_ess = [lower_b_ess, higher_b_ess],
        all_ess_stds = [lower_b_std, higher_b_std],
        direction_algorithm = direction_algo,
        distribution = distribution,
        dimension = dimension,
        nsamples = nsamples,
        nchains = nchains,
        depth = 0,
    )
    
    return d_state
end


function initialize_multiple_delta_states(deltas, direction_algos, distributions, dimensions, nsamples, nchains)
    d_states = []
    for dir_algo in direction_algos, dist in distributions, dim in dimensions
        state = initialize_delta_state(deltas, dir_algo, dist, dim, nsamples, nchains)
        push!(d_states, state)
    end
    return d_states
end


function multi_run_till_depth_reached(deltas, direction_algos, distributions, dimensions, nsamples, nchains, max_depth)
    d_states = initialize_multiple_delta_states(deltas, direction_algos, distributions, dimensions, nsamples, nchains)

    for d_state in d_states
        while d_state.depth < max_depth
            one_step_deeper!(d_state)
        end
    end

    return d_states
end

function multi_run_till_depth_reached(d_states::Vector{DeltaState}, max_depth)

    for d_state in d_states
        while d_state.depth < max_depth
            one_step_deeper!(d_state)
        end
    end

    return d_states
end



#------------------------START: old stuff----------------------------





function create_algorithm(delta, nsamples, Direction, step_var=0., var_type = NoVariation(), ch_length=10, jb_refresh=50)
    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= nsamples,
        nburnin = 0,
        nchains = 1,
        chain_length=ch_length, 
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

#creates an array where first the dimensions change and then the distributions if called linearly
function get_dist_arr(dists, dims)
    result = Vector{Any}(undef, length(dists)*length(dims))
    names = Vector{String}(undef, length(dists)*length(dims))
    i = 0
    for ds in dists
        for dm in dims
            i += 1
            result[i] = ds(dm)
            names[i] = string(dm, "D ", ds)
        end
    end

    return result, names
end


function ess_run(posterior, algorithm, nchains)
    posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
    density, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
    shape = varshape(density)


    ecmc_states = create_ecmc_states(density, algorithm, nchains)


    acc_per_chain = Vector{Float64}(undef, nchains)
    diff_to_mean_per_chain = Vector{Float64}(undef, nchains)
    samples_per_chain = []

    for c in 1:nchains
        sample = _ecmc_sample(density, algorithm, ecmc_state=ecmc_states[c], chainid=c)
        push!(samples_per_chain, sample)

        samples_trafo = shape.(convert_to_BAT_samples([sample], density))
        samples = inverse(trafo).(samples_trafo)
        acc_per_chain[c] = (ecmc_states[c].n_acc/ecmc_states[c].n_steps)
        diff_to_mean_per_chain[c] = mean(abs.(mean(samples).a))
    end

    newchains = 1
    for i in reverse(collect(2:4))
        if nchains % i == 0
            newchains = Int(nchains/i)
            break
        end
    end
    
    samples_per_newchain = []
    samples = []
    for c in 1:nchains
        push!(samples, samples_per_chain[c])
        if c % newchains == 0
            push!(samples_per_newchain, samples)
            samples = []
        end
    end

    
    ess_per_chain = Vector{Float64}(undef, newchains)
    for nc in 1:newchains
        samples_trafo = shape.(convert_to_BAT_samples(samples_per_newchain[nc], density))
        samples = inverse(trafo).(samples_trafo)
        ess_per_chain[nc] = mean(bat_eff_sample_size(samples).result.a)
    end

    mean_acc = mean(acc_per_chain)
    std_acc = std(acc_per_chain)
    mean_ess = mean(ess_per_chain)
    mean_diff = mean(diff_to_mean_per_chain)


    return mean_acc, std_acc, mean_ess, mean_diff
end



function delta_acc_for_one_algo(delta_tests_arr, nsamples, runs, distribution, dir_algo, step_var=0.1, var_type = NoVariation(),ch_length=10, jb_refresh=50)

    likelihood, prior = distribution;
    posterior = PosteriorMeasure(likelihood, prior);

    algorithms = [create_algorithm(delta_tests_arr[d], nsamples, dir_algo, step_var, var_type, ch_length, jb_refresh) for d=eachindex(delta_tests_arr)]



    k = length(delta_tests_arr)
    mean_acc = Vector{Float64}(undef, k)
    std_acc = Vector{Float64}(undef, k)
    mean_ess_arr = Vector{Float64}(undef, k)
    mean_diff = Vector{Float64}(undef, k)

    Threads.@threads for delta_index in 1:k
        mean_acc[delta_index], std_acc[delta_index], mean_ess_arr[delta_index], mean_diff[delta_index] = ess_run(posterior, algorithms[delta_index], runs) 
    end

    m = maximum(mean_ess_arr)
    m_index = findall(x->x == m ? true : false, mean_ess_arr)
    ideal_delta = delta_tests_arr[m_index]
    ideal_acc = mean_acc[m_index]



    #return mean_acc, std_acc, mean_ess_arr, mean_diff
    return ideal_delta[1], ideal_acc[1]
end


function run_all_algos(delta_tests_arr, nsamples, runs, distributions, dir_algos, step_var=0.1, var_type = NoVariation(), ch_length=10, jb_refresh=50)

    algo_count = length(dir_algos)
    dist_count = length(distributions)
    ideal_deltas = Vector{Vector{Float64}}(undef, algo_count)
    ideal_accs = Vector{Vector{Float64}}(undef, algo_count)


    for a in 1:algo_count
        idel = Vector{Float64}(undef, dist_count)
        iacc = Vector{Float64}(undef, dist_count)
        for d in 1:dist_count
            idel[d], iacc[d] = delta_acc_for_one_algo(delta_tests_arr, nsamples, runs, distributions[d], dir_algos[a], step_var, var_type, ch_length, jb_refresh)
        end
        ideal_deltas[a] = idel
        ideal_accs[a] = iacc
    end

    return ideal_deltas, ideal_accs
end


function plot_accs(ideal_deltas, ideal_accs, direction_algos, distributions)

    
    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    dist_count = length(distributions)
    
    if algo_count > 1
        final_plot = plot(layout=(Int(algo_count/2), 2))
    else
        final_plot = plot(layout=(1,1))
    end
    


    x_values = [string(distributions[i]) for i=eachindex(distributions)]


    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)
        dist_index = eachindex(distributions)
        final_plot = scatter!(dist_index .-0.5, ideal_accs[algo_index], msize=8, title=label, subplot=algo_index, xdiscrete_values=x_values)
        
        final_plot = plot!(xlabel="Distributions", ylabel="Acceptance rate", subplot=algo_index)
        mean_acc = mean(ideal_accs[algo_index])
        final_plot = plot!([mean_acc], st=hline, color=:red, lw=2, subplot=algo_index)
        diff = maximum(ideal_accs[algo_index]) - minimum(ideal_accs[algo_index])
        final_plot = annotate!(0.01*dist_count, mean_acc+diff*0.02, text(string(round(mean_acc, digits=4)), :red, :left, 8), subplot=algo_index)

    end

    final_plot = plot!(legend = false, xlims=(0., length(distributions)))

    return final_plot
end

function plot_deltas(ideal_deltas, ideal_accs, direction_algos, distributions)

    
    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    dist_count = length(distributions)
    
    if algo_count > 1
        final_plot = plot(layout=(Int(algo_count/2), 2))
    else
        final_plot = plot(layout=(1,1))
    end
    


    x_values = [string(distributions[i]) for i=eachindex(distributions)]


    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)
        dist_index = eachindex(distributions)
        final_plot = scatter!(dist_index .-0.5, ideal_deltas[algo_index], msize=8, title=label, subplot=algo_index, xdiscrete_values=x_values)
        
        final_plot = plot!(xlabel="Distributions", ylabel="Delta", subplot=algo_index)
        mean_delta = mean(ideal_deltas[algo_index])
        final_plot = plot!([mean_delta], st=hline, color=:red, lw=2, subplot=algo_index)
        diff = maximum(ideal_deltas[algo_index]) - minimum(ideal_deltas[algo_index])
        final_plot = annotate!(0.01*dist_count, mean_delta+diff*0.02, text(string(round(mean_delta, digits=6)), :red, :left, 8), subplot=algo_index)

    end

    final_plot = plot!(legend = false, xlims=(0., length(distributions)))

    return final_plot
end







#------------------------END: old stuff----------------------------





#----------------functions for plotting------------------

function plot_accs(d_states::Vector{DeltaState})

    
    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    dist_count = length(distributions)
    
    if algo_count > 1
        final_plot = plot(layout=(Int(algo_count/2), 2))
    else
        final_plot = plot(layout=(1,1))
    end
    


    x_values = [string(distributions[i]) for i=eachindex(distributions)]


    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)
        dist_index = eachindex(distributions)
        final_plot = scatter!(dist_index .-0.5, ideal_accs[algo_index], msize=8, title=label, subplot=algo_index, xdiscrete_values=x_values)
        
        final_plot = plot!(xlabel="Distributions", ylabel="Acceptance rate", subplot=algo_index)
        mean_acc = mean(ideal_accs[algo_index])
        final_plot = plot!([mean_acc], st=hline, color=:red, lw=2, subplot=algo_index)
        diff = maximum(ideal_accs[algo_index]) - minimum(ideal_accs[algo_index])
        final_plot = annotate!(0.01*dist_count, mean_acc+diff*0.02, text(string(round(mean_acc, digits=4)), :red, :left, 8), subplot=algo_index)

    end

    final_plot = plot!(legend = false, xlims=(0., length(distributions)))

    return final_plot
end

function plot_deltas(d_states::Vector{DeltaState})

    
    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    algo_count = length(direction_algos)
    dist_count = length(distributions)
    
    if algo_count > 1
        final_plot = plot(layout=(Int(algo_count/2), 2))
    else
        final_plot = plot(layout=(1,1))
    end
    


    x_values = [string(distributions[i]) for i=eachindex(distributions)]


    for algo_index in 1:algo_count
        label = chop(string(direction_algos[algo_index]), head=0, tail=0)
        dist_index = eachindex(distributions)
        final_plot = scatter!(dist_index .-0.5, ideal_deltas[algo_index], msize=8, title=label, subplot=algo_index, xdiscrete_values=x_values)
        
        final_plot = plot!(xlabel="Distributions", ylabel="Delta", subplot=algo_index)
        mean_delta = mean(ideal_deltas[algo_index])
        final_plot = plot!([mean_delta], st=hline, color=:red, lw=2, subplot=algo_index)
        diff = maximum(ideal_deltas[algo_index]) - minimum(ideal_deltas[algo_index])
        final_plot = annotate!(0.01*dist_count, mean_delta+diff*0.02, text(string(round(mean_delta, digits=6)), :red, :left, 8), subplot=algo_index)

    end

    final_plot = plot!(legend = false, xlims=(0., length(distributions)))

    return final_plot
end





#------------------initializing, running and plotting----------------
distributions = [mvnormal, funnel]
dimensions = [8, 32, 128]
nsamples = 10^5 # nsamples needed for a 32D funnel
nchains = 2


direction_algos = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]
direction_algos = [StochasticReflectDirection()]


#---







#---------saved----------

to_save = [ideal_deltas, ideal_accs];
serialize("ben_study_plots/tuning/saves/saved_best_target_accs_plots.jls", to_save);
#used values:
#distribution = mvnormal
#dimension = 16
#nsamples = 5*10^5
#runs = 4
#direction_algos = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]
#delta_tests_arr = create_delta_array(0.001, 0.1, 20)

saved = deserialize("ben_study_plots/tuning/saves/saved_best_target_accs_plots.jls");
ideal_deltas = saved[1]
ideal_accs = saved[2]

