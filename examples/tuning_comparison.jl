using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using BenchmarkTools
using Optim
using StatsBase

gr(size=(1.3*850, 1.3*600), thickness_scaling = 1.5)



#---- Multivariate Gaussian --------------------------
function MVGauss(dimension=16)
    D = dimension
    μ = fill(0.0, D)
    σ = collect(range(1, 10, D))
    truth = rand(MvNormal(μ, σ), Int(1e6))

    likelihood = let D = D, μ = μ, σ = σ
        logfuncdensity(params -> begin

            return logpdf(MvNormal(μ, σ), params.a)
        end)
    end 


    prior = BAT.NamedTupleDist(
        a = Uniform.(-5*σ, 5*σ)
    )
    return likelihood, prior
end

#---- Funnel  --------------------------
function Funnel(dimension=16)
    D = dimension
    truth = rand(BAT.FunnelDistribution(a=0.5, b=1., n=D), Int(1e6))

    likelihood = let D = D
        logfuncdensity(params -> begin

        return logpdf(BAT.FunnelDistribution(a=0.5, b=1., n=D), params.a)
        end)
    end 

    σ = 10*ones(D)
    prior = BAT.NamedTupleDist(
        a = Uniform.(-σ, σ)
    )
    return likelihood, prior
end

#-------------------------------------------
function Mixture()
    likelihood = logfuncdensity(params -> begin

        r1 = logpdf.(
        MixtureModel(Normal[
        Normal(-10.0, 1.2),
        Normal(0.0, 1.8),
        Normal(10.0, 2.5)], [0.1, 0.3, 0.6]), params.a[1])

        r2 = logpdf.(
        MixtureModel(Normal[
        Normal(-5.0, 2.2),
        Normal(5.0, 1.5)], [0.3, 0.7]), params.a[2])

        r3 = logpdf.(Normal(2.0, 1.5), params.a[3])

        return r1+r2+r3
    end)

    prior = BAT.NamedTupleDist(
        a = [-20..20, -20.0..20.0, -10..10]
    )

    return likelihood, prior
end


likelihood, prior = Funnel()

posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))


#--------------------------

include("../ecmc.jl")


algorithm = ECMCSampler(
    #trafo = NoDensityTransform(), 
    trafo = PriorToUniform(),
    nsamples=10^4,
    nburnin = 0,
    nchains = 1,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=10^-1,
    factorized = false,
    #step_var=1.5*0.04,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(max_n_steps=3*10^4, adaption_scheme=NaiveAdaption()),
)

posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
posterior_transformed, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
shape = varshape(posterior_transformed)



@with_kw mutable struct TrackingState
    params_arr::Vector{Vector{Float64}} = []
    av_steps_arr::Vector{Float64} = []
    iteration_tracker::Int64 = 0

end




function create_tuner_states(density, algorithm, nchains, params)
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:nchains]
    


    low = 10^-4 #rand(Uniform(10^-4, 10^-3))
    high = 10^4 #rand(Uniform(10^3, 10^4))
    delta = []
    for i in 1:nchains
        if i < Int(floor(nchains/2))
            push!(delta, low)
        else
            push!(delta, high)
        end
    end
    #delta = algorithm.step_amplitude

    
    params = abs.(params)

    ecmc_tuner_states = [ECMCTunerState(
        C = initial_samples[i], 
        lift_vector = lift_vectors[i], 
        delta = delta[i], 
        tuned_delta = delta[i], 
        remaining_jumps_before_refresh = algorithm.remaining_jumps_before_refresh, 
        delta_arr = [delta[i], ],
        params = params
        )  for i in 1:nchains]

    return ecmc_tuner_states
end





#----------------for finding the best naiveadaption parameters----------
function average_steps!(params, nchains=10, tracking_state = TrackingState())
    tuner_states = create_tuner_states(posterior_transformed, algorithm, nchains, params)

    tracking_state.iteration_tracker += 1
    println()
    println("Currently at Iteration: ", tracking_state.iteration_tracker)
    println()
    steps_per_chain = Vector{Float64}(undef, nchains)
    #Threads.@threads 
    Threads.@threads for c in 1:nchains
        samples_per_chain, tuned_state = _ecmc_tuning(algorithm.tuning, posterior_transformed, algorithm, ecmc_tuner_state=tuner_states[c], chainid=c) 
        steps_per_chain[c] = tuned_state.n_steps
    end

    sum_steps = 0
    for i in eachindex(steps_per_chain)
        sum_steps += steps_per_chain[i]
    end

    av_steps = sum_steps/nchains
    push!(tracking_state.params_arr, params)
    push!(tracking_state.av_steps_arr, av_steps)

    return av_steps
end


function plot_optim(tstate)
    
    p = plot(layout=(3,3), legend=false)

    plot!(tstate.av_steps_arr, subplot=1, lw=2, label="Average steps", ylabel="Average steps")
    plot!(title="Optimizing of NaiveAdaption parameters", subplot=1)
    plot!(ylims=(0, Inf), subplot=1)
    #plot!(title="Optimizing of NaiveAdaption parameters")

    plot!(xlabel="Iterations")

    
    for k in eachindex(tstate.params_arr[1])
        plot!([tstate.params_arr[x][k] for x in eachindex(tstate.params_arr)], subplot=k+1, lw=2, ylabel="Parameter $k", xlabel="Iterations")
        plot!(xlabel="Iterations", subplot=k+1)
    end


    return p
end

#using optim to find better params for naive tuning and plot the tuning
function optimize_and_plot(sparams, optim_algo=NelderMead(), nchains = 10, max_iter_steps = 10)

    tracking_state = TrackingState()


    res = optimize(params -> average_steps!(params, nchains, tracking_state),
    sparams,
    method = optim_algo,
    g_tol = 1e-8,
    iterations = max_iter_steps,
    store_trace = false,
    show_trace = false
    )

    optimal_parameters = Optim.minimizer(res)

    p_final = plot_optim(tracking_state)


    return optimal_parameters, p_final, tracking_state
end




#--------------actually finding the optimal parameters---------------------------
include("../ecmc.jl")
p1::Float64 = 12. # strongness of answer to incorrect acc
p2::Float64 = 0.19 # strongness of suppression when at correct acc
p3::Float64 = 0.08 # strongness of suppression if delta changes to rapidly
p4::Float64 = 0.4 # changes how fast the error between acc and target is considered low
p5::Float64 = 30. # error factor in context with p2
p6::Float64 = 1.2 # minor stuff related to p5
p7::Float64 = 1.1 # minor stuff related to p5
p8::Float64 = 120. # evaluation steps

starting_params = ([p1,p2,p3,p4,p5,p6,p7,p8])


chains_to_evaluate = 40 # the higher the less fluctuations in average steps evaluated
max_optim_iterations = 20 # optim changes the params 5 times as often as this value
# CAREFUL! max iterations of the tuning = chains_to_evaluate * max_optim_iterations * (approx:)5


optim_algo = ParticleSwarm(
                lower = [0.4, 0.05, 0.05, 0., 1., 0.1, 0.],
                upper = [2., 0.5, 0.5, 1., 50., 50., 3.],
                n_particles = 5
)

optim_algo = NelderMead()


optimal_parameters, final_plot, t_state = optimize_and_plot(starting_params, optim_algo, chains_to_evaluate, max_optim_iterations)



#view the results
optimal_parameters
true_parameters = abs.(optimal_parameters) # to replace the "exp(params[1])" etc in ecmc tuning

#nelder mead:
15.348506323247225
0.207210542888343
0.0732514723260891
0.4934509024569294
17.587673168668637
2.069505973296211
0.6136869940758715
163.53188824455017

#particle ParticleSwarm
1.0314262450574858
0.1975041889054433
0.10281399554024151
0.49954896991395603
19.99712649821221
0.1
1.00856400097384

final_plot = plot_optim(t_state)
png("if i misclick")
png("NelderMead")
png("ParticleSwarm nchains 100 maxiter 20 particles 4")
change = ((starting_params) - true_parameters)./(starting_params)














#---------
#-----------------calculate the average steps without anything else--------
#---------



function calculate_steps(nchains, likelihood, prior, tuning, direction_change = RefreshDirection())

    posterior = PosteriorMeasure(likelihood, prior);
    logdensityof(posterior, rand(prior))


    algorithm = ECMCSampler(
        #trafo = NoDensityTransform(), 
        trafo = PriorToUniform(),
        nsamples=10^4,
        nburnin = 0,
        nchains = 1,
        chain_length=5, 
        remaining_jumps_before_refresh=50,
        step_amplitude=10^-1,
        factorized = false,
        #step_var=1.5*0.04,
        direction_change = direction_change,
        tuning = tuning,
    )

    posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
    posterior_transformed, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
    shape = varshape(posterior_transformed)


    params = [0.98,
    0.207210542888343,
    0.0732514723260891,
    0.4934509024569294,
    17.587673168668637,
    2.069505973296211,
    0.6136869940758715,
    163.53188824455017]

    tuner_states = create_tuner_states(posterior_transformed, algorithm, nchains, params)

    steps_per_chain = Vector{Float64}(undef, nchains)
    deltas_per_chain = Vector{Float64}(undef, nchains)

    
    Threads.@threads for c in 1:nchains
        samples_per_chain, tuned_state = _ecmc_tuning(algorithm.tuning, posterior_transformed, algorithm, ecmc_tuner_state=tuner_states[c], chainid=c) 
        steps_per_chain[c] = tuned_state.n_steps
        deltas_per_chain[c] = tuned_state.tuned_delta
    end

    n_notconverged = 0
    for c in 1:nchains
        if steps_per_chain[c] == tuning.max_n_steps
            n_notconverged += 1
        end
    end
    
    av_result = mean(steps_per_chain)
    return av_result, steps_per_chain, n_notconverged, deltas_per_chain
end



function step_infos(likelihood, prior, tuning, nchains, direction_algos)
    dir_len = length(direction_algos)
    av_steps = Vector{Float64}(undef, dir_len)
    chain_steps = Vector{Vector{Float64}}(undef, dir_len)
    n_notconverged = Vector{Int64}(undef, dir_len)
    chain_deltas = Vector{Vector{Float64}}(undef, dir_len)

    for algo_index in eachindex(direction_algos)
        av_steps[algo_index], chain_steps[algo_index], n_notconverged[algo_index], chain_deltas[algo_index] = calculate_steps(nchains, likelihood, prior, tuning, direction_algos[algo_index])
    end

    return av_steps, chain_steps, n_notconverged, chain_deltas
end



function plot_histos(bsize_steps, nbins_delta, chain_steps, chain_deltas, tuning, direction_algos)


    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    histo_length = length(chain_steps)
    histo_plot = plot(layout=(histo_length, 2))
    #nchains = length(chain_steps[1])
    #title = string(tuning.adaption_scheme)
    #plot!(title = title, subplot = 1)


    #highest_steps = Vector{Int64}(undef, histo_length)
    ulimit_steps = Vector{Int64}(undef, histo_length)
    highest_steps = Vector{Float64}(undef, histo_length)
    lowest_steps = Vector{Float64}(undef, histo_length)
    highest_deltas = Vector{Float64}(undef, histo_length)
    lowest_deltas = Vector{Float64}(undef, histo_length)
    for histo_index in 1:histo_length
        highest_steps[histo_index] = maximum(chain_steps[histo_index])
        lowest_steps[histo_index] = minimum(chain_steps[histo_index])
        ulimit_steps[histo_index] = (Int(floor(maximum(highest_steps[histo_index])/bsize_steps))+1)*bsize_steps
        highest_deltas[histo_index] = maximum(chain_deltas[histo_index])
        lowest_deltas[histo_index] = minimum(chain_deltas[histo_index])
    end

    #ulimit_steps = (Int(floor(maximum(highest_steps)/bsize_steps))+1)*bsize_steps
    


    #b_range = range(0, tuning.max_n_steps+1, length=Int(floor((tuning.max_n_steps+1)/bsize_steps)))
    
    
    for histo_index in 1:histo_length
        label = string(direction_algos[histo_index])

        #steps
        #histo_plot = plot!(chain_steps[histo_index], xlabel="Steps", ylabel="Counts", subplot=(2*histo_index-1), st=:histogram, title=label, bins=b_range)
        high = highest_steps[histo_index]
        low = lowest_steps[histo_index]
        diff = high - low
        histogram = fit(Histogram, chain_steps[histo_index], nbins = Int(floor((diff)/(bsize_steps))))
        histo_plot = plot!(histogram, xlabel="Steps", ylabel="Counts", subplot=(2*histo_index-1), title=label)
        histo_plot = plot!(xlims=(0, ulimit_steps[histo_index]), subplot=(2*histo_index-1))

        steps_mean = mean(chain_steps[histo_index])
        histo_plot = plot!([steps_mean], st=:vline, subplot=(2*histo_index-1), lw=2, color=:red)

        x_translation = diff*0.01
        annotate!((steps_mean+x_translation), maximum(histogram.weights)*1.13, text(string(Int(round(steps_mean, digits=0))), :red, :left, 7), subplot=(2*histo_index-1))

        #deltas
        high = highest_deltas[histo_index]
        low = lowest_deltas[histo_index]
        diff = high - low
        histogram = fit(Histogram, chain_deltas[histo_index], nbins = nbins_delta)
        histo_plot = plot!(histogram, xlabel="Delta", ylabel="Counts", subplot=(2*histo_index), title=label, ylims=(0, Inf))

        delta_mean = mean(chain_deltas[histo_index])
        histo_plot = plot!([delta_mean], st=:vline, subplot=(2*histo_index), lw=2, color=:red)

        x_translation = diff*0.01
        annotate!((delta_mean+x_translation), maximum(histogram.weights)*1.13, text(string(round(delta_mean, digits=5)), :red, :left, 7), subplot=(2*histo_index))
    end
    histo_plot = plot!(legend = false)

    return histo_plot
end

#-----------------
include("../ecmc.jl")


nchains = 1000 # to average over
bucketsize_for_steps = 200
nbins_delta = 40

likelihood, prior = Funnel()
tuning = MFPSTuner(max_n_steps=3*10^4, adaption_scheme=GoogleAdaption())
#average_steps, steps_per_tuning, n_notconverged, deltas_per_tuning = calculate_steps(100, likelihood, prior, tuning, RefreshDirection()) # for one test
direction_algos = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]

av_steps, chain_steps, n_notconverged, chain_deltas = step_infos(likelihood, prior, tuning, nchains, direction_algos);

histo = plot_histos(bucketsize_for_steps, nbins_delta, chain_steps, chain_deltas, tuning, direction_algos)

av_steps
n_notconverged
percent_notconverged = n_notconverged/nchains

png(string(string(tuning.adaption_scheme), ", 16D-Funnel, over $nchains tunings, notconverged = $n_notconverged"))
