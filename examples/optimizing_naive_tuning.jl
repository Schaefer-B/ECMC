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

gr(size=(1.3*850, 1.3*600), thickness_scaling = 1.5)



#---- Multivariate Gaussian --------------------------
D = 16
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

#-----------------


posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))


#--------------------------

include("../ecmc.jl")


algorithm = ECMCSampler(
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
    tuning = MFPSTuner(adaption_scheme=NaiveAdaption()),
)

posterior_notrafo = convert(AbstractMeasureOrDensity, posterior)
posterior_transformed, trafo = transform_and_unshape(algorithm.trafo, posterior_notrafo)
shape = varshape(posterior_transformed)



@with_kw mutable struct TrackingState
    params_arr::Vector{Vector{Float64}} = []
    av_steps_arr::Vector{Float64} = []

end




function create_tuner_states(density, algorithm, nchains, params)
    D = totalndof(density)

    initial_samples = [rand(BAT.getprior(density)) for i in 1:nchains]
    lift_vectors = [refresh_lift_vector(D) for i in 1:nchains]
    


    low = 10^-4 #rand(Uniform(10^-4, 10^-3))
    high = 10^4 #rand(Uniform(10^3, 10^4))

    delta = [rand([low, high]) for i in 1:nchains]
    #delta = algorithm.step_amplitude

    

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



function average_steps!(params, nchains=10, tracking_state = TrackingState())
    tuner_states = create_tuner_states(posterior_transformed, algorithm, nchains, params)


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

    plot_av_steps = plot(tstate.av_steps_arr, lw=2, label="Average steps", ylabel="Average steps")
    plot!(title="Optimizing of NaiveAdaption parameters")

    plot!(xlabel="Iterations")

    plot_first_param = plot([tstate.params_arr[x][1] for x in eachindex(tstate.params_arr)], lw=2, ylabel="Parameter 1")

    plot!(xlabel="Iterations")

    plot_second_param = plot([tstate.params_arr[x][2] for x in eachindex(tstate.params_arr)], lw=2, ylabel="Parameter 2")

    plot!(xlabel="Iterations")

    plot_thrid_param = plot([tstate.params_arr[x][3] for x in eachindex(tstate.params_arr)], lw=2, ylabel="Parameter 3")

    plot!(xlabel="Iterations")

    plot_fourth_param = plot([tstate.params_arr[x][4] for x in eachindex(tstate.params_arr)], lw=2, ylabel="Parameter 4")

    plot!(xlabel="Iterations")

    return plot(plot_av_steps, plot_first_param, plot_second_param, plot_thrid_param, plot_fourth_param, layout=(2,3) ,legend=false)
end



#using optim to find better params for naive tuning and plot the tuning
function optimize_and_plot(sparams, nchains = 10, max_iter_steps = 10)

    tracking_state = TrackingState()



    res = optimize(params -> average_steps!(params, nchains, tracking_state),
    sparams,
    method = NelderMead(),
    g_tol = 1e-8,
    iterations = max_iter_steps,
    store_trace = true,
    show_trace = false
    )

    optimal_parameters = Optim.minimizer(res)

    p_final = plot_optim(tracking_state)


    return optimal_parameters, p_final, tracking_state
end




#--------------actually doing it---------------------------
p1::Float64 = 0.5 # strongness of answer to incorrect acc
p2::Float64 = 0.2 # strongness of suppression when at correct acc
p3::Float64 = 0.09 # strongness of suppression if delta changes to rapidly
p4::Float64 = 0.2 # changes how fast the error between acc and target is considered low
starting_params = log.([p1,p2,p3,p4])
chains_to_evaluate = 100 # the higher the less fluctuations in average steps evaluated
max_optim_iterations = 10 # optim changes the params 5 times as often as this value
# CAREFUL! max iterations of the tuning = chains_to_evaluate * max_optim_iterations * 5
optimal_parameters, final_plot, t_state = optimize_and_plot(starting_params, chains_to_evaluate, max_optim_iterations)



#view the results
optimal_parameters
true_parameters = exp.(optimal_parameters) # to replace the "exp(params[1])" etc in ecmc tuning
#1.020666210720955
#0.19549699103761542
#0.08396508129778243
#0.5324737893014809
final_plot
png("NelderMead nchains 100 maxiter 10")
change = (exp.(starting_params) - true_parameters)./exp.(starting_params)
