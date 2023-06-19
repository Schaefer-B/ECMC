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

function mvnormal(dimension)
    D = dimension
    μ = fill(0.0, D)
    σ = fill(1.0, D)

    likelihood = let D = D, μ = μ, σ = σ
        logfuncdensity(params -> begin

            return logpdf(MvNormal(μ, σ), params.a)
        end)
    end 

    prior = BAT.NamedTupleDist(
        a = Uniform.(-4*σ, 4*σ)
    )
    return likelihood, prior
end


#-----------functions needed------------


function create_algorithm(delta, nsamples, Direction, step_var=0.01, var_type = NoVariation(), ch_length=10, jb_refresh=50, chains=1)
    algorithm = ECMCSampler(
        trafo = NoDensityTransform(),#PriorToUniform(),
        nsamples= nsamples,
        nburnin = 0,
        nchains = chains,
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



function run_sampling(posterior, algorithm)

    sample = bat_sample(posterior, algorithm)
    samples = sample.result.v.a

    return samples
end



function run_all_algos(distribution, dimension, dir_algos, nsamples, ch_length, jb_refresh, step_var=0.01, var_type=NoVariation(), delta = 0.1)
    likelihood, prior = distribution(dimension)
    posterior = PosteriorMeasure(likelihood, prior)

    algorithms = [create_algorithm(delta, nsamples, dir_algos[d], step_var, var_type, ch_length, jb_refresh) for d=eachindex(dir_algos)]

    
    samples = Vector{Vector{Vector{Float64}}}(undef, length(dir_algos))
    for algo_index in eachindex(dir_algos)
        samples[algo_index] = run_sampling(posterior, algorithms[algo_index])
    end

    return samples
end


#----------------functions for plotting------------------

function plot_one_sampling_run(dir_algo, samples, chain_length, jumps_before_refresh, fake_chain_length, distribution, dimension)
   
    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    final_plot = plot(layout=(1,1), title=string(dir_algo))


    #---
    likelihood, prior = distribution(dimension)
    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = create_algorithm(0.01, 0.5*10^5, ReflectDirection(), 0.1, UniformVariation(), 10, 50, 2)

    res = bat_sample(posterior, algorithm);
    res_samples = res.result
    final_plot = plot!(res_samples, (:(a[1]), :(a[2])), mean = false, std = false, globalmode = false, marginalmode = false, nbins = 350, alpha=0.65)
    #---

    x = [samples[i][1] for i=eachindex(samples)]
    y = [samples[i][2] for i=eachindex(samples)]

    x_add = 0.18
    final_plot = scatter!([x[1]], [y[1]], msize=10, color=:green)
    final_plot = annotate!([x[1]+x_add], [y[1]], text(string("Start"), :black, :left, 15))

    final_plot = scatter!([x[end]], [y[end]], msize=10, color=:red)
    final_plot = annotate!([x[end]+x_add], [y[end]], text(string("End"), :black, :left, 15))

    k = length(x)
    for sample_index in 1:k-1
        final_plot = plot!(x[sample_index:sample_index+1], y[sample_index:sample_index+1], line=:arrow, lw=2)
    
        if (sample_index*chain_length) % jumps_before_refresh == 0
            final_plot = scatter!([x[sample_index]], [y[sample_index]], msize=10, color=:blue)
            final_plot = annotate!([x[sample_index]+x_add], [y[sample_index]], text(string("Refresh"), :black, :left, 15))
        end

        if sample_index % fake_chain_length == 0
            id = Int(sample_index/fake_chain_length)
            final_plot = scatter!([x[sample_index]], [y[sample_index]], msize=8, color=:yellow)
            final_plot = annotate!([x[sample_index]+x_add], [y[sample_index]], text(string("Sample ", id), :black, :left, 12))
        end
        
    end


    final_plot = plot!(xlims=(-4.1, 4.1), ylims=(-4.1, 4.1))
    final_plot = plot!(legend=false)
    return final_plot
end



#------------------initializing, running and plotting----------------
nsamples = 30
distribution = mvnormal
dimension = 2

chain_length = 1
fake_chain_length = 9
jumps_before_refresh = 20

direction_algorithms = [RefreshDirection(), ReverseDirection(), ReflectDirection(), StochasticReflectDirection()]

samples = run_all_algos(distribution, dimension, direction_algorithms, nsamples, chain_length, jumps_before_refresh)

algo_index = 1
plot_one_sampling_run(direction_algorithms[algo_index], samples[algo_index], chain_length, jumps_before_refresh, fake_chain_length, distribution, dimension)

png(string(direction_algorithms[algo_index], "_visualization"))

