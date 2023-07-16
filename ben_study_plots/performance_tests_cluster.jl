using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using StatsBase
using JLD2
using FileIO
using LaTeXStrings
using TranscodingStreams
using CodecZlib

#using HypothesisTests

include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/ecmc_cluster.jl")
#include("../ecmc.jl")

#anymeasureordensity => 
#salvatore lacanina ceph, arbeiten, bachelorarbeiten
#willy oder robin bachelorarbeiten

# IMPORTANT:
# no hmc states whatsover rn
# create_result_state for hmc states is incomplete
# create_algorithm for hmc states missing as well


#-------------structs-----------------
@with_kw mutable struct TestMeasuresStruct
    effective_sample_size
    ks_p_values # Kolmogorov Smirnov
    chisq_values # can be used to calculate pearson chi square p-values, but also count as a measure on its own
    normalized_residuals # normally distributed

    samples_mode
    samples_mean
    samples_std

    sample_time
end


@with_kw mutable struct ECMCPerformanceState
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburnin::Int64
    nchains::Int64
    tuning_max_n_steps

    adaption_scheme
    direction_change_algorithm
    start_delta
    tuned_deltas
    tuning_steps
    ntunings_not_converged
    step_variance
    variance_algorithm
    target_acc_value
    jumps_before_sample::Int64
    jumps_before_refresh::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct ECMCResultState
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburnin::Int64
    nchains::Int64
    tuning_max_n_steps

    adaption_scheme
    direction_change_algorithm
    start_delta
    tuned_deltas
    tuning_steps
    ntunings_not_converged
    step_variance
    variance_algorithm
    target_acc_value
    jumps_before_sample::Int64
    jumps_before_refresh::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct MCMCPerformanceState
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburninsteps_per_cycle::Int64
    nburnin_max_cycles::Int64
    nchains::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct MCMCResultState
    target_distribution
    dimension::Int64
    nsamples::Int64
    nburninsteps_per_cycle::Int64
    nburnin_max_cycles::Int64
    nchains::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct HMCPerformanceState
    target_distribution
    dimension::Int64
    nsamples::Int64
    nchains::Int64

    samples
    effective_sample_size
end

@with_kw mutable struct HMCResultState
    target_distribution
    dimension::Int64
    nsamples::Int64
    nchains::Int64

    samples
    effective_sample_size
end



#-------------needed functions----------
function get_posterior(distribution, dimension)
    likelihood, prior = distribution(dimension)
    posterior = PosteriorMeasure(likelihood, prior)

    return posterior
end

function get_posterior_dist(target_distribution::Type{MvNormal}, dimension)
    D = dimension
    μ = fill(0.0, D)
    σ = fill(1.0, D)
    dist = target_distribution(μ, σ)
    likelihood = let dist=dist
        logfuncdensity(params -> begin

            return logpdf(dist, params.a)
        end)
    end 

    prior = BAT.NamedTupleDist(
        a = Uniform.(-10*σ, 10*σ)
    )

    posterior = PosteriorMeasure(likelihood, prior)
    return posterior, dist
end

#---------------------------------

function create_algorithm(p_state::ECMCPerformanceState)
    algorithm = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= p_state.nsamples,
        nburnin = p_state.nburnin,
        nchains = p_state.nchains,
        chain_length = p_state.jumps_before_sample, 
        remaining_jumps_before_refresh = p_state.jumps_before_refresh,
        step_amplitude = p_state.start_delta,
        factorized = false,
        step_var = p_state.step_variance,
        variation_type = p_state.variance_algorithm,
        direction_change = p_state.direction_change_algorithm,
        tuning = MFPSTuner(target_acc=p_state.target_acc_value, adaption_scheme=p_state.adaption_scheme, max_n_steps = p_state.tuning_max_n_steps, starting_alpha=0.1),
    )

    first = ECMCSampler(
        trafo = PriorToUniform(),
        nsamples= 2,
        nburnin = 0,
        nchains = p_state.nchains,
        chain_length = p_state.jumps_before_sample, 
        remaining_jumps_before_refresh = p_state.jumps_before_refresh,
        step_amplitude = p_state.start_delta,
        factorized = false,
        step_var = p_state.step_variance,
        variation_type = p_state.variance_algorithm,
        direction_change = p_state.direction_change_algorithm,
        tuning = MFPSTuner(target_acc=p_state.target_acc_value, adaption_scheme=p_state.adaption_scheme, max_n_steps = p_state.tuning_max_n_steps, starting_alpha=0.1),
    )

    return algorithm, first
end

function create_algorithm(p_state::MCMCPerformanceState)
    algorithm = MCMCSampling(
        mcalg = MetropolisHastings(),
        nsteps = p_state.nsamples, 
        nchains = p_state.nchains,
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = p_state.nburninsteps_per_cycle, max_ncycles = p_state.nburnin_max_cycles),
        convergence = BrooksGelmanConvergence(),
        #init = MCMCChainPoolInit(nsteps_init = 1000),
    )

    first = MCMCSampling(
        mcalg = MetropolisHastings(),
        nsteps = 2, 
        nchains = p_state.nchains,
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = p_state.nburninsteps_per_cycle, max_ncycles = p_state.nburnin_max_cycles),
        convergence = BrooksGelmanConvergence(),
        #init = MCMCChainPoolInit(nsteps_init = 1000),
    )
    return algorithm, first
end



#---------------------------------


function run_sampling!(posterior, algorithm, p_state::ECMCPerformanceState)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    for chain_id in 1:p_state.nchains
        t_state = sample.ecmc_tuning_state[chain_id]
        p_state.tuning_steps = t_state.n_steps
        if t_state.n_steps == p_state.tuning_max_n_steps
            p_state.ntunings_not_converged += 1
        end
        e_state = sample.ecmc_state[chain_id]
        push!(p_state.tuned_deltas, e_state.step_amplitude)
    end
    p_state.samples = samples
    p_state.effective_sample_size = ess

    return p_state
end


function run_sampling!(posterior, algorithm, p_state::MCMCPerformanceState)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    p_state.samples = samples
    p_state.effective_sample_size = ess

    return p_state
end


function run_sampling!(posterior, algorithm, p_state::HMCPerformanceState)
    sample = bat_sample(posterior, algorithm)
    samples = sample.result

    ess = bat_eff_sample_size(samples).result.a

    p_state.samples = samples
    p_state.effective_sample_size = ess

    return p_state
end



#---------------------------------
function get_ks_p_values(samples, iid_samples)
    compare_result = BAT.bat_compare(samples, iid_samples).result
    ks_p_values = compare_result.ks_p_values

    return ks_p_values
end


function get_residual_values(samples, iid_samples)
    samples = samples.v.a
    iid_samples = iid_samples.v

    dimensions = length(samples[1])

    chisq_values = []
    normalized_residuals = []

    for d in 1:dimensions
        marginal_samples = [samples[i][d] for i=eachindex(samples)]
        iid_marginal_samples = [iid_samples[i][d] for i=eachindex(iid_samples)]


        bin_start = min(minimum(marginal_samples), minimum(iid_marginal_samples))
        bin_end = max(maximum(marginal_samples), maximum(iid_marginal_samples))
        nbins = 2000
        bin_width = (bin_end - bin_start)/nbins

        samples_hist = fit(Histogram, marginal_samples, bin_start:bin_width:bin_end)
        iid_samples_hist = fit(Histogram, iid_marginal_samples, bin_start:bin_width:bin_end)

        samples_binned = samples_hist.weights
        iid_samples_binned = iid_samples_hist.weights
        
        chisq_value = 0
        for bin in eachindex(iid_samples_binned)
            observed = samples_binned[bin]
            expected = iid_samples_binned[bin]

            if expected > 10
                residual = (observed - expected)
                standard_deviation = sqrt(expected)

                chisq_value += (residual)^2 /expected

                normalized_residual = residual/standard_deviation
                push!(normalized_residuals, normalized_residual)
            end
        end
        push!(chisq_values, chisq_value)  
    end

    chisq_values_arr = [chisq_values[i] for i=eachindex(chisq_values)]
    normalized_residuals_arr = [normalized_residuals[i] for i=eachindex(normalized_residuals)]

    return chisq_values_arr, normalized_residuals_arr
end


function calculate_test_measures(samples, effective_sample_size, nsamples, nchains, sample_time, dist)

    nsamples_iid = nsamples * nchains
    samples = samples
    iid_sample = bat_sample(dist, IIDSampling(nsamples=nsamples_iid))
    iid_samples = iid_sample.result

    effective_sample_size_arr = [effective_sample_size[i] for i=eachindex(effective_sample_size)]

    samples_mode = mode(samples)
    samples_mode_arr = [samples_mode.a[i] for i = eachindex(samples_mode.a)]
    samples_mean = mean(samples)
    samples_mean_arr = [samples_mean.a[i] for i = eachindex(samples_mean.a)]
    samples_std = std(samples)
    samples_std_arr = [samples_std.a[i] for i = eachindex(samples_std.a)]

    ks_p_values = get_ks_p_values(samples, iid_samples)
    chisq_values, normalized_residuals = get_residual_values(samples, iid_samples)

    t = TestMeasuresStruct(
        effective_sample_size = effective_sample_size_arr,
        ks_p_values = ks_p_values,
        chisq_values = chisq_values,
        normalized_residuals = normalized_residuals,
        samples_mode = samples_mode_arr,
        samples_mean = samples_mean_arr,
        samples_std = samples_std_arr,
        sample_time = sample_time,
    )

    return t
end

#---------------------------------

function create_result_state(p_state::ECMCPerformanceState)
    #println("create result state in ecmc")
    r_state = ECMCResultState(
    target_distribution = string(p_state.target_distribution),
    dimension = p_state.dimension,
    nsamples = p_state.nsamples,
    nburnin = p_state.nburnin,
    nchains = p_state.nchains,
    tuning_max_n_steps = p_state.tuning_max_n_steps,

    adaption_scheme = string(p_state.adaption_scheme),
    direction_change_algorithm = string(p_state.direction_change_algorithm),
    start_delta = p_state.start_delta,
    tuned_deltas = p_state.tuned_deltas,
    tuning_steps = p_state.tuning_steps,
    ntunings_not_converged = p_state.ntunings_not_converged,
    step_variance = p_state.step_variance,
    variance_algorithm = string(p_state.variance_algorithm),
    target_acc_value = p_state.target_acc_value,
    jumps_before_sample = p_state.jumps_before_sample,
    jumps_before_refresh = p_state.jumps_before_refresh,

    samples = p_state.samples,
    effective_sample_size = p_state.effective_sample_size,
    )

    return r_state
end



function create_result_state(p_state::MCMCPerformanceState)
    #println("create result state in mcmc")
    r_state = MCMCResultState(
    target_distribution = string(p_state.target_distribution),
    dimension = p_state.dimension,
    nsamples = p_state.nsamples,
    nburninsteps_per_cycle = p_state.nburninsteps_per_cycle,
    nburnin_max_cycles = p_state.nburnin_max_cycles,
    nchains = p_state.nchains,

    samples = p_state.samples,
    effective_sample_size = p_state.effective_sample_size,
    )
    
    return r_state
end


function create_result_state(p_state::HMCPerformanceState)

    return r_state
end

#---------------------------------

function one_state_run(p_state, runs=1, start_id=1, save_all_samples=false)

    posterior, dist = get_posterior_dist(p_state.target_distribution, p_state.dimension)

    algorithm, first_run_algo = create_algorithm(p_state)

    p_temp = p_state
    temp_time = @elapsed(run_sampling!(posterior, first_run_algo, p_temp))

    
    for run in 1:runs
        run_id = run + start_id - 1
        #println("Starting run $run_id for ", string(p_state.target_distribution, p_state.dimension,"D ", p_state.direction_change_algorithm))
        sample_time = @elapsed(run_sampling!(posterior, algorithm, p_state))
        #println("Finished run $run_id for ", string(p_state.target_distribution, p_state.dimension,"D ", p_state.direction_change_algorithm))

        
        if save_all_samples == true
            result_state = create_result_state(p_state)
            save_state(result_state, run_id)
            result_state.samples = []
            result_state.effective_sample_size = []
        end

        testmeasures = calculate_test_measures(p_state.samples, p_state.effective_sample_size ,p_state.nsamples, p_state.nchains, sample_time, dist)
        if run_id == start_id
            sample_plot = plot_samples(p_state.samples)
            save_sample_plot(p_state, sample_plot, run_id)
        end
        p_state.samples = []
        p_state.effective_sample_size = []

        save_test_measures(p_state, testmeasures, run_id)
        
    end

end


function multiple_states_run(p_states, runs=1, start_id=1, save_all_samples=false)
    #println("Starting all runs over a perfomance_state array")
    if p_states == []
        return []
    end
    nstates = length(p_states)
    #result_states = Vector{Any}(undef, nstates)
    for p_index in 1:nstates
        #println("Starting runs for performance_state $p_index")
        one_state_run(p_states[p_index], runs, start_id, save_all_samples)
        #println("Finished runs for performance_state $p_index")
    end
    #println("Finished all runs over a performance_state array")
end


function all_states_run(ecmc_states, mcmc_states, hmc_states, runs)

    multiple_states_run(ecmc_states, runs)
    println()
    println("ecmc runs finished")
    println()

    multiple_states_run(mcmc_states, runs)
    println()
    println("mcmc runs finished")
    println()

    multiple_states_run(hmc_states, runs)
    println()
    println("hmc runs finished")
    println()

    
end


#----------------saving functions-------------

function save_state(p_state::ECMCResultState, run_id=1)
    location = "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/"
    sampler = "ecmc/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.target_acc_value, "target_acc_", p_state.jumps_before_refresh, "jbr")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,name,name_add,extension)
    save(full_name, Dict("state" => p_state); compress = true)
end


function save_test_measures(p_state::ECMCPerformanceState, testmeasures, run_id=1) # to save: ess, mean(samples), std(samples), ks-test-p-values, ad-test-p-values, p-chi-p-values, time-elapsed, plot(samples 1-5d)
    location = "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/"
    sampler = "ecmc/"
    location_add = "test_measures/"
    name = string(p_state.direction_change_algorithm, p_state.target_distribution, p_state.dimension,"D_", p_state.target_acc_value, "target_acc_", p_state.jumps_before_refresh, "jbr_", p_state.nsamples, "samples_", p_state.nchains, "nchains")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,location_add,name,name_add,extension)
    save(full_name, Dict("testmeasurestruct" => testmeasures); compress = true)
end

function save_test_measures(p_state::MCMCPerformanceState, testmeasures, run_id=1) 
    location = "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/"
    sampler = "mcmc/"
    location_add = "test_measures/"
    name = string(p_state.direction_change_algorithm, p_state.target_distribution, p_state.dimension,"D_", p_state.target_acc_value, "target_acc_", p_state.jumps_before_refresh, "jbr_", p_state.nsamples, "samples_", p_state.nchains, "nchains")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,location_add,name,name_add,extension)
    save(full_name, Dict("testmeasurestruct" => testmeasures); compress = true)
end


function save_sample_plot(p_state::ECMCPerformanceState, sample_plot, run_id=1)
    location = "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/"
    sampler = "ecmc/"
    location_add = "sample_plots/"
    name = string(p_state.direction_change_algorithm, p_state.target_distribution, p_state.dimension,"D_", p_state.target_acc_value, "target_acc_", p_state.jumps_before_refresh, "jbr_", p_state.nsamples, "samples_", p_state.nchains, "nchains")
    name_add = string("_", run_id)
    extension = ".png"
    full_name = string(location,sampler,location_add,name,name_add,extension)
    savefig(sample_plot, full_name)
end


function multi_save_states(state_arr)
    for state in state_arr
        save_state(state)
    end
end


#--------------------loading functions------------------


#----------------------plotting functions-----------------------------------


function plot_samples(samples)
    p = plot(
        samples;
        vsel=collect(1:3),
        bins = 200,
        mean=true,
        std=false,
        globalmode=false,
        marginalmode=false,
        #diagonal = Dict(),
        #upper = Dict(),
        #lower = Dict(),
        vsel_label = [L"Θ_1", L"Θ_2", L"Θ_3"]
    )
    return p
end


