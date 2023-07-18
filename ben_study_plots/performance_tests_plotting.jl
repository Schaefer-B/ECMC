
include("performance_tests.jl")

include("performance_tests_all_runs_listed.jl")



function min_max_mean_ess(effective_sample_size_arr)
    a = effective_sample_size_arr

    if typeof(a[1]) == typeof(1.)
        a = [a]
    end

    min_ess_arr = [minimum(a[i]) for i=eachindex(a)]

    min_ess = mean(min_ess_arr)
    std_min_ess = std(min_ess_arr)

    return min_ess, std_min_ess
end



function min_max_mean_ess(effective_sample_size_arr)
    a = effective_sample_size_arr

    if typeof(a[1]) == typeof(1.)
        a = [a]
    end

    max_ess_arr = [maximum(a[i]) for i=eachindex(a)]
    
    min_ess = mean(min_ess_arr)
    std_min_ess = std(min_ess_arr)

    return min_ess, std_min_ess
end

function min_max_mean_ess(effective_sample_size_arr)
    a = effective_sample_size_arr

    if typeof(a[1]) == typeof(1.)
        a = [a]
    end

    min_ess_arr = [minimum(a[i]) for i=eachindex(a)]
    
    min_ess = mean(min_ess_arr)
    std_min_ess = std(min_ess_arr)

    return min_ess, std_min_ess
end

#----------------------loading functions-----------------------------------


function load_state(p_state::ECMCPerformanceState, run_id=1)
    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.direction_change_algorithm, p_state.MFPS_value, "MFPS", p_state.jumps_before_refresh, "jbr")
    name_add = string("_", run_id)
    extension = ".jld2"
    full_name = string(location,sampler,name,name_add,extension)
    saved_state = load(full_name, "state")
    return saved_state
end



function load_test_measures(p_state::ECMCPerformanceState, run_ids=1:1)
    t_measures = []

    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "ecmc/"
    location_add = "test_measures/"
    name = string(p_state.direction_change_algorithm, p_state.target_distribution, p_state.dimension,"D_", p_state.target_acc_value, "target_acc_", p_state.jumps_before_refresh, "jbr_", p_state.nsamples, "samples_", p_state.nchains, "nchains")
    
    for run_id in run_ids
        name_add = string("_", run_id)
        extension = ".jld2"
        full_name = string(location,sampler,location_add,name,name_add,extension)
        t = load(full_name, "testmeasurestruct")
        push!(t_measures, t)
    end
    return t_measures
end










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





function plot_best_ess_dir_algo(p_states, test_measures_states, runs)
    
    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    p_states = [p_states[i] for i=eachindex(p_states)]
    direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
    direction_algos = unique(direction_algos_temp)
    #println("unique direction algos = ", direction_algos)

    best_ess_time_mean_arr = []
    best_ess_time_std_arr = []
    for dir_algo in direction_algos
        #println("dir_algo = ", dir_algo)
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        #println("one_dir_indices = ", one_dir_indices)
        best_ess_time_mean = 0
        best_ess_time_std = 0
        for t_acc_index in one_dir_indices
            #println("t_acc_index = ", t_acc_index)

            test_measures = test_measures_states[t_acc_index]
            
            ess_time_arr = [mean(test_measures[i].effective_sample_size)/test_measures[i].sample_time for i=eachindex(test_measures)]
            
            ess_time_mean = mean(ess_time_arr)
            ess_time_std = std(ess_time_arr)
            

            if ess_time_mean > best_ess_time_mean
                best_ess_time_mean = ess_time_mean
                best_ess_time_std = ess_time_std
            end

        end
        push!(best_ess_time_mean_arr, best_ess_time_mean)
        push!(best_ess_time_std_arr, best_ess_time_std)

    end

    dir_index = eachindex(direction_algos)
    x_values = direction_algos

    #final_plot = scatter(dir_index .-0.5, best_ess_time_mean_arr, xdiscrete_values=x_values, yerr=best_ess_time_std_arr)
    final_plot = plot(dir_index .-0.5, best_ess_time_mean_arr, xdiscrete_values=x_values, yerr=best_ess_time_std_arr)
    plot!(xlabel="Direction change algorithms", ylabel="Mean of ESS per sample time", legend=false)

    return final_plot
end



function plot_best_target_acc(p_states, test_measures_states, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    
    p_states = [p_states[i] for i=eachindex(p_states)]
    direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
    direction_algos = unique(direction_algos_temp)
    #println("unique direction algos = ", direction_algos)


    for dir_algo in direction_algos
        
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        

        for t_acc_index in one_dir_indices # for loop over target acceptance rates
            

            test_measures = test_measures_states[t_acc_index]
            
            ess_time_arr = [mean(test_measures[i].effective_sample_size)/test_measures[i].sample_time for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_arr)
            ess_time_std = std(ess_time_arr)

            

        end
        

    end

    dir_index = eachindex(direction_algos)
    x_values = direction_algos

    #final_plot = scatter(dir_index .-0.5, best_ess_time_mean_arr, xdiscrete_values=x_values, yerr=best_ess_time_std_arr)
    final_plot = plot(dir_index .-0.5, best_ess_time_mean_arr, xdiscrete_values=x_values, yerr=best_ess_time_std_arr)
    plot!(xlabel="Direction change algorithms", ylabel="Mean of ESS per sample time", legend=false)

    return final_plot
end



function plot_jbr_tests(state_arr, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    direction_algos_temp = [string(state_arr[i].direction_change_algorithm) for i=eachindex(state_arr)]
    direction_algos = unique(direction_algos_temp)
    

    for dir_algo in direction_algos

        dir_plot = plot(layout=(3,1))

        direction_states = state_arr[findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, state_arr)]
        dimensions = unique([direction_states[i].dimension for i=eachindex(direction_states)])

        
        dir_plot = plot!(subplot=1, title=dir_algo, xlabel="Jumps before refresh", ylabel="Minimum of ESS")
        dir_plot = plot!(subplot=2, xlabel="Jumps before refresh", ylabel="Maximum of ESS")
        dir_plot = plot!(subplot=3, xlabel="Jumps before refresh", ylabel="Mean of ESS")

        for dimension in dimensions
            states = direction_states[findall(x -> x.dimension == dimension ? true : false, direction_states)]
            
            x_values = []
            min_ess_arr = []
            max_ess_arr = []
            mean_ess_arr = []
            
            for state in states
                x = state.jumps_before_refresh
                state_ess = load_effective_sample_sizes(state, runs)
                min_ess, max_ess, mean_ess = min_max_mean_ess(state_ess)
                push!(min_ess_arr, min_ess)
                push!(max_ess_arr, max_ess)
                push!(mean_ess_arr, mean_ess)
                push!(x_values, x)
            end

            println(string(dir_algo, " ",dimension,"D MvNormal"))
            println(string("   Maximal ESS value for min(ESS): ", maximum(min_ess_arr)))
            println(string("   Maximal ESS value for max(ESS): ", maximum(max_ess_arr)))
            println(string("   Maximal ESS value for mean(ESS): ", maximum(mean_ess_arr)))


            min_ess_arr = min_ess_arr ./(maximum(min_ess_arr))
            max_ess_arr = max_ess_arr ./(maximum(max_ess_arr))
            mean_ess_arr = mean_ess_arr ./(maximum(mean_ess_arr))
            

            dir_plot = plot!(x_values, min_ess_arr, subplot=1, label=string(dimension, "D", " Multivariate Normal"), lw=2)
            dir_plot = plot!(x_values, max_ess_arr, subplot=2, label=string(dimension, "D", " Multivariate Normal"), lw=2)
            dir_plot = plot!(x_values, mean_ess_arr, subplot=3, label=string(dimension, "D", " Multivariate Normal"), lw=2)

        end
        dir_plot = plot!(xaxis=:log)
        location = "plots/"
        name = string(dir_algo, "jbr_plot")
        full_string = string(location,name)
        png(full_string)
    end

end










#--------loading stuff-----------

ecmc_p_states_run_001, runs_001 = run_001()
testmeasures_001 = [load_test_measures(ecmc_p_states_run_001[i], 1:runs_001) for i=eachindex(ecmc_p_states_run_001)]



eps = ecmc_p_states_run_001[14]
t_m = load_test_measures(eps, 1:runs_001);
#--------running stuff------------

p = plot_best_ess_dir_algo(ecmc_p_states_run_001, testmeasures_001, runs_001)

png("Finding_best_dir_algo_2")







#----------testing stuff--------------

t_m[1]




t_m[1].ks_p_values


plot_mfps_tests(ecmc_p_states, runs)

plot_jbr_tests(ecmc_p_states, runs)


std(t_m[6].normalized_residuals)
plot(t_m[6].ks_p_values, st=:histogram, bins=5)
plot(t_m[1].chisq_values, st=:histogram, bins=10)
plot(t_m[6].normalized_residuals, st=:histogram, bins=100, normalized=:pdf)
plot!(rand(Normal(0.,1), 10000), st=:histogram, bins=100, normalized=:pdf)
png("p+norm")


for i in eachindex(t_m)
    println("$i diff to mean = ", mean(abs.(t_m[i].samples_mean)), " +/- ", std(abs.(t_m[i].samples_mean)))
end
t_m[1].samples_std

t_m = testmeasures_001[20];
for i in eachindex(t_m)
    println("ess/second($i) = ", mean(t_m[i].effective_sample_size)/t_m[i].sample_time)
end


# calculating std through different means:
# first:
# propagation of uncertainty
ess_time_arr = [mean(t_m[i].effective_sample_size)/t_m[i].sample_time for i=eachindex(t_m)]
# standard deviation f = A/B while B has no std: sigma(f) = sigma(A)/B
ess_time_std_arr = [std(t_m[i].effective_sample_size)/t_m[i].sample_time for i=eachindex(t_m)]
# now the stuff:
ess_time_mean_1 = mean(ess_time_arr)
# standard deviation through prop. of uncertainty says if f = sum(array)/length(array) then std(f) = sqrt( sum(std_i^2) / length(array)^2 )  
# with std_i being the standard deviation of the i-th element of the array 
# this is also the formula for the standard error (standard deviation of the mean) if all std_i were equal
ess_time_std_1 = sqrt(sum(x -> x^2, ess_time_std_arr)/length(ess_time_std_arr)^2)

# second:
# weighted mean and weighted std (look at wikipedia: inverse variance weighting)
ess_time_arr = [mean(t_m[i].effective_sample_size)/t_m[i].sample_time for i=eachindex(t_m)]
# standard deviation f = A/B while B has no std: sigma(f) = sigma(A)/B
ess_time_std_arr = [std(t_m[i].effective_sample_size)/t_m[i].sample_time for i=eachindex(t_m)]
# now the stuff:
weights = [1/ess_time_std_arr[i]^2 for i=eachindex(ess_time_std_arr)]
ess_time_mean_2 = sum(weights .* ess_time_arr)/sum(weights)
ess_time_std_2 = sqrt(1/sum(weights))

# comparison:
mean(ess_time_arr)
std(ess_time_arr)
wrong_ste = std(ess_time_arr)/sqrt(length(ess_time_arr))

