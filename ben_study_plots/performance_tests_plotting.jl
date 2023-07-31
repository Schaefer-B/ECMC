
include("performance_tests.jl")

include("performance_tests_all_runs_listed.jl")




function state_fusion(multi_states_arr)
    fused_states = []
    for states_arr in multi_states_arr
        for i in eachindex(states_arr)
            push!(fused_states, states_arr[i])
        end
    end
    return fused_states
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



function load_test_measures(p_state::MCMCPerformanceState, run_ids=1:1)
    t_measures = []

    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "mcmc/"
    location_add = "test_measures/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.nsamples, "samples_", p_state.nchains, "nchains", "_mcmc")
    
    for run_id in run_ids
        name_add = string("_", run_id)
        extension = ".jld2"
        full_name = string(location,sampler,location_add,name,name_add,extension)
        t = load(full_name, "testmeasurestruct")
        push!(t_measures, t)
    end
    return t_measures
end


function load_test_measures(p_state::HMCPerformanceState, run_ids=1:1)
    t_measures = []

    location = "ben_study_plots/saved_performance_test_result_states/"
    sampler = "hmc/"
    location_add = "test_measures/"
    name = string(p_state.target_distribution, p_state.dimension,"D_", p_state.nsamples, "samples_", p_state.nchains, "nchains", "_hmc")
    
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

    final_plots = []

    for dir_algo in direction_algos
        
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        
        ess_time_mean_arr = []
        ess_time_std_arr = []
        
        for t_acc_index in one_dir_indices # for loop over target acceptance rates
            

            test_measures = test_measures_states[t_acc_index]
            
            ess_time_runs = [mean(test_measures[i].effective_sample_size)/test_measures[i].sample_time for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_runs)
            ess_time_std = std(ess_time_runs)

            push!(ess_time_mean_arr, ess_time_mean)
            push!(ess_time_std_arr, ess_time_std)

        end
        x_values = [p_states[i].target_acc_value for i=one_dir_indices]
        final_plot = plot(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr))
        plot!(xlabel="Target acceptance rate", ylabel="Mean of ESS per sample time", legend=false)
        
        max_ess_time = maximum(ess_time_mean_arr)
        max_ess_time_std = ess_time_std_arr[findfirst(x -> x == max_ess_time, ess_time_mean_arr)]
        target_acc_rate = x_values[findfirst(x -> x == max_ess_time, ess_time_mean_arr)]

        location = "plots/"
        name = string("finding_best_target_acc_", dir_algo)
        full = string(location, name)
        png(full)
        println(dir_algo)
        println("    best target acceptance rate = ", target_acc_rate)
        println("    best ESS/second = ", max_ess_time, " +/- ", max_ess_time_std)
        println()

        push!(final_plots, final_plot)
    end

    
    return final_plots
end



function plot_best_jbr(p_states, test_measures_states, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)

    
    p_states = [p_states[i] for i=eachindex(p_states)]
    direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
    direction_algos = unique(direction_algos_temp)
    #println("unique direction algos = ", direction_algos)

    final_plots = []

    for dir_algo in direction_algos
        
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        
        ess_time_mean_arr = []
        ess_time_std_arr = []
        
        for jbr_index in one_dir_indices # for loop over jbr
            

            test_measures = test_measures_states[jbr_index]
            
            ess_time_runs = [mean(test_measures[i].effective_sample_size)/test_measures[i].sample_time for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_runs)
            ess_time_std = std(ess_time_runs)

            push!(ess_time_mean_arr, ess_time_mean)
            push!(ess_time_std_arr, ess_time_std)

        end
        x_values = [p_states[i].jumps_before_refresh for i=one_dir_indices]
        final_plot = plot(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr))
        plot!(xlabel="Jumps before refresh", ylabel="Mean of ESS per sample time", legend=false)
        
        max_ess_time = maximum(ess_time_mean_arr)
        max_ess_time_std = ess_time_std_arr[findfirst(x -> x == max_ess_time, ess_time_mean_arr)]
        jbr = x_values[findfirst(x -> x == max_ess_time, ess_time_mean_arr)]

        location = "plots/"
        name = string("finding_best_jbr_", dir_algo)
        full = string(location, name)
        png(full)
        println(dir_algo)
        println("    best jumps before refresh = ", jbr)
        println("    best ESS/second = ", max_ess_time, " +/- ", max_ess_time_std)
        println()

        push!(final_plots, final_plot)
    end

    
    return final_plots
end




function plot_ess_time(p_states, mcmc_states, hmc_states, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    location = "plots/"

    ess_time_plot = plot(xlabel="Dimension", ylabel="ESS/s")
    

    if p_states != []
        p_states = [p_states[i] for i=eachindex(p_states)]
        ecmc_t_measures = [load_test_measures(p_states[i], 1:runs) for i=eachindex(p_states)]

        direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
        direction_algos = unique(direction_algos_temp)

        for dir_algo in direction_algos
            one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)

            ess_time_mean_arr = []
            ess_time_std_arr = []
            
            for dim_index in one_dir_indices # for loop over dimensions
                
                test_measures = ecmc_t_measures[dim_index]
                
                ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time - test_measures[i].tuning_time) for i=eachindex(test_measures)]
                ess_time_mean = mean(ess_time_runs)
                ess_time_std = std(ess_time_runs)

                push!(ess_time_mean_arr, ess_time_mean)
                push!(ess_time_std_arr, ess_time_std)



            end

            x_values = [p_states[i].dimension for i=one_dir_indices]

            ess_time_plot = plot!(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr), label=dir_algo)
        end
    end


    if mcmc_states != []
        mcmc_states = [mcmc_states[i] for i=eachindex(mcmc_states)]
        mcmc_t_measures = [load_test_measures(mcmc_states[i], 1:runs) for i=eachindex(mcmc_states)]

        ess_time_mean_arr = []
        ess_time_std_arr = []
        
        for dim_index in eachindex(mcmc_states)

            test_measures = mcmc_t_measures[dim_index]
            
            ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time - test_measures[i].tuning_time) for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_runs)
            ess_time_std = std(ess_time_runs)

            push!(ess_time_mean_arr, ess_time_mean)
            push!(ess_time_std_arr, ess_time_std)

        end

        x_values = [mcmc_states[i].dimension for i=eachindex(mcmc_states)]

        ess_time_plot = plot!(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr), label="Metropolis-Hastings")
    end


    if hmc_states != []
        hmc_states = [hmc_states[i] for i=eachindex(hmc_states)]
        hmc_t_measures = [load_test_measures(hmc_states[i], 1:runs) for i=eachindex(hmc_states)]

        ess_time_mean_arr = []
        ess_time_std_arr = []
        
        for dim_index in eachindex(hmc_states)

            test_measures = hmc_t_measures[dim_index]
            
            ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time - test_measures[i].tuning_time) for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_runs)
            ess_time_std = std(ess_time_runs)

            push!(ess_time_mean_arr, ess_time_mean)
            push!(ess_time_std_arr, ess_time_std)

        end

        x_values = [hmc_states[i].dimension for i=eachindex(hmc_states)]

        ess_time_plot = plot!(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr), label="Hamilton MC")
    end


    plot!(yaxis=:log)
    name = string("ess_time")
    extension = string(".png")
    full = string(location, name, extension)
    savefig(ess_time_plot, full)
    
    return ess_time_plot
end


function plot_mini_ess_time(p_states, mcmc_states, hmc_states, runs, min_dim, max_dim, tuner_time_subtracted)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    location = "plots/"

    ess_time_plot = plot(xlabel="Dimension", ylabel="ESS/s")
    

    if p_states != []
        p_states = [p_states[i] for i=eachindex(p_states)]
        ecmc_t_measures = [load_test_measures(p_states[i], 1:runs) for i=eachindex(p_states)]

        direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
        direction_algos = unique(direction_algos_temp)

        for dir_algo in direction_algos
            one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)

            ess_time_mean_arr = []
            ess_time_std_arr = []
            
            for dim_index in one_dir_indices # for loop over dimensions
                
                if p_states[dim_index].dimension >= min_dim && p_states[dim_index].dimension <= max_dim

                    test_measures = ecmc_t_measures[dim_index]

                    if tuner_time_subtracted == true
                        ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time - test_measures[i].tuning_time) for i=eachindex(test_measures)]
                    else
                        ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time) for i=eachindex(test_measures)]
                    end
                    ess_time_mean = mean(ess_time_runs)
                    ess_time_std = std(ess_time_runs)

                    push!(ess_time_mean_arr, ess_time_mean)
                    push!(ess_time_std_arr, ess_time_std)

                end

            end

            x_values = []
            for i in one_dir_indices
                if p_states[i].dimension >= min_dim && p_states[i].dimension <= max_dim
                push!(x_values, p_states[i].dimension)
                end
            end

            ess_time_plot = plot!(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr), label=dir_algo)
        end
    end


    if mcmc_states != []
        mcmc_states = [mcmc_states[i] for i=eachindex(mcmc_states)]
        mcmc_t_measures = [load_test_measures(mcmc_states[i], 1:runs) for i=eachindex(mcmc_states)]

        ess_time_mean_arr = []
        ess_time_std_arr = []
        
        for dim_index in eachindex(mcmc_states)

            if p_states[dim_index].dimension >= min_dim && p_states[dim_index].dimension <= max_dim
            
                test_measures = mcmc_t_measures[dim_index]

                if tuner_time_subtracted == true
                    ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time - test_measures[i].tuning_time) for i=eachindex(test_measures)]
                else
                    ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time) for i=eachindex(test_measures)]
                end
                ess_time_mean = mean(ess_time_runs)
                ess_time_std = std(ess_time_runs)

                push!(ess_time_mean_arr, ess_time_mean)
                push!(ess_time_std_arr, ess_time_std)
            end

        end

        x_values = []
        for i in eachindex(mcmc_states)
            if mcmc_states[i].dimension >= min_dim && mcmc_states[i].dimension <= max_dim
            push!(x_values, mcmc_states[i].dimension)
            end
        end

        ess_time_plot = plot!(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr), label="Metropolis-Hastings")
    end


    if hmc_states != []
        hmc_states = [hmc_states[i] for i=eachindex(hmc_states)]
        hmc_t_measures = [load_test_measures(hmc_states[i], 1:runs) for i=eachindex(hmc_states)]

        ess_time_mean_arr = []
        ess_time_std_arr = []
        
        for dim_index in eachindex(hmc_states)

            if p_states[dim_index].dimension >= min_dim && p_states[dim_index].dimension <= max_dim

                test_measures = hmc_t_measures[dim_index]
                
                if tuner_time_subtracted == true
                ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time - test_measures[i].tuning_time) for i=eachindex(test_measures)]
                else
                ess_time_runs = [mean(test_measures[i].effective_sample_size)/(test_measures[i].sample_time) for i=eachindex(test_measures)]
                end
                ess_time_mean = mean(ess_time_runs)
                ess_time_std = std(ess_time_runs)

                push!(ess_time_mean_arr, ess_time_mean)
                push!(ess_time_std_arr, ess_time_std)
            end

        end

        x_values = []
        for i in eachindex(hmc_states)
            if hmc_states[i].dimension >= min_dim && hmc_states[i].dimension <= max_dim
            push!(x_values, hmc_states[i].dimension)
            end
        end

        ess_time_plot = plot!(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr), label="Hamilton MC")
    end


    if tuner_time_subtracted == true
        name = string("mini_ess_time_tunertimesubstracted")
    else
        name = string("mini_ess_time_fullsampletime")
    end
    extension = string(".png")
    full = string(location, name, extension)
    #savefig(ess_time_plot, full)
    
    return ess_time_plot
end



function plot_ks_p(p_states, mcmc_states, hmc_states, runs, max_points)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    #location = "plots/"

    spots_maximum = max_points

    ksp_plots = []
    

    if p_states != []
        p_states = [p_states[i] for i=eachindex(p_states)]
        ecmc_t_measures = [load_test_measures(p_states[i], 1:runs) for i=eachindex(p_states)]

        direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
        direction_algos = unique(direction_algos_temp)

        for dir_algo in direction_algos
            one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)

            ksp_per_dim = []
            for dim_index in one_dir_indices # for loop over dimensions
                
                test_measures = ecmc_t_measures[dim_index]

                ksp_one_dim = []
                for tm in eachindex(test_measures)
                    for kp in eachindex(test_measures[tm].ks_p_values)
                        ksp = test_measures[tm].ks_p_values[kp]
                        push!(ksp_one_dim, ksp)
                    end
                end

                push!(ksp_per_dim, ksp_one_dim)

            end

            x_values = [p_states[i].dimension for i=one_dir_indices]

            ksp_plot = plot(xlabel="Dimension", ylabel="Kolmogorov-Smirnov test p-value", legend=false)
            for x in eachindex(x_values)
                r = min(spots_maximum, length(ksp_per_dim[x]))
                x_vals = fill(x_values[x], r)
                ksp_plot = scatter!(x_vals, ksp_per_dim[x], color=:blue)
            end
            push!(ksp_plots, ksp_plot)
        end
    end


    if mcmc_states != []
        mcmc_states = [mcmc_states[i] for i=eachindex(mcmc_states)]
        mcmc_t_measures = [load_test_measures(mcmc_states[i], 1:runs) for i=eachindex(mcmc_states)]

        ksp_per_dim = []
        for dim_index in eachindex(mcmc_states)

            test_measures = mcmc_t_measures[dim_index]


            ksp_one_dim = []
            for tm in eachindex(test_measures)
                for kp in eachindex(test_measures[tm].ks_p_values)
                    ksp = test_measures[tm].ks_p_values[kp]
                    push!(ksp_one_dim, ksp)
                end
            end

            push!(ksp_per_dim, ksp_one_dim)

        end

        x_values = [mcmc_states[i].dimension for i=eachindex(mcmc_states)]

        ksp_plot = plot(xlabel="Dimension", ylabel="Kolmogorov-Smirnov test p-value", legend=false)
        for x in eachindex(x_values)
            r = min(spots_maximum, length(ksp_per_dim[x]))
            x_vals = fill(x_values[x], r)
            ksp_plot = scatter!(x_vals, ksp_per_dim[x], color=:blue)
        end
        push!(ksp_plots, ksp_plot)
    end


    if hmc_states != []
        hmc_states = [hmc_states[i] for i=eachindex(hmc_states)]
        hmc_t_measures = [load_test_measures(hmc_states[i], 1:runs) for i=eachindex(hmc_states)]

        
        ksp_per_dim = []
        for dim_index in eachindex(hmc_states)

            test_measures = hmc_t_measures[dim_index]
            
            ksp_one_dim = []
            for tm in eachindex(test_measures)
                for kp in eachindex(test_measures[tm].ks_p_values)
                    ksp = test_measures[tm].ks_p_values[kp]
                    push!(ksp_one_dim, ksp)
                end
            end

            push!(ksp_per_dim, ksp_one_dim)

        end

        x_values = [hmc_states[i].dimension for i=eachindex(hmc_states)]

        ksp_plot = plot(xlabel="Dimension", ylabel="Kolmogorov-Smirnov test p-value", legend=false)
        for x in eachindex(x_values)
            r = min(spots_maximum, length(ksp_per_dim[x]))
            x_vals = fill(x_values[x], r)
            ksp_plot = scatter!(x_vals, ksp_per_dim[x], color=:blue)
        end
        push!(ksp_plots, ksp_plot)
    end


    
    return ksp_plots
end



function plot_chisq(p_states, test_measures_states, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    location = "plots/"
    
    p_states = [p_states[i] for i=eachindex(p_states)]
    direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
    direction_algos = unique(direction_algos_temp)
    #println("unique direction algos = ", direction_algos)

    chisq_plots = []

    for dir_algo in direction_algos
        
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        #p_algo_states = p_states(one_dir_indices)

    
        chisq_mean_arr = []
        chisq_std_arr = []
        for dim_index in one_dir_indices # for loop over dimensions
            

            test_measures = test_measures_states[dim_index]
            
            chisq_values = []
            for i in eachindex(test_measures)
                chisq_values = vcat(chisq_values, test_measures[i].chisq_values)
            end
            chisq_mean = mean(chisq_values)
            chisq_std = std(chisq_values)
            push!(chisq_mean_arr, chisq_mean)
            push!(chisq_std_arr, chisq_std)

        end

        x_values = [p_states[i].dimension for i=one_dir_indices]

        chisq_plot = plot(x_values, chisq_mean_arr, ribbon=(chisq_std_arr, chisq_std_arr))
        plot!(xlabel="Dimension", ylabel="Mean of squared Distances", legend=false) # CHANGE NAME OF YLABEL
        name = string("chisq_", dir_algo)
        full = string(location, name)
        png(full)
    


        push!(chisq_plots, chisq_plot)
    end

    
    return chisq_plots
end




function plot_pulls(p_states, test_measures_states, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    location = "plots/"
    
    p_states = [p_states[i] for i=eachindex(p_states)]
    direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
    direction_algos = unique(direction_algos_temp)
    #println("unique direction algos = ", direction_algos)

    pull_plots = []

    for dir_algo in direction_algos
        
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        #p_algo_states = p_states(one_dir_indices)


        pulls_mean_arr = []
        pulls_std_arr = []
        
        for dim_index in one_dir_indices # for loop over dimensions
            

            test_measures = test_measures_states[dim_index]
            
            ess_time_runs = [mean(test_measures[i].effective_sample_size)/test_measures[i].sample_time for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_runs)
            ess_time_std = std(ess_time_runs)
            push!(ess_time_mean_arr, ess_time_mean)
            push!(ess_time_std_arr, ess_time_std)

            pulls = []
            for i in eachindex(test_measures)
                pulls = vcat(pulls, test_measures[i].normalized_residuals)
            end
            pulls_mean = mean(pulls)
            pulls_std = std(pulls)
            push!(pulls_mean_arr, pulls_mean)
            push!(pulls_std_arr, pulls_std)



        end

        x_values = [p_states[i].dimension for i=one_dir_indices]

        pull_plot = plot(x_values, pulls_mean_arr, ribbon=(pulls_std_arr,pulls_std_arr))
        #pull_plot = plot(x_values, pulls_mean_arr, ribbon=(pulls_std_arr,pulls_std_arr))
        plot!(xlabel="Dimension", ylabel="Mean of pulls", legend=false)
        name = string("pulls_", dir_algo)
        full = string(location, name)
        png(full)
        
        
        


        push!(pull_plots, pull_plot)
    end

    
    return pull_plots
end



function plot_diffs(p_states, test_measures_states, runs)

    gr(size=(1.3*850, 1.3*800), thickness_scaling = 1.5)
    location = "plots/"
    
    p_states = [p_states[i] for i=eachindex(p_states)]
    direction_algos_temp = [string(p_states[i].direction_change_algorithm) for i=eachindex(p_states)]
    direction_algos = unique(direction_algos_temp)
    #println("unique direction algos = ", direction_algos)

    mean_diff_plots = []
    std_diff_plots = []

    for dir_algo in direction_algos
        
        one_dir_indices = findall(x -> string(x.direction_change_algorithm) == dir_algo ? true : false, p_states)
        #p_algo_states = p_states(one_dir_indices)


        mean_diff_arr = []
        std_diff_arr = []
        
        for dim_index in one_dir_indices # for loop over dimensions
            

            test_measures = test_measures_states[dim_index]
            
            ess_time_runs = [mean(test_measures[i].effective_sample_size)/test_measures[i].sample_time for i=eachindex(test_measures)]
            ess_time_mean = mean(ess_time_runs)
            ess_time_std = std(ess_time_runs)
            push!(ess_time_mean_arr, ess_time_mean)
            push!(ess_time_std_arr, ess_time_std)

            ksp_values = []
            for i in eachindex(test_measures)
                ksp_values = vcat(ksp_values, test_measures[i].ks_p_values)
            end
            push!(ksp_arr, ksp_values)



        end

        x_values = [p_states[i].dimension for i=one_dir_indices]

        ess_time_plot = plot(x_values, ess_time_mean_arr, ribbon=(ess_time_std_arr,ess_time_std_arr))
        plot!(xlabel="Dimension", ylabel="Mean of ESS per sample time", legend=false)
        name = string("ess_time_", dir_algo)
        full = string(location, name)
        png(full)
        
        ksp_plot = plot(xlabel="Dimension", ylabel="Kolmogorov-Smirnov test p-value", legend=false)
        for i in eachindex(ksp_arr[1])
            y_values = [ksp_arr[dim][i] for dim=eachindex(ksp_arr)]
            ksp_plot = scatter!(x_values, y_values, color=:blue)
        end
        
        name = string("ksp_", dir_algo)
        full = string(location, name)
        png(full)
        
        

        



        push!(mean_diff_plots, mean_diff_plot)
        push!(std_diff_plots, std_diff_plot)
    end

    
    return mean_diff_plots, std_diff_plots
end


#--------loading stuff-----------

ecmc_p_states_run_001, runs_001 = run_001()
testmeasures_001 = [load_test_measures(ecmc_p_states_run_001[i], 1:runs_001) for i=eachindex(ecmc_p_states_run_001)]
best_dir_algo_plots = plot_best_ess_dir_algo(ecmc_p_states_run_001, testmeasures_001, runs_001)

ecmc_p_states_run_002, runs_002 = run_002()
testmeasures_002 = [load_test_measures(ecmc_p_states_run_002[i], 1:runs_002) for i=eachindex(ecmc_p_states_run_002)]
t_acc_plots = plot_best_target_acc(ecmc_p_states_run_002, testmeasures_002, runs_002)

ecmc_p_states_run_003, runs_003 = run_003()
testmeasures_003 = [load_test_measures(ecmc_p_states_run_003[i], 1:runs_003) for i=eachindex(ecmc_p_states_run_003)]
best_jbr_for_reflect_plot = plot_best_jbr(ecmc_p_states_run_003, testmeasures_003, runs_003)
best_jbr_for_reflect_plot[1]
best_jbr_for_reflect_plot[2]
for jbr in 1:16
    println("ess mean = ", mean([mean(testmeasures_003[jbr][i].effective_sample_size) for i=1:10]))
    println("ess std = ", std([mean(testmeasures_003[jbr][i].effective_sample_size) for i=1:10]))
    println("time mean = ", mean([testmeasures_003[jbr][i].sample_time for i=1:10]))
    println("time std = ", std([testmeasures_003[jbr][i].sample_time for i=1:10]))
    println()
end # Die mean ESS sind fast immer gleich und die schwankungen auch nicht so groß, aber die sample time ist erheblich unterschiedlich und schwankt zwischen den runs (für das selbe jbr) sehr. und da die sample time im nenner steht macht das einen erheblichen unterschied aus




#---PERFORMANCE TEST PLOTS------
reflect_performance_states, runs = ecmc_reflect_performance()
stochasticreflect_performance_states, runs = ecmc_stochasticreflect_performance()
refresh_performance_states, runs = ecmc_refresh_performance()
all_ecmc_states = state_fusion([reflect_performance_states, stochasticreflect_performance_states, refresh_performance_states])
mcmc_performance_states, runs = mcmc_performance()
hmc_performance_states, runs = hmc_performance()

#ess_plot = plot_ess_time(both_p_states, mcmc_p_states_run_006, [], 10)

ess_plot = plot_ess_time(all_ecmc_states, mcmc_performance_states, hmc_performance_states, 10)
ess_plot


min_dim = 64
max_dim = 200
tuner_time_subtracted = true
mini_ess_plot = plot_mini_ess_time(all_ecmc_states, mcmc_performance_states, hmc_performance_states, 10, min_dim, max_dim, tuner_time_subtracted)
mini_ess_plot_fullsampletime = plot_mini_ess_time(all_ecmc_states, mcmc_performance_states, hmc_performance_states, 10, min_dim, max_dim, false)

#It seems like the ReflectDirection() algorithm is performing better than the HMC algorithm for high dimensions.
#But looking at the plot showing the p-values for the Kolmogorov-Smirnov plot, it is clearly shown, that the ReflectDirection() is performing worse.
#That means the ESS got overestimated.


max_points = 100
ks_plots = plot_ks_p(all_ecmc_states, mcmc_performance_states, hmc_performance_states, 10, max_points)
ks_plots[1]





refresh_test, runs = ecmc_refresh_targetacc()
t_m = [load_test_measures(refresh_test[i], 1:runs) for i=eachindex(refresh_test)]
refresh_target_plot = plot_best_target_acc(refresh_test, t_m, runs)
refresh_target_plot[1]
savefig(refresh_target_plot[1], "plots/best_target_acc_refresh_direction.png")








#----------plotting ks_p_values: -------------


testmeasures_004 = [load_test_measures(ecmc_p_states_run_004[i], 1:runs_004) for i=eachindex(ecmc_p_states_run_004)]
testmeasures_005 = [load_test_measures(ecmc_p_states_run_005[i], 1:runs_005) for i=eachindex(ecmc_p_states_run_005)]
dimension = 7
kssk1 = []
kssk2 = []
for k = 1:10
    for i in 1:2^(dimension-1)
        k1 = testmeasures_004[dimension][k].ks_p_values[i]
        push!(kssk1, k1)
        k2 = testmeasures_005[dimension][k].ks_p_values[i]
        push!(kssk2, k2)
    end
end
kssk1
plot(kssk1, st=:hist, bins=0:0.02:1)
plot(kssk2, st=:hist, bins=0:0.02:1)

#----------testing stuff--------------


#-----------------------------
eps = ecmc_p_states_run_001[14]
t_m = load_test_measures(eps, 1:runs_001);
#--------running stuff------------

p = plot_best_ess_dir_algo(ecmc_p_states_run_001, testmeasures_001, runs_001)

png("Finding_best_dir_algo_2")

t_acc_plots = plot_best_target_acc(ecmc_p_states_run_002, testmeasures_002, runs_002)

t_acc_plots[4]


t_m[1]

ks = []
for i in eachindex(t_m)
    ks = vcat(ks, t_m[i].ks_p_values)
end
ks
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
t_m[1]

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

