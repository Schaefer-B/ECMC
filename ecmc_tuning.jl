abstract type ECMCTuner end
abstract type ECMCStepSizeAdaptor end
abstract type ECMCTuningConvergenceCheck end


#------ ECMC Tuners ---------------------------------------------------------------------- 
export ECMCTuner

# for sampling without tuning
struct ECMCNoTuner <: ECMCTuner end
export ECMCNoTuner

@with_kw struct MFPSTuner{A<:ECMCStepSizeAdaptor, C<:ECMCTuningConvergenceCheck} <: ECMCTuner
    target_mfps::Float64 = 3.
    target_acc::Float64 = target_mfps / (target_mfps  + 1)
    max_n_steps::Int64 = 3*10^4
    adaption_scheme::A = NaiveAdaption() #NaiveAdaption() 
    #tuning_convergence_check::C = (target_mfps == 0. ? AcceptanceRatioConvergence(target_acc = target_acc) : AcceptanceRatioConvergence(target_acc = target_mfps / (target_mfps  + 1)))
    tuning_convergence_check::C = AcceptanceRatioConvergence(target_acc = target_acc)
    starting_alpha::Float64 = 0.1
end
export MFPSTuner


#----- Tuning Convergence checks ---------------------------------------------------------
# For MFPS tuning: criterion for convergence of stepsize tuning

@with_kw struct AcceptanceRatioConvergence <: ECMCTuningConvergenceCheck
    target_acc::Float64 = 0.9 #TODO
    Npercent::Float64 = 0.3 # percentage of steps to account for in acceptance
    standard_deviation::Float64 = 0.003
    abs_dif_mean::Float64 = 0.002
end



#----- Stepsize Adaptors -----------------------------------------------------------------
# For MFPS tuning: different schemes for stepsize (delta) adaption

struct NaiveAdaption <: ECMCStepSizeAdaptor end
export NaiveAdaption

function adapt_delta(adaption_scheme::NaiveAdaption, delta, ecmc_tuner_state, tuner::MFPSTuner)
    target_acc =  tuner.tuning_convergence_check.target_acc #TODO: compute once and store in algorithm struct

    acc_array = ecmc_tuner_state.acc_C
    steps = ecmc_tuner_state.n_steps
    delta_arr = ecmc_tuner_state.delta_arr


    params = ecmc_tuner_state.params


    eval_steps = Int(floor(params[8]))

    #if steps%eval_steps == 0
        if steps > eval_steps
            current_acc = (acc_array[end]- acc_array[end-eval_steps])/eval_steps


            # acc_gradient blocks changes due to statistical fluctuations in the current_acc
            n_steps_calc =  2*eval_steps
            N = n_steps_calc
            if steps > (n_steps_calc)
                Nhalf = Int(floor(N/2))
                acc_gradient = abs((acc_array[end] + acc_array[end-N] - 2*acc_array[end-Nhalf])/Nhalf)
            else
                acc_gradient = 1 # gradient for start steps
            end
            

            # delta_gradient blocks mostly strong changes in delta to help minimizing overshooting
            test_length = 10
            if steps > test_length
                delta_gradient = 0
                for i in 1:test_length
                    delta_gradient += sign(delta_arr[end-i+1])
                end
                delta_gradient = abs(delta_gradient/(test_length))
            else
                delta_gradient = 0
            end
            

            err = (target_acc - current_acc)
            err = sign(err)*(abs(err))^(1+params[4])

            #trial and error part:
            err_factor = params[1] # 0.5
            acc_grad_factor = params[2] # acc_gradient is ca 0.02 at the end, so something like 0.3 is nice
            delta_grad_factor = params[3] # 0.09 # maximum delta gradient is 1

            steps > 10^4 ? max_sup = 0.9 : max_sup = 0.9

            suppression = delta_grad_factor * delta_gradient + 1/(max(0.001, params[6]) + (acc_grad_factor * acc_gradient + params[5]*abs(err))^(1+params[7]))
            #suppression = 0

            #change based on error
            pid = err_factor*(1 - min(max_sup, suppression))*err


            #pid times delta to stay in same magnitude
            new_delta = max(1e-6, delta - delta*pid)


        else
            err = (target_acc - acc_array[end]/steps)
            new_delta = max(1e-6, delta - delta*err*0.25)
            #new_delta = delta
        end
    #else
    #    new_delta = delta
    #end
        #steps > eval_steps ? new_delta = max(1e-6, delta - delta*(target_acc - (acc_array[end]- acc_array[end-eval_steps])/eval_steps)) : new_delta = max(1e-6, delta - delta*(target_acc - acc_array[end]/steps))#truly naive tuning

    return new_delta
end





struct ManonAdaption <: ECMCStepSizeAdaptor end
export ManonAdaption

function adapt_delta(adaption_scheme::ManonAdaption, delta, ecmc_state, tuner::MFPSTuner)
    target_acc = tuner.tuning_convergence_check.target_acc
    
    #current_acc = m/(m+1)
    current_acc = ecmc_state.n_acc/ecmc_state.n_steps

    #TODO: Δacc = (target_acc - current_acc) ?
    Δacc = sign(target_acc - current_acc)

    new_delta = maximum([1e-4, (1-(10^-4* Δacc)/ecmc_state.n_steps) * delta - (Δacc/ecmc_state.n_steps) ])

    return new_delta
end



@with_kw struct GoogleAdaption <: ECMCStepSizeAdaptor
    automatic_adjusting::Bool = true
end
export GoogleAdaption

function adapt_delta(adaption_scheme::GoogleAdaption, delta, ecmc_tuner_state, tuner::MFPSTuner)
    target_acc =  tuner.tuning_convergence_check.target_acc #TODO: compute once and store in algorithm struct
    
    #n_acc_arr = ecmc_tuner_state.n_acc_arr
    #steps = ecmc_tuner_state.n_steps

    γ = ecmc_tuner_state.γ
    α = ecmc_tuner_state.α

    if adaption_scheme.automatic_adjusting == true
        min_α = 0.001
        max_α = 1
        change = 0.99
        eval_steps = 240

        acc_array = ecmc_tuner_state.acc_C
        #target_acc =  tuner.target_mfps / (tuner.target_mfps  + 1)
        #target_acc =  tuner.tuning_convergence_check.target_acc

        if ecmc_tuner_state.n_steps > eval_steps
            current_acc = (acc_array[end]- acc_array[end-eval_steps])/eval_steps
            err = abs(target_acc - current_acc)
            if γ == 0
                delta_p_std = std(ecmc_tuner_state.delta_arr[end-eval_steps:end])/mean(ecmc_tuner_state.delta_arr[end-eval_steps:end])
                if err < 0.01
                    α = max(min_α, α*change)
                    #ecmc_tuner_state.α_acc_err = err
                else
                    α = min(max_α, α/change)
                end
                
                

            #elseif err > 0.05 && abs(γ)<max_α
            #    α = abs(γ)
            end
            
            ecmc_tuner_state.α = α
        end
    end

    β = target_acc/(1-target_acc) * α

    if ecmc_tuner_state.step_acc == 1
        γ = γ + α
    else
        γ = γ - β
    end
    #if steps >= 2
    #    if length(ecmc_tuner_state.reject_step_arr) > 1
    #        new_accepts = n_acc_arr[end] - n_acc_arr[ecmc_tuner_state.reject_step_arr[end-1]]
    #    else
    #        new_accepts = 0
    #    end
        
    #    γ = γ - β + new_accepts*α

    #    new_delta = delta*exp(γ)
    #else
    #    new_delta = delta
    #end
    
    new_delta = max(delta*exp(γ),10^-8)
    return new_delta

end


#     # adapt delta according to paper
#     l = length(ecmc_state.mfps_arr)
#     m = m = mean(ecmc_state.mfps_arr)#l > 10 ? mean(ecmc_state.mfps_arr[end-10]) : mean(ecmc_state.mfps_arr)
#     γ = ecmc_state.γ
#     α = 0.1 * algorithm.step_amplitude #TODO

#     n = tuner.target_mfps / (tuner.target_mfps  + 1)
#     β = n /(1-n) * α
#     # @show α
#     # @show β
#     # @show γ

#     if m < tuner.target_mfps 
#         γ = γ - β
#     else
#         γ = γ + α
#     end 
#     #@show γ

#     new_delta = algorithm.step_amplitude * exp(γ) #TODO
#     ecmc_state.γ = γ
    
#     return new_delta
# end




