abstract type ECMCTuner end
abstract type ECMCStepSizeAdaptor end
abstract type ECMCTuningConvergenceCheck end


#------ ECMC Tuners ---------------------------------------------------------------------- 
export ECMCTuner

# for sampling without tuning
struct ECMCNoTuner <: ECMCTuner end
export ECMCNoTuner

@with_kw struct MFPSTuner{A<:ECMCStepSizeAdaptor, C<:ECMCTuningConvergenceCheck} <: ECMCTuner
    target_mfps::Int64 = 5
    max_n_steps::Int64 = 2*10^4
    adaption_scheme::A = GoogleAdaption() #NaiveAdaption() 
    tuning_convergence_check::C = AcceptanceRatioConvergence(target_acc = target_mfps / (target_mfps  + 1))
end
export MFPSTuner


#----- Tuning Convergence checks ---------------------------------------------------------
# For MFPS tuning: criterion for convergence of stepsize tuning

@with_kw struct AcceptanceRatioConvergence <: ECMCTuningConvergenceCheck
    target_acc::Float64 = 0.9 #TODO
    Npercent::Float64 = 0.3 # percentage of steps to account for in acceptance
    standard_deviation::Float64 = 0.001
    rel_dif_mean::Float64 = 0.01 
end



#----- Stepsize Adaptors -----------------------------------------------------------------
# For MFPS tuning: different schemes for stepsize (delta) adaption

struct NaiveAdaption <: ECMCStepSizeAdaptor end
export NaiveAdaption

function adapt_delta(adaption_scheme::NaiveAdaption, delta, ecmc_tuner_state, tuner::MFPSTuner)
    target_acc =  tuner.target_mfps / (tuner.target_mfps  + 1) #TODO: compute once and store in algorithm struct
    
    acc_array = ecmc_tuner_state.acc_C
    steps = ecmc_tuner_state.n_steps
    delta_arr = ecmc_tuner_state.delta_arr

    eval_steps = 240


    if steps > eval_steps
        current_acc = (acc_array[end]*steps - acc_array[end-eval_steps]*(steps-eval_steps))/eval_steps


        # acc_gradient blocks changes due to statistical fluctuations in the current_acc
        n_steps_calc =  2*eval_steps
        #N_percent = min( 50*eval_steps, Int(floor(0.2*steps)))
        #N_percent > n_steps_calc ? N=N_percent : N=n_steps_calc
        N = n_steps_calc
        if steps > (n_steps_calc) # put in new function to call with size of N as argument etc
            Nhalf = Int(floor(N/2))
            acc_gradient = abs((acc_array[end]*steps + acc_array[end-N]*(steps-N) - 2*acc_array[end-Nhalf]*(steps-Nhalf))/Nhalf)
        else
            acc_gradient = 1 # for start steps
        end
        

        # delta_gradient blocks mostly strong changes from a high delta to a low delta at the start of the tuning
        test_length = 4
        if steps > test_length
            delta_gradient = 0
            for i in 1:test_length
                delta_gradient += delta_arr[end-i+1]
            end
            delta_gradient = abs(delta_gradient/(test_length*delta))
        else
            delta_gradient = 0
        end
        
        #integral = ecmc_tuner_state.n_acc

        err = (target_acc - current_acc)
        #sign(err) < 0 ? err = err*target_acc/(1-target_acc) : err = err
        err = sign(err)*(abs(err))^1.2

        #trial and error part:
        err_factor = 1 # 1 is fine
        acc_grad_factor = 0.1 # acc_gradient is ca 0.02 at the end
        delta_grad_factor = 0.

        steps > 10^4 ? max_sup = 0.9 : max_sup = 0.4

        suppression = delta_grad_factor * delta_gradient + 1/(1 + (acc_grad_factor * acc_gradient + 10*err)^2)
        #suppression = 0
        #change based on error
        pid = err_factor*(1 - min(max_sup, suppression))*err
        #pid = err_factor*err

        #pid times delta to stay in same magnitude
        new_delta = max(1e-6, delta - delta*pid)


    else
        err = (target_acc - acc_array[end])
        new_delta = max(1e-6, delta - delta*err*0.2)
        #new_delta = delta
    end


    #new_delta = delta - delta*Δacc # * 1/sqrt(ecmc_state.n_lifts)
    return new_delta
end





struct ManonAdaption <: ECMCStepSizeAdaptor end
export ManonAdaption

function adapt_delta(adaption_scheme::ManonAdaption, delta, ecmc_state, tuner::MFPSTuner)
    target_acc = tuner.target_mfps / (tuner.target_mfps  + 1)
    
    #current_acc = m/(m+1)
    current_acc = ecmc_state.n_acc/ecmc_state.n_steps

    #TODO: Δacc = (target_acc - current_acc) ?
    Δacc = sign(target_acc - current_acc)

    new_delta = maximum([1e-4, (1-(10^-4* Δacc)/ecmc_state.n_steps) * delta - (Δacc/ecmc_state.n_steps) ])

    return new_delta
end



struct GoogleAdaption <: ECMCStepSizeAdaptor end
export GoogleAdaption

function adapt_delta(adaption_scheme::GoogleAdaption, delta, ecmc_tuner_state, tuner::MFPSTuner)
    target_acc =  tuner.target_mfps / (tuner.target_mfps  + 1) #TODO: compute once and store in algorithm struct
    
    #n_acc_arr = ecmc_tuner_state.n_acc_arr
    #steps = ecmc_tuner_state.n_steps

    γ = ecmc_tuner_state.γ
    α = 0.002


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
    new_delta = delta*exp(γ)

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




