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
    max_n_steps::Int64 = 100_000
    adaption_scheme::A = NaiveAdaption() 
    tuning_convergence_check::C = AcceptanceRatioConvergence(target_acc = target_mfps / (target_mfps  + 1))
end
export MFPSTuner


#----- Tuning Convergence checks ---------------------------------------------------------
# For MFPS tuning: criterion for convergence of stepsize tuning

@with_kw struct AcceptanceRatioConvergence <: ECMCTuningConvergenceCheck
    target_acc::Float64 = 0.9 #TODO
    Npercent::Float64 = 0.3 # percentage of steps to account for in acceptance
    std::Float64 = 0.001
    rel_dif_mean::Float64 = 0.01 
end



#----- Stepsize Adaptors -----------------------------------------------------------------
# For MFPS tuning: different schemes for stepsize (delta) adaption

struct NaiveAdaption <: ECMCStepSizeAdaptor end
export NaiveAdaption

function adapt_delta(adaption_scheme::NaiveAdaption, delta, ecmc_state, tuner::MFPSTuner)
    target_acc =  tuner.target_mfps / (tuner.target_mfps  + 1) #TODO: compute once and store in algorithm struct
    #current_acc = m/(m+1)
    current_acc = ecmc_state.n_acc/ecmc_state.n_steps

    Δacc = (target_acc - current_acc)
    #new_delta = delta * (1-Δacc) 
    new_delta = delta - delta*Δacc # * 1/sqrt(ecmc_state.n_lifts)
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

# function adapt_delta(adaption_scheme::GoogleAdaption, delta, ecmc_state, tuner::MFPSTuner)
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





#----- Check Tuning Convergence ----------------------------------------------------------
function check_tuning_convergence!(
    ecmc_tuner_state::ECMCTunerState,
    tuning_convergence_check::AcceptanceRatioConvergence,
)
    target_acc = tuning_convergence_check.target_acc
    acc_C = ecmc_tuner_state.acc_C # TODO: make acc_C & delta_arr fixed-size (N) arrays that are updated in rolling fahsion
    
    # NOTE: N is increasing with n_steps. Mean of growing array gets slow.
    N = minimum([length(acc_C), Int(floor(tuning_convergence_check.Npercent * ecmc_tuner_state.n_steps))]) #user input
    mean_acc_C = mean(acc_C[end-N+1:end])  

    mean_has_converged = abs(mean_acc_C - target_acc) < tuning_convergence_check.rel_dif_mean * target_acc 
    std_is_low_enough = std(acc_C[end-N+1:end]) < tuning_convergence_check.std

    mean_delta = mean(ecmc_tuner_state.delta_arr[end-N+1:end])
    #tuned_algorithm = reconstruct(algorithm, step_amplitude=mean_delta)

    ecmc_tuner_state.tuned_delta = mean_delta

    if mean_has_converged & std_is_low_enough
        #@show mean_delta
        #@show abs(mean_acc_C - target_acc) / target_acc
        #@show std(acc_C[end-N+1:end])

        return true#, tuned_algorithm
    end 

    return false#, tuned_algorithm
end