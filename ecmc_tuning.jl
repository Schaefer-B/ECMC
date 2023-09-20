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
    adaption_scheme::A = GoogleAdaption(automatic_adjusting=true)
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



@with_kw struct GoogleAdaption <: ECMCStepSizeAdaptor
    automatic_adjusting::Bool = true
end
export GoogleAdaption

function adapt_delta(adaption_scheme::GoogleAdaption, delta, ecmc_tuner_state, tuner::MFPSTuner)
    target_acc =  tuner.tuning_convergence_check.target_acc 
    
    γ = ecmc_tuner_state.γ
    α = ecmc_tuner_state.α

    if adaption_scheme.automatic_adjusting == true
        min_α = 0.001
        max_α = 1
        change = 0.99
        eval_steps = 240

        acc_array = ecmc_tuner_state.acc_C

        if ecmc_tuner_state.n_steps > eval_steps
            current_acc = (acc_array[end]- acc_array[end-eval_steps])/eval_steps
            err = abs(target_acc - current_acc)
            if γ == 0
                if err < 0.01
                    α = max(min_α, α*change)
                else
                    α = min(max_α, α/change)
                end
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

    new_delta = max(delta*exp(γ),10^-8)
    return new_delta

end




