
abstract type AbstractStepAmplitudeVariation end
export AbstractStepAmplitudeVariation

struct NoVariation <: AbstractStepAmplitudeVariation end
export NoVariation

struct UniformVariation <: AbstractStepAmplitudeVariation end
export UniformVariation

struct NormalVariation <: AbstractStepAmplitudeVariation end
export NormalVariation

struct ExponentialVariation <: AbstractStepAmplitudeVariation end
export ExponentialVariation



function refresh_delta(step_amplitude::Float64, step_var::Float64, delta::Float64, variation_type::NoVariation) 
    return delta
end

function refresh_delta(step_amplitude::Float64, step_var::Float64, delta::Float64, variation_type::UniformVariation)
    step_amplitude = step_amplitude
    step_var = step_var
    #return rand(Normal(step_amplitude, step_var*step_amplitude))
    u = rand(Uniform(1-step_var, 1+step_var))
    new_delta =  u * step_amplitude
    return max(new_delta, 10^-15)
end

function refresh_delta(step_amplitude::Float64, step_var::Float64, delta::Float64, variation_type::NormalVariation)
    step_amplitude = step_amplitude
    step_var = step_var
    new_delta = rand(Normal(step_amplitude, step_amplitude*step_var))
    return max(new_delta, 10^-15)
end

function refresh_delta(step_amplitude::Float64, step_var::Float64, delta::Float64, variation_type::ExponentialVariation)
    step_amplitude = step_amplitude
    step_var = step_var
    u = rand(Normal(0, 1))
    new_delta = step_amplitude * exp(u * step_var)
    return max(new_delta, 10^-15)
end