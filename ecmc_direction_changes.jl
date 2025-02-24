# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
abstract type AbstractECMCDirection

Abstract type for the direction changes in the sampling region.
"""
abstract type AbstractECMCDirection end
export AbstractECMCDirection



"""
struct BAT.ReverseDirection <: AbstractECMCDirection

Use the reverse direction: current_direction --> -current_direction
"""
struct ReverseDirection <: AbstractECMCDirection end
export ReverseDirection


"""
struct BAT.RefreshDirection <: AbstractECMCDirection

Refresh the direction in the multidimensional space using a normal distribution 
and accept it with Metropolis probability.
"""
struct RefreshDirection <: AbstractECMCDirection end
export RefreshDirection


"""
struct BAT.ReflectDirection <: AbstractECMCDirection

Reflect the direction on the plane that has the gradient as its normal vector 
and accept it with Metropolis probability.
"""
struct ReflectDirection <: AbstractECMCDirection end
export ReflectDirection


"""
struct BAT.StochasticReflectDirection <: AbstractECMCDirection

Reflect the direction on the plane that has the gradient as its normal vector, 
randomize the perpendicular component, and accept it with Metropolis probability.
"""
struct StochasticReflectDirection <: AbstractECMCDirection end
export StochasticReflectDirection




function _accepted_lift_vector(density, C, current_energy, delta, proposed_lift_vector, current_lift_vector)
    

    prob_current_C = 1.0 .- exp.(-_energy_difference(density, C + current_lift_vector * delta, current_energy))
    prob_proposed_C = 1.0 .- exp.(-_energy_difference(density, C - proposed_lift_vector * delta, current_energy))
    acceptance_probability = minimum([1.0, prob_proposed_C[1] / prob_current_C[1]])

    rand(Uniform(0, 1)) <= acceptance_probability ? (return proposed_lift_vector) : (return -current_lift_vector)
end




function _change_direction(
    direction_type::ReverseDirection,
    C::Vector{Float64},
    current_energy::Float64,
    delta::Float64,
    lift_vector::Vector{Float64},
    proposed_C::Vector{Float64},
    density::AbstractMeasureOrDensity
)
    return -lift_vector
end




function _change_direction(
    direction_type::RefreshDirection,
    C::Vector{Float64},
    current_energy::Float64,
    delta::Float64,
    lift_vector::Vector{Float64},
    proposed_C::Vector{Float64},
    density::AbstractMeasureOrDensity
)
    current_lift_vector = lift_vector

    D = length(current_lift_vector)
    proposed_lift_vector = refresh_lift_vector(D)

    return _accepted_lift_vector(density, C, current_energy, delta, proposed_lift_vector, current_lift_vector)
end





function _change_direction(
    direction_type::ReflectDirection,
    C::Vector{Float64},
    current_energy::Float64,
    delta::Float64,
    lift_vector::Vector{Float64},
    proposed_C::Vector{Float64},
    density::AbstractMeasureOrDensity
)
    current_lift_vector = lift_vector
    n_normal = normalize(_energy_gradient(density, C))

    proposed_lift_vector = current_lift_vector - 2 * dot(current_lift_vector, n_normal) * n_normal

    return _accepted_lift_vector(density, C, current_energy, delta, proposed_lift_vector, current_lift_vector)
end




# https://arxiv.org/pdf/1702.08397v1.pdf  (Eq. 23)
# direct update
function _change_direction(
    direction_type::StochasticReflectDirection,
    C::Vector{Float64},
    current_energy::Float64,
    delta::Float64,
    lift_vector::Vector{Float64},
    proposed_C::Vector{Float64},
    density::AbstractMeasureOrDensity
)
    current_lift_vector = lift_vector
    n_normal = normalize(_energy_gradient(density, C))
    D = length(C)

    n_perp = normalize(current_lift_vector - dot(current_lift_vector, n_normal) * n_normal)
    
    a = rand(Uniform(0., 1.))^(1. /(D-1))
    parallel_component = -sqrt(1-a^2)

    new_lift_vector = sqrt(1. - parallel_component^2) * n_perp + parallel_component * n_normal

    old_lift_vector = current_lift_vector
    prob_current_C = 1.0 .- exp.(-_energy_difference(density, C + old_lift_vector * delta, C))
    prob_proposed_C = 1.0 .- exp.(-_energy_difference(density, C - new_lift_vector * delta, C))

    proposal_lift = maximum([0., dot(n_normal, -new_lift_vector)])
    proposal_current = maximum([0., dot(n_normal, old_lift_vector)])

    acceptance_probability  = minimum([1.0, (proposal_current* prob_proposed_C) / (proposal_lift * prob_current_C)])

    rand(Uniform(0, 1)) <= acceptance_probability ? (return new_lift_vector) : (return -old_lift_vector)
end




#---------BENS TESTS--------------

struct GradientRefreshDirection <: AbstractECMCDirection end
export GradientRefreshDirection

function _change_direction(
    direction_type::GradientRefreshDirection,
    C::Vector{Float64},
    current_energy::Float64,
    delta::Float64,
    lift_vector::Vector{Float64},
    proposed_C::Vector{Float64},
    density::AbstractMeasureOrDensity
)
    current_lift_vector = lift_vector

    D = length(current_lift_vector)
    refreshed_vector = refresh_lift_vector(D)

    n_normal = - normalize(_energy_gradient(density, C))
    n_normal_part = dot(n_normal, refreshed_vector)

    if n_normal_part > 0
        proposed_lift_vector = refreshed_vector
    else
        proposed_lift_vector = refreshed_vector - 2. * n_normal_part * n_normal
    end
    

    return _accepted_lift_vector(density, C, current_energy, delta, proposed_lift_vector, current_lift_vector)
end




struct TestDirection <: AbstractECMCDirection end
export TestDirection


function _change_direction(
    direction_type::TestDirection,
    C::Vector{Float64},
    current_energy::Float64,
    delta::Float64,
    lift_vector::Vector{Float64},
    proposed_C::Vector{Float64},
    density::AbstractMeasureOrDensity
)
    D = length(C)
    current_lift_vector = lift_vector
    n_normal = - normalize(_energy_gradient(density, C))

    
    parallel = dot(current_lift_vector, n_normal) * n_normal
    refreshed = rand(Normal(0., 1.), D)
    perpendicular = normalize(refreshed - dot(refreshed, n_normal) * n_normal)
    a = sqrt(1. - dot(parallel, parallel))

    proposed_lift_vector = a * perpendicular + parallel


    return _accepted_lift_vector(density, C, current_energy, delta, proposed_lift_vector, current_lift_vector)
end