using Parameters
using Random
using LinearAlgebra
using ForwardDiff
using ProgressMeter
using ValueShapes
using UnPack
import BAT: AbstractSamplingAlgorithm, bat_sample_impl, AbstractMeasureOrDensity, transform_and_unshape, totalndof


abstract type AbstractECMCState end

#include("ecmc_direction_changes.jl")
#include("ecmc_tuning.jl")
#include("ecmc_ECMCstate.jl")
#include("ecmc_sampler.jl")


include("ecmc_direction_changes.jl")
include("ecmc_ECMCstate.jl")
include("ecmc_tuning.jl")
include("ecmc_sampler.jl")