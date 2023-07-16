using Parameters
using Random
using LinearAlgebra
using ForwardDiff
using ProgressMeter
using ValueShapes
using UnPack
import BAT: AbstractSamplingAlgorithm, bat_sample_impl, AbstractMeasureOrDensity, transform_and_unshape, totalndof


abstract type AbstractECMCState end

include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/ecmc_direction_changes.jl")
include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/ecmc_tuning.jl")
include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/ecmc_refresh_delta.jl")
include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/ecmc_ECMCstate.jl")

include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/bschaefer/performance_tests/ecmc_sampler_cluster.jl")
