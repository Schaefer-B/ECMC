using Test
using BAT
using DensityInterface
using InverseFunctions
using Distributions
using Random
using ForwardDiff
using Plots

include("ecmc.jl")

#---- Multivariate Gaussian --------------------------
D = 4
μ = fill(0.0, D)
σ = collect(range(1, 10, D))
truth = rand(MvNormal(μ, σ), Int(1e6))

likelihood = let D = D, μ = μ, σ = σ
    logfuncdensity(params -> begin

       return logpdf(MvNormal(μ, σ), params.a)
    end)
end 


prior = BAT.NamedTupleDist(
    a = Uniform.(-5*σ, 5*σ)
)

posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))




algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^5,
    nburnin = 0,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=0.1,
    factorized = false,
    step_var=0.2,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(),
)

#------------------------------------------------------
Random.seed!(1234)

density_notrafo = convert(AbstractMeasureOrDensity, posterior)
density, trafo = transform_and_unshape(algorithm.trafo, density_notrafo)

C = trafo(rand(prior))
delta = 0.1
current_lift_vector = normalize(rand(Uniform(-1, 1), D))
proposed_C = C + delta * current_lift_vector


#----- Test Refresh Functions ------------------------
samples, ecmc_state = _generate_ecmc_samples(density, algorithm)

Random.seed!(1234)
@test refresh_lift_vector(D) == [-0.25409284917356173, 0.7679446890333682, -0.2963751848387071, 0.5077987085412164]
@test refresh_delta(ecmc_state, 0.5, 0.1) == 0.4853111644399212 # for uniform 


#----- Test Energy Functions ------------------------
@test _energy_function(density, C) == 22.676195804702928

@test _energy_difference(density, proposed_C, C) == 0.6643895110929208

@test _energy_gradient(density, C) == [-17.402327113640514, 4.905113631556688, -28.14133451811693, 39.424542820098836]


#----- Test update schemes ----------------------------
Random.seed!(25)

flip_direction = _change_direction(
    ReverseDirection(),
    C, delta, current_lift_vector, proposed_C, density,
)

@test flip_direction == -current_lift_vector



fullref_update_direction = _change_direction(
    RefreshDirection(),
    C, delta, current_lift_vector, proposed_C, density,
)

@test fullref_update_direction == [-0.2662094336712443, 0.5649491805767868, -0.4399806715753523, -0.6452766611540778]



reflect_direction = _change_direction(
    ReflectDirection(),
    C, delta, current_lift_vector, proposed_C, density,
)

@test reflect_direction == [-0.23608589286146675, -0.19139932332469628, 0.82892242886362, 0.4695927567046678]


Random.seed!(360)
update_direction = _change_direction(
    StochasticReflectDirection(),
    C, delta, current_lift_vector, proposed_C, density,
)

@test update_direction == [-0.126265499510203, -0.20892331575497952, 0.9434747870406662, 0.2241949999129186]


#----- Test gen_samples ----------------------------
include("ecmc.jl")
Random.seed!(100)
samples, ecmc_state = _generate_ecmc_samples(density, algorithm)

@test samples[1] == [0.1752868201955589, 0.21633856381962946, 0.3760037403013978, 0.7683091006566242]

@test samples[5] == [0.36439140096201095, 0.448605996531853, 0.48614075945638663, 0.8562747332372502]

@test ecmc_state.C == [0.47625582050342424, 0.37490323781350227, 0.43982852316025023, 0.4772685498054118]
@test ecmc_state.lift_vector == [0.6728482961695516, 0.13726496474197558, 0.7200402218971148, -0.09987781859086269]
@test ecmc_state.delta == 0.11912622245467414
@test ecmc_state.remaining_jumps_before_refresh == 50

@test ecmc_state.n_steps == 500000
@test ecmc_state.n_acc == 308872
@test ecmc_state.n_lifts == 191128
@test ecmc_state.n_acc_lifts == 100398

@test ecmc_state.mfp == 30720.867363099343
@test ecmc_state.mfps == 3
#@test ecmc_state.mfps_arr[1] == 3
#@test ecmc_state.mfps_arr[5000] == 1

# #----- Test _run_ecmc ----------------------------
@test _run_ecmc(density, algorithm, ecmc_state) == [0.4762558205034242, 0.37490323781350227, 0.43982852316025023, 0.4772685498054118]
@test _run_ecmc(density, algorithm, ecmc_state) == [0.5584206020887925, 0.430354096532232, 0.44647229539588945, 0.39181708305702034]

#----- Test _tune_ecmc ----------------------------
@test_broken _tune_ecmc(density, algorithm, ecmc_state) == [0.5869046274021789, 0.3667092530181758, 0.35468853421667024, 0.4215806471738283]
@test_broken _tune_ecmc(density, algorithm, ecmc_state) == [0.6372239168635468, 0.398628090144305, 0.49677526905572816, 0.3702593019166032]

@test_broken ecmc_state.delta_arr == [0.1, 0.11912622245467414, 0.09344378119440809, 0.09344378119440809, 0.09344378119440809, 0.07329824896205839, 0.07329824896205839, 0.07329824896205839, 0.07329824896205839, 0.057495966867305084, 0.057495966867305084]
@test_broken ecmc_state.acc_C == [0.6177444096229883, 0.6177444096229883, 0.6177439386575949, 0.6177447031483119, 0.6177447031483119, 0.6177442321845701, 0.6177449966701132, 0.6177457611525985, 0.6177457611525985, 0.6177452901883924]



#----- Test _run_ecmc -----------------------------
include("ecmc.jl")
Random.seed!(300)

ecmc_state2 = ECMCState(
    C, current_lift_vector, delta, 
    1, 1, 1, 1, 1, 
    1., 1,
)


@test _run_ecmc(density, algorithm, ecmc_state2) == [0.5423708458880638, 0.5536665920416103, 0.082002510994545, 0.502345099889361]
@test _run_ecmc(density, algorithm, ecmc_state2) == [0.7287446180895639, 0.5030543863343889, 0.3093926985429676, 0.6873704522590462]


Random.seed!(320)
ecmc_state2 = ECMCState(
    C, current_lift_vector, delta, 
    1, 1, 1, 1, 1, 
    1., 1, [], 1., [], []
)


ECMCTunerState(density, algorithm)
_ecmc_tuning(density, algorithm)

@test _tune_ecmc(density, algorithm, ecmc_state2) == [0.2986022009563688, 0.5234866033148833, 0.30184470270190045, 0.9477163775998195]
@test _tune_ecmc(density, algorithm, ecmc_state2) == [0.37391281851966385, 0.6415877184743679, 0.6551727166665336, 0.9854166517381415]


#-------- BAT Sample --------------------------------------------------------------------

using Test
using BAT
using DensityInterface
using InverseFunctions
using Distributions
using Random
using ForwardDiff
using Plots

include("ecmc.jl")

D = 4
μ = fill(0.0, D)
σ = collect(range(1, 10, D))
truth = rand(MvNormal(μ, σ), Int(1e6))

likelihood = let D = D, μ = μ, σ = σ
    logfuncdensity(params -> begin

       return logpdf(MvNormal(μ, σ), params.a)
    end)
end 


prior = BAT.NamedTupleDist(
    a = Uniform.(-5*σ, 5*σ)
)

posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))

#Random.seed!(100)

algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^5,
    nburnin = 0,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=0.04,
    factorized = false,
    step_var=1.5*0.04,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(),
)


sampling_result = bat_sample(posterior, algorithm)
samples = sampling_result.result


p = plot(layout=(4,4), size=(1600, 1000))
for i in 1:D
    p = plot!(samples, i, subplot=i, legend=false)
    p = plot!(truth[i, :], subplot=i, lw=2, lc=:black, st=:stephist, normed=true)
end 
p