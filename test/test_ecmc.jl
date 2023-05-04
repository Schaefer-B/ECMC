using Test
using BAT
using DensityInterface
using InverseFunctions
using Distributions
using Random
using ForwardDiff
using Plots
ENV["JULIA_DEBUG"] = "BAT"

include("../ecmc.jl")

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

@test isapprox(reflect_direction, [-0.23608589286146675, -0.19139932332469628, 0.82892242886362, 0.4695927567046678], atol=1e-15)


Random.seed!(360)
update_direction = _change_direction(
    StochasticReflectDirection(),
    C, delta, current_lift_vector, proposed_C, density,
)

@test isapprox(update_direction, [-0.126265499510203, -0.20892331575497952, 0.9434747870406662, 0.2241949999129186], atol=1e-15)


#----- Test gen_samples ----------------------------
# The tests below are broken since we switched to running multiple chains in parallel.


# include("ecmc.jl")
# Random.seed!(100)

# samples =  _ecmc_sample(density, algorithm, ecmc_state=ECMCState(density, algorithm)[1])

# @test samples[1] == [0.1752868201955589, 0.21633856381962946, 0.3760037403013978, 0.7683091006566242]

# @test samples[5] == [0.36439140096201095, 0.448605996531853, 0.48614075945638663, 0.8562747332372502]

# @test ecmc_state.C == [0.47625582050342424, 0.37490323781350227, 0.43982852316025023, 0.4772685498054118]
# @test ecmc_state.lift_vector == [0.6728482961695516, 0.13726496474197558, 0.7200402218971148, -0.09987781859086269]
# @test ecmc_state.delta == 0.11912622245467414
# @test ecmc_state.remaining_jumps_before_refresh == 50

# @test ecmc_state.n_steps == 500000
# @test ecmc_state.n_acc == 308872
# @test ecmc_state.n_lifts == 191128
# @test ecmc_state.n_acc_lifts == 100398

# @test ecmc_state.mfp == 30720.867363099343
# @test ecmc_state.mfps == 3
# #@test ecmc_state.mfps_arr[1] == 3
# #@test ecmc_state.mfps_arr[5000] == 1

# # #----- Test _run_ecmc ----------------------------
# @test _run_ecmc(density, algorithm, ecmc_state) == [0.4762558205034242, 0.37490323781350227, 0.43982852316025023, 0.4772685498054118]
# @test _run_ecmc(density, algorithm, ecmc_state) == [0.5584206020887925, 0.430354096532232, 0.44647229539588945, 0.39181708305702034]

# #----- Test _tune_ecmc ----------------------------
# @test_broken _tune_ecmc(density, algorithm, ecmc_state) == [0.5869046274021789, 0.3667092530181758, 0.35468853421667024, 0.4215806471738283]
# @test_broken _tune_ecmc(density, algorithm, ecmc_state) == [0.6372239168635468, 0.398628090144305, 0.49677526905572816, 0.3702593019166032]

# @test_broken ecmc_state.delta_arr == [0.1, 0.11912622245467414, 0.09344378119440809, 0.09344378119440809, 0.09344378119440809, 0.07329824896205839, 0.07329824896205839, 0.07329824896205839, 0.07329824896205839, 0.057495966867305084, 0.057495966867305084]
# @test_broken ecmc_state.acc_C == [0.6177444096229883, 0.6177444096229883, 0.6177439386575949, 0.6177447031483119, 0.6177447031483119, 0.6177442321845701, 0.6177449966701132, 0.6177457611525985, 0.6177457611525985, 0.6177452901883924]



# #----- Test _run_ecmc -----------------------------
# include("ecmc.jl")
# Random.seed!(300)

# ecmc_state2 = ECMCState(
#     C, current_lift_vector, delta, 
#     1, 1, 1, 1, 1, 
#     1., 1,
# )


# @test _run_ecmc(density, algorithm, ecmc_state2) == [0.5423708458880638, 0.5536665920416103, 0.082002510994545, 0.502345099889361]
# @test _run_ecmc(density, algorithm, ecmc_state2) == [0.7287446180895639, 0.5030543863343889, 0.3093926985429676, 0.6873704522590462]


# Random.seed!(320)
# ecmc_state2 = ECMCState(
#     C, current_lift_vector, delta, 
#     1, 1, 1, 1, 1, 
#     1., 1, [], 1., [], []
# )


# ECMCTunerState(density, algorithm)
# _ecmc_tuning(density, algorithm)

# @test _tune_ecmc(density, algorithm, ecmc_state2) == [0.2986022009563688, 0.5234866033148833, 0.30184470270190045, 0.9477163775998195]
# @test _tune_ecmc(density, algorithm, ecmc_state2) == [0.37391281851966385, 0.6415877184743679, 0.6551727166665336, 0.9854166517381415]










#-------Testing of _run_ecmc!, _ecmc_sample, _ecmc_multichain_sample---------

#-------preparation for the following tests---------
#using Random123
#rng = Philox4x()
dimension = 16 #dimension of the distribution


#posterior density to be called during tests of algorithms with different trafos
function _initialize_density(trafo = PriorToUniform(), dimension = 8)
    D = dimension
    μ = fill(0.0, D)
    σ = collect(range(0.1, 10, D))

    prior = BAT.NamedTupleDist(
        a = Uniform.(-6*σ, 6*σ)
    )

    likelihood = let D = D, μ = μ, σ = σ
        logfuncdensity(params -> begin

            return logpdf(MvNormal(μ, σ), params.a)

        end)
    end

    posterior = PosteriorMeasure(likelihood, prior)

    density_notrafo = convert(AbstractMeasureOrDensity, posterior)
    density, trafo = transform_and_unshape(trafo, density_notrafo)
    shape = varshape(density)

    return density, trafo, shape
end


#function to generate ecmcstates with a fixed starting- and liftvector
#called later so different stepamplitued etc can be used
function _initialize_ecmc_state(algorithm, density, dimension=8, set_C_to_origin=false)
    D = dimension
    

    #start_C = [trafo(rand(BAT.getprior(density))) for i in 1:nstates]

    if set_C_to_origin == true
        C = fill(0, D)
        start_liftvector = fill(1/sqrt(D), D)
        delta = algorithm.step_amplitude

        initialized_state = ECMCState(
            C, start_liftvector, delta, algorithm.step_amplitude, algorithm.step_var, algorithm.remaining_jumps_before_refresh, 0, 0, 0, 0, 0., 0
        )

    else
        C = rand(BAT.getprior(density))
        tuning_state = ECMCTunerState(density, algorithm)[1]

        tuning_samples, tuned_state = _ecmc_tuning(algorithm.tuning, density, algorithm, ecmc_tuner_state=tuning_state)
        initialized_state = ECMCState(tuned_state, algorithm)
    end

    return initialized_state
end



#one algorithm for each change direction algorithm
direction_change_algorithms = [ReverseDirection(), RefreshDirection(), ReflectDirection(), StochasticReflectDirection()]
#direction_change_algorithms = [ReverseDirection(), RefreshDirection(), ReflectDirection()]

algorithms = [
    ECMCSampler(
        trafo = PriorToUniform(),
        nsamples = 10^5,
        nburnin = 10^3,
        nchains = 1,
        chain_length = 5, #jumps_before_sample
        remaining_jumps_before_refresh=50,
        step_amplitude = 0.04,
        step_var = 0.01,
        direction_change = algo,
        tuning = MFPSTuner(),
        factorized = false
    ) for algo in direction_change_algorithms
]








#------_run_ecmc! Tests------

run_ecmc_results = Dict()
runs = 100

for algorithm in algorithms
    
    density, trafo, shape = _initialize_density(algorithm.trafo, dimension)

    ecmc_state = _initialize_ecmc_state(algorithm, density, dimension, true)
    
    #println(algorithm.direction_change, "start C = ", ecmc_state.C)
    #println(algorithm.direction_change, "start lift = ", ecmc_state.lift_vector)
    #println(algorithm.direction_change, "delta = ", ecmc_state.delta)

    samples = []
    Random.seed!(42)
    for i in 1:runs
        next_sample = [_run_ecmc!(ecmc_state, density, algorithm)]
        push!(samples, next_sample)
    end
    #println("samples = ", samples)
    samples_trafo = shape.(convert_to_BAT_samples(samples, density))
    samples_notrafo = inverse(trafo).(samples_trafo)
    sample_dict = Dict(algorithm.direction_change => samples_notrafo)

    merge!(run_ecmc_results, sample_dict)
end



@test isapprox(run_ecmc_results[ReverseDirection()].v[1].a, [-0.54, -4.104, -7.667999999999999, -11.232, -14.796000000000001, -18.36, -21.924, -25.488, -29.052, -32.616, -36.18, -39.744, -43.308, -46.872, -50.436, -54.0], atol=1e-15)
@test isapprox(run_ecmc_results[RefreshDirection()].v[1].a, [-0.54, -4.104, -7.667999999999999, -11.232, -14.796000000000001, -18.36, -21.924, -25.488, -29.052, -32.616, -36.18, -39.744, -43.308, -46.872, -50.436, -54.0], atol=1e-15)
@test isapprox(run_ecmc_results[ReflectDirection()].v[1].a, [-0.54, -4.104, -7.667999999999999, -11.232, -14.796000000000001, -18.36, -21.924, -25.488, -29.052, -32.616, -36.18, -39.744, -43.308, -46.872, -50.436, -54.0], atol=1e-15)
@test isapprox(run_ecmc_results[StochasticReflectDirection()].v[1].a, [-0.54, -4.104, -7.667999999999999, -11.232, -14.796000000000001, -18.36, -21.924, -25.488, -29.052, -32.616, -36.18, -39.744, -43.308, -46.872, -50.436, -54.0], atol=1e-15)

@test isapprox(run_ecmc_results[ReverseDirection()].v[100].a, [-0.01875905563770197, -0.13481600712882802, 1.9263659301473322, 0.12372814499449802, 2.13832820561122, 3.3218469019043155, -1.3170374341169975, -1.2920113792038705, 0.8867666721301006, 1.944403627638799, 1.2859929052670012, 1.516161230928077, 2.12242276073399, 1.0762365033922663, 2.640575758119745, 2.6281928296924164], atol=1e-15)
@test isapprox(run_ecmc_results[RefreshDirection()].v[100].a, [0.07534389424352739, 0.5084252992325906, 0.7626410228943765, -1.995879176909492, 2.6153794469306355, 0.9946184856324258, 0.3428133555360695, -3.7077441775231037, 5.281683300305765, -1.0458899415631535, -0.7926682541016845, 1.3559750594345559, -3.6088106578072257, 4.008609129306578, -1.9668755038233243, -2.6815303239411534], atol=1e-15)
@test isapprox(run_ecmc_results[ReflectDirection()].v[100].a, [0.0238002834318759, 0.548652931082767, -1.504017603143275, -1.2962734739743418, 3.5419781302526836, -4.6405802720772815, -3.068395300202617, -7.43569926904571, -4.404596620480508, 0.44901806765918906, -11.524435189793106, -4.624163844472164, 0.06602032166976102, -15.891062065479893, -10.776390274323184, 4.393433043919558], atol=1e-15)
@test isapprox(run_ecmc_results[StochasticReflectDirection()].v[100].a, [-0.006102245871822287, -0.08657770542281007, 2.8400220765621995, -1.4868965424502552, -0.8444150972582349, 0.3844184884983335, 0.2364111577358763, 10.595427500334829, 1.5197365593320527, -4.455083633325948, 0.7984591925193243, 6.5312962858262225, -17.8440013620808, 4.5940554218472585, -4.8361453406851425, -2.3829548173632418], atol=1e-15)




#---------_ecmc_sample---------

#generating samples using _ecmc_sample and putting them in a dictionary so you can specify tests for algorithms and samples later as needed
#key is the algorithm.direction_change
results = Dict()

Threads.@threads for algorithm in algorithms
    
    density, trafo, shape = _initialize_density(algorithm.trafo, dimension)

    ecmc_state = _initialize_ecmc_state(algorithm, density, dimension, false)

    samples = [_ecmc_sample(density, algorithm, ecmc_state = ecmc_state, chainid = 0)]

    samples_trafo = shape.(convert_to_BAT_samples(samples, density))
    samples_notrafo = inverse(trafo).(samples_trafo)
    sample_dict = Dict(algorithm.direction_change => samples_notrafo)

    merge!(results, sample_dict)
end




@test isapprox(mean(results[ReverseDirection()]).a, fill(0, dimension), atol=1)
@test isapprox(mean(results[RefreshDirection()]).a, fill(0, dimension), atol=1)
@test isapprox(mean(results[ReflectDirection()]).a, fill(0, dimension), atol=1)
@test isapprox(mean(results[StochasticReflectDirection()]).a, fill(0, dimension), atol=1)




@test isapprox(sqrt.(diag(cov(unshaped.(results[ReverseDirection()])))), collect(range(0.1, 10, dimension)), atol=1)
@test isapprox(sqrt.(diag(cov(unshaped.(results[RefreshDirection()])))), collect(range(0.1, 10, dimension)), atol=1)
@test isapprox(sqrt.(diag(cov(unshaped.(results[ReflectDirection()])))), collect(range(0.1, 10, dimension)), atol=1)
@test isapprox(sqrt.(diag(cov(unshaped.(results[StochasticReflectDirection()])))), collect(range(0.1, 10, dimension)), atol=1)












