using BAT
using Plots
using ValueShapes
using Distributions
using IntervalSets
using ForwardDiff
using InverseFunctions
using DensityInterface
using BenchmarkTools

gr(size=(1.3*850, 1.3*600), thickness_scaling = 1.5)



#---- Multivariate Gaussian --------------------------
D = 16
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


#---- Funnel  --------------------------
D = 16
truth = rand(BAT.FunnelDistribution(a=0.5, b=1., n=D), Int(1e6))

likelihood = let D = D
    logfuncdensity(params -> begin

       return logpdf(BAT.FunnelDistribution(a=0.5, b=1., n=D), params.a)
    end)
end 

σ = 10*ones(D)
prior = BAT.NamedTupleDist(
    a = Uniform.(-σ, σ)
)


#-------------------------------------------
likelihood = logfuncdensity(params -> begin

    r1 = logpdf.(
    MixtureModel(Normal[
    Normal(-10.0, 1.2),
    Normal(0.0, 1.8),
    Normal(10.0, 2.5)], [0.1, 0.3, 0.6]), params.a[1])

    r2 = logpdf.(
    MixtureModel(Normal[
    Normal(-5.0, 2.2),
    Normal(5.0, 1.5)], [0.3, 0.7]), params.a[2])

    r3 = logpdf.(Normal(2.0, 1.5), params.a[3])

    return r1+r2+r3
end)

prior = BAT.NamedTupleDist(
    a = [-20..20, -20.0..20.0, -10..10]
)







#-----------------


posterior = PosteriorMeasure(likelihood, prior);
logdensityof(posterior, rand(prior))


#--------------------------

#for plotting
function idontwannacalliteverytime(bucket=100)
    new_acc_C = []
    bucket_width = bucket
    k_max = Int(floor(length(tuning_state.acc_C)/bucket_width))
    for k in 1:k_max
        for i in (1+(k-1)*bucket_width):(k*bucket_width)
            acc = (tuning_state.acc_C[k*bucket_width]*k*bucket_width - tuning_state.acc_C[(1+(k-1)*bucket_width)]*(1+(k-1)*bucket_width))/(bucket_width-1)
            push!(new_acc_C, acc)
        end
    end


    p1 = plot(title="MFPSTuner")
    plot!(new_acc_C, lw=2, label="Current ratio", ylabel="Acceptance ratio")
    target_acc = algorithm.tuning.target_mfps/(algorithm.tuning.target_mfps+1)
    plot!([target_acc], st=:hline, lw=2, label="Target")

    p2 = plot(tuning_state.delta_arr[1:end], lw=2, label = "Current delta", xlabel="Steps", ylabel="Delta")
    N = minimum([length(tuning_state.acc_C), Int(floor(0.3*tuning_state.n_steps))])
    plot!([mean(tuning_state.delta_arr[end-N+1:end])], st=:hline, lw=2, label="Mean[N:end]")

    p = plot(p1, p2, layout=(2,1) ,legend=true)
end



include("../ecmc.jl")

using Optim
include("../examples/tuning_with_optim.jl")


algorithm = ECMCSampler(
    trafo = PriorToUniform(),
    nsamples=10^4,
    nburnin = 0,
    nchains = 1,
    chain_length=5, 
    remaining_jumps_before_refresh=50,
    step_amplitude=10^-2,
    factorized = false,
    #step_var=1.5*0.04,
    direction_change = RefreshDirection(),
    tuning = MFPSTuner(),
)


sample = bat_sample(posterior, algorithm);
samples = sample.result

tuning_state = sample.ecmc_tuning_state[1] # tuning state for chain 1


#---------Ben Plot-----------
idontwannacalliteverytime(240)

tuning_state.tuned_delta
length(tuning_state.delta_arr)
std(tuning_state.delta_arr[end-Int(floor(tuning_state.n_steps*0.2)):end])/tuning_state.tuned_delta


png("MFPSTuner - google spike")

#average steps used after x tuning_samples
average_steps_calc = 0
x = 10
for i in 1:x
    sample = bat_sample(posterior, algorithm)
    tuning_state = sample.ecmc_tuning_state[1]
    average_steps_calc += length(tuning_state.delta_arr)
end
average_steps = average_steps_calc/x


#delta at start
plot(tuning_state.delta_arr[1:100], label = "delta")

plot(tuning_state.reject_step_arr[1:10])


#----- Plot Acc_C -------------------------------------
plot(tuning_state.acc_C, lw=2, label="Acc_C")
target_acc = algorithm.tuning.target_mfps/(algorithm.tuning.target_mfps+1)
plot!([target_acc], st=:hline, lw=2, label="Target Acc_C")



#----- Plot new Acc_C -------------------------------------
new_acc_C = []
bucket_width = 2500 # set to iter_steps if optim tuning
k_max = Int(floor(length(tuning_state.acc_C)/bucket_width))
for k in 1:k_max
    for i in (1+(k-1)*bucket_width):(k*bucket_width)
        acc = (tuning_state.acc_C[k*bucket_width]*k*bucket_width - tuning_state.acc_C[(1+(k-1)*bucket_width)]*(1+(k-1)*bucket_width))/(bucket_width-1)
        push!(new_acc_C, acc)
    end
end

plot(new_acc_C, lw=2, label="new_Acc_C")
target_acc = algorithm.tuning.target_mfps/(algorithm.tuning.target_mfps+1)
plot!([target_acc], st=:hline, lw=2, label="Target Acc_C")

#----- Plot Delta ------------------------------------------
N = minimum([length(tuning_state.acc_C), Int(floor(0.3*tuning_state.n_steps))])

plot(tuning_state.delta_arr[250:30000], label = "delta")
plot!([mean(tuning_state.delta_arr)], st=:hline, label="mean")
plot!([mean(tuning_state.delta_arr[end-N+1:end])], st=:hline, label="mean[N:end]")

tuning_state.delta_arr[end]

#----- Plot MFPS ------------------------------------------
plot(tuning_state.mfps_arr, st=:histogram)


mean(tuning_state.mfps_arr)
best_delta = mean(tuning_state.delta_arr[end-N+1:end])
tuning_state.tuned_delta


#----- Plot samples vs. truth ------------------------------
p = plot(layout=(4,4), size=(1600, 1000))
for i in 1:D
    p = plot!(samples, i, subplot=i, legend=false)
    p = plot!(truth[i, :], subplot=i, lw=2, lc=:black, st=:stephist, normed=true)
end 
p



#-----------save plot--------------------
#p = plot(layout=(2,1), size=(1600, 1000))

#p = plot!(samples, i, subplot=i, legend=false)


idontwannacalliteverytime(1000)

tuning_state.tuned_delta

png("MFPSTuner - test 1 mit gradient")