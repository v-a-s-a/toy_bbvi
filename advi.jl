
using Distributions
using Flux
using Plots
using LinearAlgebra

"""
A 2D non-gaussian log-density.
"""
function log_density(params)
    mu, log_sigma = params
    d1 = Normal(0, 1.35)
    d2 = Normal(0, exp(log_sigma))
    d1_density = logpdf(d1, log_sigma)
    d2_density = logpdf(d2, mu)
    return d1_density + d2_density
end

x = -2:0.1:2
y = -4:0.1:2

X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))

Z = Array{Float64}(undef, size(X))
for i in 1:size(X)[1]
    for j in 1:size(X)[2]
        Z[i, j] = exp(log_density([X[i, j], Y[i, j]]))
    end
end

plot(contour(x, y, Z))

D = 2  # dimensions of approximate posterior
num_samples = 100
"""
Entropy of the Gaussian distribution.
"""
function gaussian_entropy(log_std)
    H = 0.5 * D * (1.0 + log(2 * pi)) + sum(log_std)
    return H
end

function variational_objective(parameters; D=2)
    mu, log_std = parameters
    samples = rand(Normal(), num_samples, D) .* sqrt.(log_std) .+ mu
    log_px = mapslices(log_density, samples; dims=2) # eval log(target) for all samples of params (i.e. cols)
    elbo = gaussian_entropy(log_std) + mean(log_px)
    return -elbo
end

mu = Flux.Tracker.param(reshape([-1, -1], 1, :))
sigma = Flux.Tracker.param(reshape([1, 1], 1, :))

parameters = Flux.Tracker.Params([mu, sigma])
elbo_gradient = Flux.Tracker.gradient(() -> variational_objective(parameters), parameters)

elbo = [variational_objective(parameters)]
steps = 1000

opt = ADAM(0.01)
for i in 1:steps
    println(i)
    for p in (mu, sigma)
        Flux.Tracker.update!(p, Flux.Optimise.update!(opt, p, Flux.data(elbo_gradient[p])))
        push!(elbo, variational_objective(parameters))
    end
end

q = MultivariateNormal(mu[1,:].data, Diagonal(exp.(2*sigma[1,:].data)))
Z_q = Array{Float64}(undef, size(X))
for i in 1:size(X)[1]
    for j in 1:size(X)[2]
        Z[i, j] = pdf(q, [X[i, j], Y[i, j]])
    end
end

plot(contour(x, y, Z_q))

