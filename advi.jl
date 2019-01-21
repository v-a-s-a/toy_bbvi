
using Distributions
using Flux.Tracker, Flux.Optimise
using Plots

function log_density(params)
    """
    A 2D non-gaussian log-density.
    """
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
Z = exp.(log_density.(X, Y))

plot(contour(x, y, Z))

D = 2  # dimensions of approximate posterior
num_samples = 100

function gaussian_entropy(log_std)
    """
    Entropy of the Gaussian distribution.
    """
    H = 0.5 * D * (1.0 + log(2 * pi)) + sum(log_std)
    return H
end

function variational_objective(mu, log_std; D=2)
    samples = rand(Normal(), num_samples, D) .* sqrt.(log_std) .+ mu
    log_px = mapslices(log_density, samples; dims=2) # eval log(target) for all samples of params (i.e. cols)
    evidence_lower_bound = gaussian_entropy(log_std) + mean(log_px)
    return -evidence_lower_bound
end

mu = param(reshape([-1, -1], 1, :))
sigma = param(reshape([5, 5], 1, :))

elbo = []

push!(elbo, variational_objective(mu, sigma))
elbo_gradient = Tracker.gradient(variational_objective, mu, sigma)

Tracker.update!(mu, -0.0001 * elbo_gradient[1])
Tracker.update!(sigma, -0.0001 * elbo_gradient[1])

push!(elbo, variational_objective(mu, sigma))

Tracker.update!(mu, -0.0001 * elbo_gradient[1])
Tracker.update!(sigma, -0.0001 * elbo_gradient[1])

push!(elbo, variational_objective(mu, sigma))

Tracker.update!(mu, -0.0001 * elbo_gradient[1])
Tracker.update!(sigma, -0.0001 * elbo_gradient[1])

push!(elbo, variational_objective(mu, sigma))

Tracker.update!(mu, -0.0001 * elbo_gradient[1])
Tracker.update!(sigma, -0.0001 * elbo_gradient[1])

push!(elbo, variational_objective(mu, sigma))

println(elbo)