
using Distributions
using Flux.Tracker
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

function variational_objective(mu, log_std, log_density; D)
    samples = rand(Normal(), D, num_samples) .* sqrt.(log_std) .+ mu
    log_px = mapslices(log_density, samples; dims=1) # eval log(target) for all samples of params (i.e. cols)
    evidence_lower_bound = gaussian_entropy(log_std) + mean(log_px)
    return -evidence_lower_bound
end

Mu = param(reshape([-1, -1], :, 1))
Log_std = param(reshape([5, 5], :, 1))

mu = reshape([-1, -1], :, 1)
sigma = reshape([5, 5], :, 1)

test_elbo = variational_objective(mu, sigma; D=2)

elbo = variational_objective(Mu, Log_std)
elbo_gradient = Tracker.gradient(variational_objective, Mu, Log_std)[1]