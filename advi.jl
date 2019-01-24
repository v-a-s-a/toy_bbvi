
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

#plot(contour(x, y, Z))

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
    mu = reshape(parameters[1:D], 1, :)
    log_std = reshape(parameters[D+1:end], 1, :)
    samples = rand(Normal(), num_samples, D) .* exp.(log_std) .+ mu
    log_px = mapslices(log_density, samples; dims=2) # eval log(target) for all samples of params (i.e. cols)
    elbo = gaussian_entropy(log_std) + mean(log_px)
    return -elbo
end


mu_init = [-1, -1]
sigma_init = [-5, -5]

parameters = Flux.Tracker.param(cat(mu_init, sigma_init, dims=1))
elbo_gradient = Flux.Tracker.gradient(variational_objective, parameters)

steps = 2000
elbo = Array{Float64}(undef, steps)

mu_trajectory = Array{Float64}(undef, steps, D)
sigma_trajectory = Array{Float64}(undef, steps, D)

opt = ADAM(0.1)
for i in 1:steps
    println(i)
    Δ = Flux.Optimise.update!(opt, parameters, Flux.data(elbo_gradient[1]))
    Flux.Tracker.update!(parameters, Δ)
    elbo[i] = variational_objective(parameters.data)
    mu_trajectory[i, :] = parameters[1:D].data
    sigma_trajectory[i, :] = parameters[D+1:end].data
end

q = MultivariateNormal(mu[1,:].data, Diagonal(exp.(2*sigma[1,:].data)))
Z_q = Array{Float64}(undef, size(X))
for i in 1:size(X)[1]
    for j in 1:size(X)[2]
        Z_q[i, j] = pdf(q, [X[i, j], Y[i, j]])
    end
end

pyplot()
contour(x, y, Z)
contour!(x, y, Z_q)
