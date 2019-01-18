
using Distributions

function lower_bound(Mu, Sigma, logprob_func, D, num_samples) 

    d =  rand(Normal(), 100, 1000) * sqrt(Sigma) + Mu

    elbo = 

end

