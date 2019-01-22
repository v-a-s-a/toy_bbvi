# juali_advi
Automatic Differentiation Variational Inference in Julia

## Roadmap

Recreating "Black-Box Stochastic Variational Inference in Five Lines of Python"
See: https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py

Creating the toy example. A small gif of maximizing the evidence lower-bound.

Move onto a data based problem. Maybe linear/logistic regression or a bayesnet.

Batched inference and stochastic variational inference.

GPU support.

## Automatic Differentiation

There are a few automatic differentiation libs available. The discussion in Turing.jl is worth looking into:

https://github.com/TuringLang/Turing.jl/issues/511

In short, Distributions.jl may not play nicely with various libs.

Libs:
  - Zygote.jl
  - Flux.Tracker
  - Turings Autodiff


## GPU support

Distributions.jl also may not play nicely with GPUs!
