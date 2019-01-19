# juali_advi
Automatic Differentiation Variational Inference in Julia

## Roadmap

try to recreate the following:
https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py

GPU support

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
