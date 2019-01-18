# juali_advi
Automatic Differentiation Variational Inference in Julia

## Roadmap

try to recreate the following:
https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py

## Notes

There are a few automatic differentiation libs available. The discussion in Turing.jl is worth looking into:

https://github.com/TuringLang/Turing.jl/issues/511

In general, I'm leaning toward using Zygote.jl, unless it performs really poorly for the simplest use case.
