# DiffOpt.jl

[DiffOpt](https://github.com/AKS1996/JuMP.jl) is a package for differentiating convex optimization program ([JuMP.jl](https://github.com/jump-dev/JuMP.jl) or [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) models) with respect to program parameters.

!!! warning
    Currently supports *linear programs*, *convex quadratic programs* and *convex conic programs* (SDP, SOCP constraints only). 

Contents
--------

```@contents
Pages = [
    "reference.md",
    "solve-LP.md",
    "solve-QP.md",
]
Depth = 1
```

## Why are Differentiable optimization problems important?
Differentiable optimization is a promising field of convex optimization and has many potential applications in game theory, control theory and machine learning (specifically deep learning - refer [this video](https://www.youtube.com/watch?v=NrcaNnEXkT8) for more). Recent work has shown how to differentiate specific subclasses of convex optimization problems. But several applications remain unexplored (refer section 8 of this [really good thesis](https://github.com/bamos/thesis)). With the help of automatic differentiation, differentiable optimization can a significant impact on creating end-to-end systems for modelling a neural network, stochastic process, or a game.
