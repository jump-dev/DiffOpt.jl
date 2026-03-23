# DiffOpt.jl

[DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) is a package for differentiating convex and non-convex optimization program ([JuMP.jl](https://github.com/jump-dev/JuMP.jl) or [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) models) with respect to program parameters. Note that this package does not contain any solver.
This package has two major backends, available via the `reverse_differentiate!` and `forward_differentiate!` methods, to differentiate models (quadratic or conic) with optimal solutions.

!!! note
    Currently supports *linear programs* (LP), *convex quadratic programs* (QP), *convex conic programs* (SDP, SOCP, exponential cone constraints only), and *general nonlinear programs* (NLP).


## Installation

DiffOpt can be installed through the Julia package manager:
```
(v1.3) pkg> add https://github.com/jump-dev/DiffOpt.jl
```

## Why are Differentiable optimization problems important?

Differentiable optimization is a promising field of constrained optimization and has many potential applications in game theory, control theory and machine learning (specifically deep learning - refer [this video](https://www.youtube.com/watch?v=NrcaNnEXkT8) for more).
Recent work has shown how to differentiate specific subclasses of constrained optimization problems. But several applications remain unexplored (refer section 8 of this [really good thesis](https://github.com/bamos/thesis)). With the help of automatic differentiation, differentiable optimization can have a significant impact on creating end-to-end differentiable systems to model neural networks, stochastic processes, or a game.


## Contributing

Contributions to this package are more than welcome, if you find a bug or have any suggestions for the documentation please post it on the [github issue tracker](https://github.com/jump-dev/DiffOpt.jl/issues).

When contributing please note that the package follows the [JuMP style guide](https://jump.dev/JuMP.jl/stable/style/)
