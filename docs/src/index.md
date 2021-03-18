# DiffOpt.jl
[![Build Status](https://travis-ci.org/AKS1996/DiffOpt.jl.svg?branch=master)](https://travis-ci.org/AKS1996/DiffOpt.jl) 
[![Coverage Status](https://coveralls.io/repos/github/AKS1996/DiffOpt.jl/badge.svg?branch=master)](https://coveralls.io/github/AKS1996/DiffOpt.jl?branch=master)
[![AppVeyor Status](https://ci.appveyor.com/api/projects/status/github/AKS1996/DiffOpt.jl?branch=master&svg=true)](https://ci.appveyor.com/project/AKS1996/diffopt-jl)
[![Docs status](https://img.shields.io/badge/docs-dev-blue.svg)](https://aks1996.github.io/DiffOpt.jl/dev/)

[DiffOpt](https://github.com/AKS1996/JuMP.jl) is a package for differentiating convex optimization program ([JuMP.jl](https://github.com/jump-dev/JuMP.jl) or [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) models) with respect to program parameters. Note that this package does not contains any solver. This package has two major backends, available via `backward_quad` and `backward_conic` methods, to differentiate models with optimal solutions.

!!! note
    Currently supports *linear programs*, *convex quadratic programs* and *convex conic programs* (SDP, SOCP constraints only). 


## Installation
DiffOpt can be installed through the Julia package manager:
```
(v1.3) pkg> add https://github.com/jump-dev/DiffOpt.jl
```

## Why are Differentiable optimization problems important?
Differentiable optimization is a promising field of convex optimization and has many potential applications in game theory, control theory and machine learning (specifically deep learning - refer [this video](https://www.youtube.com/watch?v=NrcaNnEXkT8) for more). Recent work has shown how to differentiate specific subclasses of convex optimization problems. But several applications remain unexplored (refer section 8 of this [really good thesis](https://github.com/bamos/thesis)). With the help of automatic differentiation, differentiable optimization can a significant impact on creating end-to-end systems for modelling a neural network, stochastic process, or a game.


## Contributing
Contributions to this package are more than welcome, if you find a bug or have any suggestions for the documentation please post it on the [github issue tracker](https://github.com/jump-dev/DiffOpt.jl/issues).

When contributing please note that the package follows the [JuMP style guide](https://jump.dev/JuMP.jl/stable/style/)