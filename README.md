# DiffOpt.jl
[![Build Status](https://travis-ci.org/AKS1996/DiffOpt.jl.svg?branch=master)](https://travis-ci.org/AKS1996/DiffOpt.jl) 
[![Coverage Status](https://coveralls.io/repos/github/AKS1996/DiffOpt.jl/badge.svg?branch=master)](https://coveralls.io/github/AKS1996/DiffOpt.jl?branch=master)
[![AppVeyor Status](https://ci.appveyor.com/api/projects/status/github/AKS1996/DiffOpt.jl?branch=master&svg=true)](https://ci.appveyor.com/project/AKS1996/diffopt-jl)
[![][docs-dev-img]][docs-dev-url]


Differentiating convex optimization program (`JuMP.jl` or `MathOptInterface.jl` models) with respect to program parameters. Currently supports LPs, QPs.

## Installation
DiffOpt can be installed through the Julia package manager:
```
(v1.3) pkg> add https://github.com/jump-dev/DiffOpt.jl
```

## Usage
Create a differentiable model from [existing optimizers](https://www.juliaopt.org/JuMP.jl/stable/installation/)
```julia
    using DiffOpt
    using GLPK
    
    diff = diff_optimizer(GLPK.Optimizer)
```
Update and solve the model 
```julia
    x = MOI.add_variables(diff, 2)
    c = MOI.add_constraint(diff, ...)
    
    MOI.optimize!(diff)
```
Finally differentiate the model (primal and dual variables specifically) to obtain product of jacobians with respect to problem parameters and a backward pass vector
```julia
    grads = backward!(diff, ["Q", "q", "h"], [1.0 1.0])
```

## Note
- Package developed using [PkgTemplates](https://github.com/invenia/PkgTemplates.jl)
- This is a [NumFOCUS Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)
- Benchmarking with CVXPY or QPTH: Refer relevant examples as in [test/MOI_wrapper.jl](https://github.com/jump-dev/DiffOpt.jl/blob/master/test/MOI_wrapper.jl#L130)


[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://aks1996.github.io/DiffOpt.jl/dev/