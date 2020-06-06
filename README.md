# DiffOpt.jl
[![Build Status](https://travis-ci.org/AKS1996/DiffOpt.jl.svg?branch=master)](https://travis-ci.org/AKS1996/DiffOpt.jl) 
[![Coverage Status](https://coveralls.io/repos/github/AKS1996/DiffOpt.jl/badge.svg?branch=master)](https://coveralls.io/github/AKS1996/DiffOpt.jl?branch=master)


Differentiating convex optimization program (`JuMP.jl` or `MathOptInterface.jl` models) with respect to program parameters. Currently supports LPs, QPs.

## Installation
DiffOpt can be installed through the Julia package manager:
```
(v1.3) pkg> add https://github.com/AKS1996/DiffOpt.jl
```

## Usage
Create a differentiable model from an existing `MathOptInterface.jl` model
```julia
    using DiffOpt
    
    ...
    
    diff = diff_model(model)
```
Solve the model with any of the [existing optimizers](https://www.juliaopt.org/JuMP.jl/stable/installation/)
```julia
    ẑ = diff.forward()
```
Finally differentiate the model (primal and dual variables specifically) to obtain their jacobians with respect to problem data
```julia
    grads = diff.backward(["Q", "q", "h"])
```

## Note
- Package developed using [PkgTemplates](https://github.com/invenia/PkgTemplates.jl)
- This is a [NumFOCUS Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)
