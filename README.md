# DiffOpt.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jump-dev.org/DiffOpt.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jump-dev.org/DiffOpt.jl/dev)
[![Build Status](https://github.com/jump-dev/DiffOpt.jl/workflows/CI/badge.svg)](https://github.com/jump-dev/DiffOpt.jl/actions)
[![Coverage](https://codecov.io/gh/jump-dev/DiffOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/DiffOpt.jl)

Differentiating convex optimization program (`JuMP.jl` or `MathOptInterface.jl` models) with respect to program parameters. Currently supports LPs, QPs.

## Installation
DiffOpt can be installed through the Julia package manager:
```
(v1.3) pkg> add https://github.com/jump-dev/DiffOpt.jl
```

## Usage

Create a differentiable model from
[existing optimizers](https://www.juliaopt.org/JuMP.jl/stable/installation/):

```julia
    using DiffOpt
    using GLPK

    diff = diff_optimizer(GLPK.Optimizer)
```

Update and solve the model:
```julia
x = MOI.add_variables(diff, 2)
c = MOI.add_constraint(diff, ...)

MOI.optimize!(diff)
```

Finally, differentiate the model (primal and dual variables specifically) to
obtain product of jacobians with respect to problem parameters and a backward
pass vector.

The optimization problem is assumed to be of the form:
```
minimize_z z^T Q z / 2 + q^T z
subject to: Az = b,
            Gz ≤ h
```

```julia
grads = backward!(diff, ["Q", "q", "h"], [1.0 1.0])
```

## Note

- This is a [NumFOCUS Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)
- Benchmarking with CVXPY or QPTH: Refer relevant examples as in [test/MOI_wrapper.jl](https://github.com/jump-dev/DiffOpt.jl/blob/master/test/MOI_wrapper.jl#L130)
