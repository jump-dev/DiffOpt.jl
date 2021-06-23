# DiffOpt.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jump.dev/DiffOpt.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jump.dev/DiffOpt.jl/dev)
[![Build Status](https://github.com/jump-dev/DiffOpt.jl/workflows/CI/badge.svg?branch=master)](https://github.com/jump-dev/DiffOpt.jl/actions?query=workflow%3ACI)
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
using MathOptInterface
const MOI = MathOptInterface

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

Currently, DiffOpt supports two backends. If the optimization problem is of quadratic form i.e.
```
minimize_z z^T Q z / 2 + q^T z
subject to: Az = b,
            Gz ≤ h
```
then one can compute gradients by providing a backward pass vector
```julia
bpv = [1.0, 1.0]
grads = backward(diff, ["Q", "q", "h"], bpv)
```

Secondly, for a conic problem of the format:
```
minimize_x c^T x
subject to: Ax + b in K
```
where
- the objective is linear
- `K` is a Cartesian product of linear, semidefinite, second-order cones
then one can compute gradients by providing perturbations
```julia
grads = backward(diff, dA, db, dc)
```

## Note

- This is a [NumFOCUS Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)
- Benchmarking with CVXPY or QPTH: Refer relevant examples as in [test/MOI_wrapper.jl](https://github.com/jump-dev/DiffOpt.jl/blob/master/test/MOI_wrapper.jl#L130)
