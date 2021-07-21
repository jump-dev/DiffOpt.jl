# DiffOpt.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jump.dev/DiffOpt.jl/dev)
[![Build Status](https://github.com/jump-dev/DiffOpt.jl/workflows/CI/badge.svg?branch=master)](https://github.com/jump-dev/DiffOpt.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/jump-dev/DiffOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/DiffOpt.jl)


DiffOpt is a package for differentiating convex optimization programs with respect to the program parameters. It currently supports linear, quadratic and conic programs. Refer to [the  documentation](https://jump.dev/DiffOpt.jl/dev) for examples. Powered by [JuMP.jl](https://jump.dev/DiffOpt.jl/dev), DiffOpt allows creating a differentiable optimization model from many
[existing optimizers](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers).


## Installation
DiffOpt can be installed via the Julia package manager:

```
(v1.3) pkg> add https://github.com/jump-dev/DiffOpt.jl
```

## Example

1. Create a model using the wrapper.
```julia
using JuMP, DiffOpt, Clp

model = JuMP.Model(() -> diff_optimizer(Clp.Optimizer))
```

2. Define your model and solve it a single line.
```julia
@variable(model, x)
@constraint(
  model, 
  cons, 
  x >= 3,
)
@objective(
  model, 
  Min, 
  2x,
)

optimize!(model) # solve
```

3. Choose the problem parameters to differentiate with and set their perturbations.
```julia
MOI.set.(  # set pertubations / gradient inputs
    model, 
    DiffOpt.BackwardInVariablePrimal(), 
    x, 
    1.0,
)
```

4. Differentiate the model (primal, dual variables specifically) and fetch the gradients
```julia
DiffOpt.backward(model) # differentiate

grad_exp = MOI.get(   # -3x+1
    model, 
    DiffOpt.BackwardOutConstraint(), 
    cons
)
JuMP.constant(grad_exp)  # 1
JuMP.coefficient(grad_exp, x)  # -3
```

<!-- Currently, DiffOpt supports two backends. If the optimization problem is of quadratic form i.e.
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
``` -->


## Note

- DiffOpt began as a [NumFOCUS sponsored Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)