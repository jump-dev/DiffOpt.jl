# DiffOpt.jl

[![stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://jump.dev/DiffOpt.jl/stable)
[![development docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jump.dev/DiffOpt.jl/dev)
[![Build Status](https://github.com/jump-dev/DiffOpt.jl/workflows/CI/badge.svg?branch=master)](https://github.com/jump-dev/DiffOpt.jl/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/gh/jump-dev/DiffOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/DiffOpt.jl)

[DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) is a package for
differentiating convex optimization programs with respect to the program
parameters. DiffOpt currently supports linear, quadratic, and conic programs.

## License

`DiffOpt.jl` is licensed under the
[MIT License](https://github.com/jump-dev/DiffOpt.jl/blob/master/LICENSE.md).

## Installation

Install DiffOpt using `Pkg.add`:

```julia
import Pkg
Pkg.add("DiffOpt")
```

## Documentation

The [documentation for DiffOpt.jl](https://jump.dev/DiffOpt.jl/stable/)
includes a detailed description of the theory behind the package, along with
examples, tutorials, and an API reference.

## Use with JuMP

Use DiffOpt with JuMP by following this brief example:

```julia
using JuMP, DiffOpt, HiGHS
# Create a model using the wrapper
model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer))
# Define your model and solve it
@variable(model, x)
@constraint(model, cons, x >= 3)
@objective(model, Min, 2x)
optimize!(model)
# Choose the problem parameters to differentiate with respect to, and set their
# perturbations.
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)
# Differentiate the model
DiffOpt.reverse_differentiate!(model)
# fetch the gradients
grad_exp = MOI.get(model, DiffOpt.ReverseConstraintFunction(), cons)  # -3 x - 1
constant(grad_exp)        # -1
coefficient(grad_exp, x)  # -3
```

## GSOC2020

DiffOpt began as a [NumFOCUS sponsored Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)
