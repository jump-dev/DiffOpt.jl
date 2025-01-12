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

### DiffOpt-JuMP API with `Parameters`

```julia
using JuMP, DiffOpt, HiGHS

# model = Model(
#     () -> DiffOpt.diff_optimizer(
#         HiGHS.Optimizer;
#         with_parametric_opt_interface = true,
#     ),
# )
model = DiffOpt.diff_model(
    HiGHS.Optimizer;
    with_parametric_opt_interface = true,
)
set_silent(model)

p_val = 4.0
pc_val = 2.0
@variable(model, x)
@variable(model, p in Parameter(p_val))
@variable(model, pc in Parameter(pc_val))
@constraint(model, cons, pc * x >= 3 * p) #??? InvalidConstraintRef TODO
@objective(model, Min, 2x)
optimize!(model)
@show value(x) == 3 * p_val / pc_val

# the function is
# x(p, pc) = 3p / pc
# hence,
# dx/dp = 3 / pc
# dx/dpc = -3p / pc^2

# First, try forward mode AD

# differentiate w.r.t. p
direction_p = 3.0
DiffOpt.set_forward_parameter(model, p, direction_p)
DiffOpt.forward_differentiate!(model)
@show DiffOpt.get_forward_variable(model, x) == direction_p * 3 / pc_val

# update p and pc
p_val = 2.0
pc_val = 6.0
set_parameter_value(p, p_val)
set_parameter_value(pc, pc_val)
# re-optimize
optimize!(model)
# check solution
@show value(x) ≈ 3 * p_val / pc_val

# stop differentiating with respect to p
DiffOpt.empty_input_sensitivities!(model)
# differentiate w.r.t. pc
direction_pc = 10.0
DiffOpt.set_forward_parameter(model, pc, direction_pc)
DiffOpt.forward_differentiate!(model)
@show abs(DiffOpt.get_forward_variable(model, x) -
    -direction_pc * 3 * p_val / pc_val^2) < 1e-5

# always a good practice to clear previously set sensitivities
DiffOpt.empty_input_sensitivities!(model)
# Now, reverse model AD
direction_x = 10.0
DiffOpt.set_reverse_variable(model, x, direction_x)
DiffOpt.reverse_differentiate!(model)
@show DiffOpt.get_reverse_parameter(model, p) == direction_x * 3 / pc_val
@show abs(DiffOpt.get_reverse_parameter(model, pc) -
    -direction_x * 3 * p_val / pc_val^2) < 1e-5
```

### Low level DiffOpt-JuMP API:

A brief example:

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
