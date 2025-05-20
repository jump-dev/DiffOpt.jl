# DiffOpt.jl

[![stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://jump.dev/DiffOpt.jl/stable)
[![development docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jump.dev/DiffOpt.jl/dev)
[![Build Status](https://github.com/jump-dev/DiffOpt.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jump-dev/DiffOpt.jl/actions?query=workflow%3ACI)
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

Here is an example with a Parametric **Linear Program**:

```julia
using JuMP, DiffOpt, HiGHS

model = DiffOpt.quadratic_diff_model(HiGHS.Optimizer)
set_silent(model)

p_val = 4.0
pc_val = 2.0
@variable(model, x)
@variable(model, p in Parameter(p_val))
@variable(model, pc in Parameter(pc_val))
@constraint(model, cons, pc * x >= 3 * p)
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
@show DiffOpt.get_reverse_parameter(model, pc) == -direction_x * 3 * p_val / pc_val^2
```

Available models:
* `DiffOpt.quadratic_diff_model`: Quadratic Programs (QP) and Linear Programs
(LP)
* `DiffOpt.conic_diff_model`: Conic Programs (CP) and Linear Programs (LP)
* `DiffOpt.nonlinear_diff_model`: Nonlinear Programs (NLP), Quadratic Program
(QP) and Linear Programs (LP)
* `DiffOpt.diff_model`: Nonlinear Programs (NLP), Conic Programs (CP),
Quadratic Programs (QP) and Linear Programs (LP)


## Citing DiffOpt.jl

If you find DiffOpt.jl useful in your work, we kindly request that you cite the
following [paper](https://pubsonline.informs.org/doi/10.1287/ijoc.2022.0283):
```bibtex
@article{besancon2023diffopt,
    title={Flexible Differentiable Optimization via Model Transformations},
    author={Besançon, Mathieu and Dias Garcia, Joaquim and Legat, Beno{\^\i}t and Sharma, Akshay},
    journal={INFORMS Journal on Computing},
    year={2023},
    volume={36},
    number={2},
    pages={456--478},
    doi={10.1287/ijoc.2022.0283},
    publisher={INFORMS}
}
```
A preprint of this paper is [freely available](https://arxiv.org/abs/2206.06135).

## GSOC2020

DiffOpt began as a [NumFOCUS sponsored Google Summer of Code (2020) project](https://summerofcode.withgoogle.com/organizations/4727917315096576/?sp-page=2#5232064888045568)
