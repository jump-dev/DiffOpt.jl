# Usage

Create a differentiable model from [existing optimizers](https://www.juliaopt.org/JuMP.jl/stable/installation/)
```julia
using JuMP
import DiffOpt
import SCS

model = DiffOpt.diff_optimizer(SCS.Optimizer)
```

Update and solve the model 
```julia
x = MOI.add_variables(model, 2)
c = MOI.add_constraint(model, ...)

MOI.optimize!(model)
```

Finally differentiate the model (primal and dual variables specifically) to obtain product of jacobians with respect to problem parameters and a backward pass vector. Currently `DiffOpt` supports two backends for differentiating a model:

1. To differentiate Convex Quadratic Program

```math
\begin{align*}
& \min_{x \in \mathbb{R}^n} & \frac{1}{2} x^T Q x + q^T x  & \\
& \text{s.t.}               & A x = b        \qquad        & b \in \mathbb{R}^m \\
&                           & G x \leq h     \qquad        & h \in \mathbb{R}^p
\end{align*}
```

we can use the `reverse_differentiate!` method
```julia
MOI.set.(model,
    DiffOpt.ReverseVariablePrimal(), x, ones(2))
DiffOpt.reverse_differentiate!(model)
grad_obj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
grad_con = MOI.get.(model, DiffOpt.ReverseConstraintFunction(), c)
```

2. To differentiate convex conic program

```math
\begin{align*}
& \min_{x \in \mathbb{R}^n} & c^T x \\
& \text{s.t.}               & A x + s = b  \\
&                           & b \in \mathbb{R}^m  \\
&                           & s \in \mathcal{K}
\end{align*}
```

we can use the `forward_differentiate!` method with perturbations in matrices `A`, `b`, `c`:
```julia
import LinearAlgebra: ⋅
MOI.set(model, DiffOpt.ForwardObjectiveFunction(), ones(2) ⋅ x)
DiffOpt.forward_differentiate!(model)
grad_x = MOI.get.(model, DiffOpt.ForwardVariablePrimal(), x)
```

3. To differentiate a general nonlinear program, have to use the API for Parameterized JuMP models. For example, consider the following nonlinear program:

```julia
using JuMP, DiffOpt, HiGHS

model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model)

p_val = 4.0
pc_val = 2.0
@variable(model, x)
@variable(model, p in Parameter(p_val))
@variable(model, pc in Parameter(pc_val))
@constraint(model, cons, pc * x >= 3 * p)
@objective(model, Min, x^4)
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
MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), Parameter(direction_p))
DiffOpt.forward_differentiate!(model)
@show MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) == direction_p * 3 / pc_val

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
MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(pc), Parameter(direction_pc))
DiffOpt.forward_differentiate!(model)
@show abs(MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) -
    -direction_pc * 3 * p_val / pc_val^2) < 1e-5

# always a good practice to clear previously set sensitivities
DiffOpt.empty_input_sensitivities!(model)
# Now, reverse model AD
direction_x = 10.0
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_x)
DiffOpt.reverse_differentiate!(model)
@show MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) == MOI.Parameter(direction_x * 3 / pc_val)
@show abs(MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(pc)).value -
    -direction_x * 3 * p_val / pc_val^2) < 1e-5
```

## Calculating objective sensitivity with respect to parameters (currently only supported for Nonlinear Programs)

Consider a differentiable model with parameters `p` and `pc` as in the previous example:

```julia
using JuMP, DiffOpt, HiGHS

model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model)

p_val = 4.0
pc_val = 2.0
@variable(model, x)
@variable(model, p in Parameter(p_val))
@variable(model, pc in Parameter(pc_val))
@constraint(model, cons, pc * x >= 3 * p)
@objective(model, Min, x^4)
optimize!(model)

direction_p = 3.0
MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), Parameter(direction_p))
DiffOpt.forward_differentiate!(model)

```

Using Lagrangian duality we could already calculate the objective sensitivity with respect to parameters that appear in the RHS of the constraints (e.g, `cons` in this case for parameter `p`) - i.e. The objective sensitivity w.r.t. a parameter change in the RHS of the constraints is given by the optimal multiplier.

On the other hand, if the parameter appears in the LHS of the constraints, we can calculate the objective sensitivity with respect to the parameter using: the sensitivities of the variables with respect to the parameter, \( \frac{\partial x}{\partial p} \), and the gradient of the objective with respect to the variables \( \frac{\partial f}{\partial x} \):

```math
\frac{\partial f}{\partial p} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial p}
```
 - A consequence of the chain-rule.

In order to calculate the objective perturbation with respect to the parameter perturbation vector, we can use the following code:

```julia
# Always a good practice to clear previously set sensitivities
DiffOpt.empty_input_sensitivities!(model)

MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), Parameter(3.0))
MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p_c), Parameter(3.0))
DiffOpt.forward_differentiate!(model)

MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())
```

In the backward mode, we can calculate the parameter perturbation with respect to the objective perturbation:

```julia
# Always a good practice to clear previously set sensitivities
DiffOpt.empty_input_sensitivities!(model)

MOI.set(
    model,
    DiffOpt.ReverseObjectiveSensitivity(),
    0.1,
)

DiffOpt.reverse_differentiate!(model)

MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p))
```

It is important to note that the (reverse) parameter perturbation given an objective perturbation is somewhat equivalent to the perturbation with respect to solution (since one can be calculated from the other). Therefore, one cannot set both the objective sensitivity (`DiffOpt.ReverseObjectiveSensitivity`) and the solution sensitivity (e.g. `DiffOpt.ReverseVariablePrimal`) at the same time - the code will throw an error if you try to do so.
```
