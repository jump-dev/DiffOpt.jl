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

3. To differentiate a general nonlinear program, we can use the `forward_differentiate!` method with perturbations in the objective function and constraints through perturbations in the problem parameters. For example, consider the following nonlinear program:
```julia
model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
@variable(model, p ∈ MOI.Parameter(0.1))
@variable(model, x >= p)
@variable(model, y >= 0)
@objective(model, Min, x^2 + y^2)
@constraint(model, con, x + y >= 1)

# Solve
JuMP.optimize!(model)

# Set parameter pertubations
MOI.set(model, DiffOpt.ForwardParameter(), params[1], 0.2)

# forward differentiate
DiffOpt.forward_differentiate!(model)

# Retrieve sensitivities
dx = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x)
dy = MOI.get(model, DiffOpt.ForwardVariablePrimal(), y)
```

or we can use the `reverse_differentiate!` method:
```julia
# Set Primal Pertubations
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)

# Reverse differentiation
DiffOpt.reverse_differentiate!(model)

# Retrieve reverse sensitivities (example usage)
dp= MOI.get(model, DiffOpt.ReverseParameter(), p)
```