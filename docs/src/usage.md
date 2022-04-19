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
grad_obj = MOI.get(model, DiffOpt.BackwardOutObjective())
grad_con = MOI.get.(model, DiffOpt.BackwardOutConstraint(), c)
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
MOI.set(model, DiffOpt.ForwardInObjective(), ones(2) ⋅ x)
DiffOpt.forward_differentiate!(model)
grad_x = MOI.get.(model, DiffOpt.ForwardOutVariablePrimal(), x)
```
