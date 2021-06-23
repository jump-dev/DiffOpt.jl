# Usage

Create a differentiable model from [existing optimizers](https://www.juliaopt.org/JuMP.jl/stable/installation/)
```julia
    using DiffOpt
    using SCS
    
    model = diff_optimizer(SCS.Optimizer)
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

we can use the `backward` method
```julia
    MOI.set.(model,
        DiffOpt.BackwardInVariablePrimal(), x, ones(2))
    DiffOpt.backward(model)
    grad_obj = MOI.get.(model, DiffOpt.BackwardOut{DiffOpt.LinearObjective}(), x)
    grad_rhs = MOI.get.(model, DiffOpt.BackwardOut{DiffOpt.ConstraintConstant}(), c)
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

we can use the `forward` method with perturbations in matrices `A`, `b`, `c`
```julia
using LinearAlgebra # for `⋅`
MOI.set(model, DiffOpt.ForwardInObjective(), ones(2) ⋅ MOI.SingleVariable.(x))
DiffOpt.forward(model)
grad_x = MOI.get.(model, DiffOpt.ForwardOutVariablePrimal(), x)
```
