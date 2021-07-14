# Sensitivity Analysis of Ridge Regression using DiffOpt.jl

This example illustrates the sensitivity analysis of data points in a [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression) problem. The general form of the problem is given below:

```math
\begin{split}
\begin{array} {ll}
\mbox{minimize} & \sum_{i=1}^{N} (y_{i} - w x_{i} - b)^2 + \alpha (w^2 + b^2) \\
\end{array}
\end{split}
```
where
- `w`, `b` are slope and intercept of the regressing line
- `x`, `y` are the N data points
- `α` is the regularization constant

## Define and solve the problem
```@example 2
import Random
import OSQP
import Plots
using DiffOpt
using JuMP
using LinearAlgebra
```

Construct a set of noisy (guassian) data points around a line.
```@example 2
function create_problem(N=100)
    m = 2*abs(randn())
    b = rand()
    X = randn(N)
    Y = m*X .+ b + 0.8*randn(N)
    
    return X, Y
end

X, Y = create_problem();
nothing # hide
```

The helper method `fitRidge` defines and solves the corresponding model.
```@example 2
function fitRidge(X,Y,alpha=0.1)
    model = Model(() -> diff_optimizer(OSQP.Optimizer))

    # add variables
    @variable(model, w)
    @variable(model, b)
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    @objective(
        model,
        Min,
        sum((Y - w*X .- b).*(Y - w*X .- b)) + alpha*(sum(w*w)+sum(b*b)),
    )

    optimize!(model)

    loss = objective_value(model)
    return model, w, b, loss, value(w), value(b)
end
nothing # hide
```

Train on the data generated.
```@example 2
model, w, b, loss_train, ŵ, b̂ = fitRidge(X, Y)
nothing # hide
```

We can visualize the approximating line. 
```@example 2
p = Plots.scatter(X, Y, label="")
mi, ma = minimum(X), maximum(X)
Plots.plot!(p, [mi, ma], [mi*ŵ+b̂, ma*ŵ+b̂], color=:red, label="")
nothing # hide
``` 
    

## Differentiate
Now that we've solved the problem, we can compute the sensitivity of optimal values -- the approximating line components `w`, `b` in this case -- with respect to perturbations in the data points (`x`,`y`).
```@example 2
∇ = zero(X)

for i in 1:length(X)
    MOI.set(
        model, 
        DiffOpt.ForwardInObjective(), 
        MOI.ScalarQuadraticFunction(
            [MOI.ScalarAffineTerm(-2(Y[1] + X[1]), w.index)], 
            [MOI.ScalarQuadraticTerm(2X[1], w.index, w.index)], 
            0.0
        )
    )
    
    DiffOpt.forward(model)

    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(), 
        b
    )

    ∇[i] = db
end
normalize!(∇);
nothing # hide
```

Visualize point sensitivities with respect to regressing line. Note that the gradients are normalized.
```@example 2
p = Plots.scatter(
    X, Y,
    color=[x>0 ? :red : :blue for x in ∇],
    markersize=[25*abs(x) for x in ∇],
    label=""
)
mi, ma = minimum(X), maximum(X)
Plots.plot!(p, [mi, ma], [mi*ŵ+b̂, ma*ŵ+b̂], color=:red, label="")
nothing # hide
``` 
