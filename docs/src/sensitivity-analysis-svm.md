# Sensitivity Analysis of SVM using DiffOpt.jl

This notebook illustrates sensitivity analysis of data points in an [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine) (inspired from [@matbesancon](http://github.com/matbesancon)'s [SimpleSVMs](http://github.com/matbesancon/SimpleSVMs.jl).)

For reference, Section 10.1 of https://online.stat.psu.edu/stat508/book/export/html/792 gives an intuitive explanation of what does it means to have a sensitive hyperplane or data point. The general form of SVM is given below (without regularization):

```math
\begin{split}
\begin{array} {ll}
\mbox{minimize} & \sum_{i=1}^{N} \xi_{i} \\
\mbox{s.t.} & \xi_{i} \ge 0 \quad i=1..N  \\
            & y_{i} (w^T X_{i} + b) \ge 1 - \xi[i]\\
\end{array}
\end{split}
```
where
- `X`, `y` are the `N` data points
- $\xi$ is soft-margin loss

## Define and solve the SVM

Import the libraries.
```julia
import Random
using Test
import SCS
import Plots
using DiffOpt
using LinearAlgebra
using MathOptInterface

const MOI = MathOptInterface;
```

Construct separatable, non-trivial data points.
```julia
N = 50
D = 2
Random.seed!(rand(1:100))
X = vcat(randn(N, D), randn(N,D) .+ [4.0,1.5]')
y = append!(ones(N), -ones(N));
```

Let's define the variables.
```julia
(nobs, nfeat) = size(X)

model = diff_optimizer(SCS.Optimizer) 

# add variables
l = MOI.add_variables(model, nobs)
w = MOI.add_variables(model, nfeat)
b = MOI.add_variable(model)
```

Add the constraints.
```julia
MOI.add_constraint(
    model,
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:nobs, MOI.ScalarAffineTerm.(1.0, l)), zeros(nobs)
    ), 
    MOI.Nonnegatives(nobs)
)

# define the whole matrix Ax, it'll be easier then
# refer https://discourse.julialang.org/t/solve-minimization-problem-where-constraint-is-the-system-of-linear-inequation-with-mathoptinterface-efficiently/23571/4
Ax = Array{MOI.ScalarAffineTerm{Float64}}(undef, nobs, nfeat+2)
for i in 1:nobs
    Ax[i, :] = MOI.ScalarAffineTerm.([1.0; y[i]*X[i,:]; y[i]], [l[i]; w; b])
end
terms = MOI.VectorAffineTerm.(1:nobs, Ax)
f = MOI.VectorAffineFunction(
    vec(terms),
    -ones(nobs)
)
MOI.add_constraint(
    model,
    f,
    MOI.Nonnegatives(nobs)
)
```

Define the linear objective function and solve the SVM model.
```julia
objective_function = MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.(ones(nobs), l),
                        0.0
                    )
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

MOI.optimize!(model);

loss = MOI.get(model, MOI.ObjectiveValue())
wv = MOI.get(model, MOI.VariablePrimal(), w)
bv = MOI.get(model, MOI.VariablePrimal(), b)
```


We can visualize the separating hyperplane. 
```julia
p = Plots.scatter(X[:,1], X[:,2], color = [yi > 0 ? :red : :blue for yi in y], label = "")
Plots.yaxis!(p, (-2, 4.5))
Plots.plot!(p, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
```
    
![svg](sensitivity-analysis-svm-img-1.svg)
    


## Experiment 1: Changing the labels
Let's change data point labels `y` without changing data points themselves `X`. Construct the perturbations.
```julia
ðA = zeros(2*nobs, nobs+nfeat+1)
ðb = zeros(2*nobs)
ðc = zeros(nobs+nfeat+1); # c = sum(`l`) + 0'w + 0.b

∇ = []

# begin differentiating
for Xi in 1:nobs
    ðA[nobs+Xi, nobs+nfeat+1] = 1.0
    
    dx, dy, ds = backward(model, ðA, ðb, ðc)
    dl, dw, db = dx[1:nobs], dx[nobs+1:nobs+1+nfeat], dx[nobs+1+nfeat]
    push!(∇, norm(dw)+norm(db))
    
    ðA[nobs+Xi, nobs+nfeat+1] = 0.0
end
∇ = normalize(∇);
```

Visualize point sensitvities with respect to separating hyperplane. Note that the gradients are normalized.
```julia
p2 = Plots.scatter(
    X[:,1], X[:,2], 
    color = [yi > 0 ? :red : :blue for yi in y], label = "",
    markersize = ∇ * 20
)
Plots.yaxis!(p2, (-2, 4.5))
Plots.plot!(p2, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
```

    
![svg](sensitivity-analysis-svm-img-2.svg)



## Experiment 2: Changing the data points

Similar to previous example, we can change labels `y` and data points `X` too.
```julia
# constructing perturbations
ðA = zeros(2*nobs, nobs+nfeat+1)
ðb = zeros(2*nobs)
ðc = zeros(nobs+nfeat+1); # c = sum(`l`) + 0'w + 0.b

∇ = []

# begin differentiating
for Xi in 1:nobs
    ðA[nobs+Xi, nobs.+(1:nfeat+1)] = ones(3)
    
    dx, dy, ds = backward(model, ðA, ðb, ðc)
    dl, dw, db = dx[1:nobs], dx[nobs+1:nobs+1+nfeat], dx[nobs+1+nfeat]
    push!(∇, norm(dw)+norm(db))
    
    ðA[nobs+Xi, nobs.+(1:nfeat+1)] = zeros(3)
end
∇ = normalize(∇);
```

We can visualize point sensitvity with respect to the separating hyperplane. Note that the gradients are normalized.
```julia
p3 = Plots.scatter(
    X[:,1], X[:,2], 
    color = [yi > 0 ? :red : :blue for yi in y], label = "",
    markersize = ∇ * 20
)
Plots.yaxis!(p3, (-2, 4.5))
Plots.plot!(p3, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
```
    
![svg](sensitivity-analysis-svm-img-3.svg)
    