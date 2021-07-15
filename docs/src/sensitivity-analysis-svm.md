# Sensitivity Analysis of SVM using DiffOpt.jl

This notebook illustrates sensitivity analysis of data points in an [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine) (inspired from [@matbesancon](http://github.com/matbesancon)'s [SimpleSVMs](http://github.com/matbesancon/SimpleSVMs.jl).)

For reference, Section 10.1 of https://online.stat.psu.edu/stat508/book/export/html/792 gives an intuitive explanation of what does it means to have a sensitive hyperplane or data point. The general form of the SVM training problem is given below (without regularization):

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
- `ξ` is the soft-margin loss.

## Define and solve the SVM

Import the libraries.

```@example 1
import Random
import SCS
import Plots
using DiffOpt
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
nothing # hide
```

Construct separatable, non-trivial data points.
```@example 1
N = 100
D = 2
Random.seed!(62)
X = vcat(randn(N÷2, D), randn(N÷2,D) .+ [4.0,1.5]')
y = append!(ones(N÷2), -ones(N÷2))
nothing # hide
```

Let's define the variables.
```@example 1
model = diff_optimizer(SCS.Optimizer)
MOI.set(model, MOI.Silent(), true)

# add variables
l = MOI.add_variables(model, N)
w = MOI.add_variables(model, D)
b = MOI.add_variable(model)
nothing # hide
```

Add the constraints.
```@example 1
MOI.add_constraint(
    model,
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:N, MOI.ScalarAffineTerm.(1.0, l)), zeros(N),
    ),
    MOI.Nonnegatives(N),
)

# define the whole matrix Ax, it'll be easier then
# refer https://discourse.julialang.org/t/solve-minimization-problem-where-constraint-is-the-system-of-linear-inequation-with-mathoptinterface-efficiently/23571/4
Ax = Matrix{MOI.ScalarAffineTerm{Float64}}(undef, N, D+2)
for i in 1:N
    Ax[i, :] = MOI.ScalarAffineTerm.([1.0; y[i]*X[i,:]; y[i]], [l[i]; w; b])
end
terms = MOI.VectorAffineTerm.(1:N, Ax)
f = MOI.VectorAffineFunction(
    vec(terms),
    -ones(N),
)
cons = MOI.add_constraint(
    model,
    f,
    MOI.Nonnegatives(N),
)
nothing # hide
```

Define the linear objective function and solve the SVM model.
```@example 1
objective_function = MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.(ones(N), l),
                        0.0,
                    )
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

MOI.optimize!(model)

loss = MOI.get(model, MOI.ObjectiveValue())
wv = MOI.get(model, MOI.VariablePrimal(), w)
bv = MOI.get(model, MOI.VariablePrimal(), b)
nothing # hide
```

We can visualize the separating hyperplane.

```@example 1
# build SVM points
svm_x = [0.0, 3.0]
svm_y = (-bv .- wv[1] * svm_x )/wv[2]

p = Plots.scatter(X[:,1], X[:,2], color = [yi > 0 ? :red : :blue for yi in y], label = "")
Plots.yaxis!(p, (-2, 4.5))
Plots.plot!(p, svm_x, svm_y, label = "loss = $(round(loss, digits=2))", width=3)
```


# Experiments
Now that we've solved the SVM, we can compute the sensitivity of optimal values -- the separating hyperplane in our case -- with respect to perturbations of the problem data -- the data points -- using DiffOpt. For illustration, we've explored two questions:

- How does a change in labels of the data points (`y=1` to `y=-1`, and vice versa) affect the position of the hyperplane? This is achieved by finding the gradient of `w`, `b` with respect to `y[i]`, the classification label of the ith data point.
- How does a change in coordinates of the data points, `X`, affects the position of the hyperplane? This is achieved by finding gradient of `w`, `b` with respect to `X[i]`, 2D coordinates of the data points.

Note that finding the optimal SVM can be modelled as a conic optimization problem:

```math
\begin{align*}
& \min_{x \in \mathbb{R}^n} & c^T x \\
& \text{s.t.}               & A x + s = b  \\
&                           & b \in \mathbb{R}^m  \\
&                           & s \in \mathcal{K}
\end{align*}
```

where
```math
\begin{align*}
c &= [l_1 - 1, l_2 -1, ... l_N -1, 0, 0, ... 0 \text{(D+1 times)}] \\\\

A &=
\begin{bmatrix}
 -l_1 &    0 & ... &    0 &            0 & ... & 0 & 0  \\
    0 & -l_2 & ... &    0 &            0 & ... & 0 & 0  \\
    : &    : & ... &    : &            0 & ... & 0 & 0  \\
    0 &    0 & ... & -l_N &            0 & ... & 0 & 0  \\
    0 &    0 & ... &    0 & -y_1 X_{1,1} & ... & -y_1 X_{1,N} & -y_1  \\
    0 &    0 & ... &    0 & -y_2 X_{2,1} & ... & -y_1 X_{2,N} & -y_2  \\
    : &    : & ... &    : &           :  & ... &          :   & :   \\
    0 &    0 & ... &    0 & -y_N X_{N,1} & ... & -y_N X_{N,N} & -y_N  \\
\end{bmatrix} \\\\

b &= [0, 0, ... 0 \text{(N times)}, l_1 - 1, l_2 -1, ... l_N -1] \\\\

\mathcal{K} &= \text{Set of Nonnegative cones}
\end{align*}
```


## Experiment 1: Gradient of hyperplane wrt the data point labels

Construct perturbations in data point labels `y` without changing the data point coordinates `X`.

```@example 1
∇ = Float64[]
dy = zeros(N)

# begin differentiating
for Xi in 1:N
    dy[Xi] = 1.0  # set

    MOI.set(
        model,
        DiffOpt.ForwardInConstraint(),
        cons,
        MOI.Utilities.vectorize(dy .* MOI.SingleVariable(b)),
    )

    DiffOpt.forward(model)

    dw = MOI.get.(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        w
    )
    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        b
    )
    push!(∇, norm(dw) + norm(db))

    dy[Xi] = 0.0  # reset the change made above
end
LinearAlgebra.normalize!(∇)
nothing # hide
```

Visualize point sensitivities with respect to separating hyperplane. Note that the gradients are normalized.
```@example 1
p2 = Plots.scatter(
    X[:,1], X[:,2],
    color = [yi > 0 ? :red : :blue for yi in y], label = "",
    markersize = ∇ * 20,
)
Plots.yaxis!(p2, (-2, 4.5))
Plots.plot!(p2, svm_x, svm_y, label = "loss = $(round(loss, digits=2))", width=3)
```


## Experiment 2: Gradient of hyperplane wrt the data point coordinates

Similar to previous example, construct perturbations in data points coordinates `X`.
```@example 1
∇ = Float64[]
dX = zeros(N, D)

# begin differentiating
for Xi in 1:N
    dX[Xi, :] = ones(D)  # set

    for i in 1:D
        MOI.set(
            model,
            DiffOpt.ForwardInConstraint(),
            cons,
            MOI.Utilities.vectorize(dX[:,i] .* MOI.SingleVariable(w[i])),
        )
    end

    DiffOpt.forward(model)

    dw = MOI.get.(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        w
    )
    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        b
    )
    push!(∇, norm(dw) + norm(db))

    dX[Xi, :] = zeros(D)  # reset the change made ago
end
LinearAlgebra.normalize!(∇)
```

We can visualize point sensitivity with respect to the separating hyperplane. Note that the gradients are normalized.
```@example 1
p3 = Plots.scatter(
    X[:,1], X[:,2],
    color = [yi > 0 ? :red : :blue for yi in y], label = "",
    markersize = ∇ * 20,
)
Plots.yaxis!(p3, (-2, 4.5))
Plots.plot!(p3, svm_x, svm_y, label = "loss = $(round(loss, digits=2))", width=3)
```
