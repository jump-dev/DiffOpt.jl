# # Sensitivity Analysis of Ridge Regression

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/docs/src/examples/sensitivity-analysis-ridge.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/sensitivity-analysis-ridge.ipynb)

# This example illustrates the sensitivity analysis of data points in a
# [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression) problem.
# The general form of the problem is given below:

# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & \sum_{i=1}^{N} (y_{i} - w x_{i} - b)^2 + \alpha (w^2 + b^2) \\
# \end{array}
# \end{split}
# ```
# where
# - `w`, `b` are slope and intercept of the regressing line
# - `x`, `y` are the N data points
# - `α` is the regularization constant
#
# which is equivalent to:
# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & e^{\top}e + \alpha (w^2 + b^2) \\
# \mbox{s.t.} & e_{i} = y_{i} - w x_{i} - b \quad \quad i=1..N  \\
# \end{array}
# \end{split}
# ```



# This tutorial uses the following packages

using JuMP
import DiffOpt
import Random
import OSQP
import Plots
import LinearAlgebra: normalize!, dot

## Define and solve the problem

# Construct a set of noisy (guassian) data points around a line.

Random.seed!(42)

N = 100

w = 2 * abs(randn())
b = rand()
X = randn(N)
Y = w * X .+ b + 0.8 * randn(N)

# The helper method `fitRidge` defines and solves the corresponding model.

function fitRidge(X, Y, alpha = 0.1)
    N = length(Y)
    model = Model(() -> DiffOpt.diff_optimizer(OSQP.Optimizer))
    set_silent(model)
    @variable(model, w)
    @variable(model, b)
    @variable(model, e[1:N])
    @constraint(model, cons[i=1:N], e[i] == Y[i] - w * X[i] - b)
    @objective(
        model,
        Min,
        dot(e, e) + alpha * (sum(w * w) + sum(b * b)),
    )
    optimize!(model)
    return model, w, b, cons
end


# Train on the data generated.

model, w, b, cons = fitRidge(X, Y)
ŵ, b̂ = value(w), value(b)

# We can visualize the approximating line.

p = Plots.scatter(X, Y, label="")
mi, ma = minimum(X), maximum(X)
Plots.plot!(p, [mi, ma], [mi * ŵ + b̂, ma * ŵ + b̂], color=:red, label="")


## Differentiate

# Now that we've solved the problem, we can compute the sensitivity of optimal
# values -- the approximating line component `b` in this case -- with
# respect to perturbations in the data points (`x`,`y`).

∇ = zero(X)

# Begin differentiating the model.
# analogous to varying θ in the expression:
# ```math
# e_i = (y_{i} + \theta_{y_i}) - w (x_{i} + \theta_{x_{i}}) - b
# ```

for i in 1:N
    for j in 1:N
        MOI.set(
            model,
            DiffOpt.ForwardInConstraint(),
            cons[j],
            i == j ? index(w) + 1.0 : 0.0 * index(w)
        )
    end
    DiffOpt.forward(model)
    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        b
    )
    ∇[i] = db
end

normalize!(∇);


# Visualize point sensitivities with respect to regressing line.
# Note that the gradients are normalized.

p = Plots.scatter(
    X, Y,
    color = [x > 0 ? :red : :blue for x in ∇],
    markersize = [25 * abs(x) for x in ∇],
    label = ""
)
mi, ma = minimum(X), maximum(X)
Plots.plot!(p, [mi, ma], [mi * ŵ + b̂, ma * ŵ + b̂], color = :red, label = "")

