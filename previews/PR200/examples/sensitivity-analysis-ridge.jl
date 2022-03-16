# # Sensitivity Analysis of Ridge Regression

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/sensitivity-analysis-ridge.jl)

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
import Ipopt
import Plots
import LinearAlgebra: normalize!, dot

# ## Define and solve the problem

# Construct a set of noisy (guassian) data points around a line.

Random.seed!(42)

N = 100

w = 2 * abs(randn())
b = rand()
X = randn(N)
Y = w * X .+ b + 0.8 * randn(N);

# The helper method `fit_ridge` defines and solves the corresponding model.
# The ridge regression is modeled with quadratic programming
# (quadratic objective and linear constraints) and solved in generic methods
# of Ipopt. This is not the standard way of solving the ridge regression problem
# this is done here for didactic purposes.

function fit_ridge(X, Y, alpha = 0.1)
    N = length(Y)
    ## Initialize a JuMP Model with Ipopt solver
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)
    @variable(model, w) # angular coefficient
    @variable(model, b) # linear coefficient
    ## expression defining approximation error
    @expression(model, e[i=1:N], Y[i] - w * X[i] - b)
    ## objective minimizing squared error and ridge penalty
    @objective(
        model,
        Min,
        1 / N * dot(e, e) + alpha * (w^2 + b^2),
    )
    optimize!(model)
    return model, w, b # return model & variables
end


# Plot the data points and the fitted line for different alpha values

p = Plots.scatter(X, Y, label=nothing, legend=:topleft)
mi, ma = minimum(X), maximum(X)
Plots.title!("Fitted lines and points")

for alpha in 0.5:0.5:1.5
    local model, w, b = fit_ridge(X, Y, alpha)
    ŵ = value(w)
    b̂ = value(b)
    Plots.plot!(p, [mi, ma], [mi * ŵ + b̂, ma * ŵ + b̂], label="alpha=$alpha", width=2)
end
p

# ## Differentiate

# Now that we've solved the problem, we can compute the sensitivity of optimal
# values of the slope `w` with
# respect to perturbations in the data points (`x`,`y`).

alpha = 0.4
model, w, b = fit_ridge(X, Y, alpha)
ŵ = value(w)
b̂ = value(b)

# We first compute sensitivity of the slope with respect to a perturbation of the independent
# variable `x`.

# Recalling that the points $(x_i, y_i)$ appear in the objective function as:
# `(yi - b - w*xi)^2`, the `DiffOpt.ForwardInObjective` attribute must be set accordingly,
# with the terms multiplying the parameter in the objective.

∇x = zero(X)
∇y = zero(X)

# Sensitivity with respect to x and y
for i in 1:N
    MOI.set(
        model,
        DiffOpt.ForwardInObjective(),
        -(w^2 * X[i] + 2b * w - 2 * w * Y[i])
    )
    DiffOpt.forward(model)
    ∇x[i] = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        w
    )
    MOI.set(
        model,
        DiffOpt.ForwardInObjective(),
        (Y[i] - 2b - 2w * X[i]),
    )
    DiffOpt.forward(model)
    ∇y[i] = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        w
    )
end

# Visualize point sensitivities with respect to regression points.

p = Plots.scatter(
    X, Y,
    color = [dw < 0 ? :blue : :red for dw in ∇x],
    markersize = [5 * abs(dw) + 1.2 for dw in ∇x],
    label = ""
)
mi, ma = minimum(X), maximum(X)
Plots.plot!(p, [mi, ma], [mi * ŵ + b̂, ma * ŵ + b̂], color = :blue, label = "")
Plots.title!("Regression slope sensitivity with respect to x")

#

p = Plots.scatter(
    X, Y,
    color = [dw < 0 ? :blue : :red for dw in ∇y],
    markersize = [5 * abs(dw) + 1.2 for dw in ∇y],
    label = ""
)
mi, ma = minimum(X), maximum(X)
Plots.plot!(p, [mi, ma], [mi * ŵ + b̂, ma * ŵ + b̂], color = :blue, label = "")
Plots.title!("Regression slope sensitivity with respect to y")

# Note the points with less central `x` values induce a greater y sensitivity of the slope,
# while points further away from the regression line (with greater absolute error) induce more sensitivity
# in the x direction.
