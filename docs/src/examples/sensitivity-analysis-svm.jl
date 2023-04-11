# # Sensitivity Analysis of SVM

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/sensitivity-analysis-svm.jl)

# This notebook illustrates sensitivity analysis of data points in a [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine) (inspired from [@matbesancon](http://github.com/matbesancon)'s [SimpleSVMs](http://github.com/matbesancon/SimpleSVMs.jl).)

# For reference, Section 10.1 of https://online.stat.psu.edu/stat508/book/export/html/792 gives an intuitive explanation of what it means to have a sensitive hyperplane or data point. The general form of the SVM training problem is given below (with $\ell_2$ regularization):

# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & \lambda||w||^2 + \sum_{i=1}^{N} \xi_{i} \\
# \mbox{s.t.} & \xi_{i} \ge 0 \quad \quad i=1..N  \\
#             & y_{i} (w^T X_{i} + b) \ge 1 - \xi_{i} \quad i=1..N  \\
# \end{array}
# \end{split}
# ```
# where
# - `X`, `y` are the `N` data points
# - `w` is the support vector
# - `b` determines the offset `b/||w||` of the hyperplane with normal `w`
# - `ξ` is the soft-margin loss
# - `λ` is the $\ell_2$ regularization.
#
# This tutorial uses the following packages

using JuMP     # The mathematical programming modelling language
import DiffOpt # JuMP extension for differentiable optimization
import Ipopt   # Optimization solver that handles quadratic programs
import Plots   # Graphing tool
import LinearAlgebra: dot, norm
import Random

# ## Define and solve the SVM

# Construct two clusters of data points.

N = 100
D = 2

Random.seed!(62)
X = vcat(randn(N ÷ 2, D), randn(N ÷ 2, D) .+ [2.0, 2.0]')
y = append!(ones(N ÷ 2), -ones(N ÷ 2))
λ = 0.05;

# Let's initialize a special model that can understand sensitivities

model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
MOI.set(model, MOI.Silent(), true)

# Add the variables

@variable(model, ξ[1:N] >= 0)
@variable(model, w[1:D])
@variable(model, b);

# Add the constraints.

@constraint(model, con[i in 1:N], y[i] * (dot(X[i, :], w) + b) >= 1 - ξ[i]);

# Define the objective and solve

@objective(model, Min, λ * dot(w, w) + sum(ξ),)

optimize!(model)

# We can visualize the separating hyperplane.

loss = objective_value(model)

wv = value.(w)

bv = value(b)

svm_x = [-2.0, 4.0] # arbitrary points
svm_y = (-bv .- wv[1] * svm_x) / wv[2]

p = Plots.scatter(
    X[:, 1],
    X[:, 2],
    color = [yi > 0 ? :red : :blue for yi in y],
    label = "",
)
Plots.plot!(
    p,
    svm_x,
    svm_y,
    label = "loss = $(round(loss, digits=2))",
    width = 3,
)

# ## Gradient of hyperplane wrt the data point coordinates

# Now that we've solved the SVM, we can compute the sensitivity of optimal
# values -- the separating hyperplane in our case -- with respect to
# perturbations of the problem data -- the data points -- using DiffOpt.

# How does a change in coordinates of the data points, `X`,
# affects the position of the hyperplane?
# This is achieved by finding gradients of `w` and `b` with respect to `X[i]`.

# Begin differentiating the model.
# analogous to varying θ in the expression:
# ```math
# y_{i} (w^T (X_{i} + \theta) + b) \ge 1 - \xi_{i}
# ```
∇ = zeros(N)
for i in 1:N
    for j in 1:N
        if i == j
            ## we consider identical perturbations on all x_i coordinates
            MOI.set(
                model,
                DiffOpt.ForwardConstraintFunction(),
                con[j],
                y[j] * sum(w),
            )
        else
            MOI.set(model, DiffOpt.ForwardConstraintFunction(), con[j], 0.0)
        end
    end
    DiffOpt.forward_differentiate!(model)
    dw = MOI.get.(model, DiffOpt.ForwardVariablePrimal(), w)
    db = MOI.get(model, DiffOpt.ForwardVariablePrimal(), b)
    ∇[i] = norm(dw) + norm(db)
end

# We can visualize the separating hyperplane sensitivity with respect to the data points.
# Note that all the small numbers were converted into 1/10 of the
# largest value to show all the points of the set.

p3 = Plots.scatter(
    X[:, 1],
    X[:, 2],
    color = [yi > 0 ? :red : :blue for yi in y],
    label = "",
    markersize = 2 * (max.(1.8∇, 0.2 * maximum(∇))),
)
Plots.yaxis!(p3, (-2, 4.5))
Plots.plot!(p3, svm_x, svm_y, label = "", width = 3)
Plots.title!("Sensitivity of the separator to data point variations")
