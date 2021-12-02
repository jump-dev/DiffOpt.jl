# # Sensitivity Analysis of SVM

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/docs/src/examples/sensitivity-analysis-svm.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/sensitivity-analysis-svm.ipynb)


# This notebook illustrates sensitivity analysis of data points in a [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine) (inspired from [@matbesancon](http://github.com/matbesancon)'s [SimpleSVMs](http://github.com/matbesancon/SimpleSVMs.jl).)

# For reference, Section 10.1 of https://online.stat.psu.edu/stat508/book/export/html/792 gives an intuitive explanation of what it means to have a sensitive hyperplane or data point. The general form of the SVM training problem is given below (without regularization):

# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & \lambda||w||^2 \sum_{i=1}^{N} \xi_{i} \\
# \mbox{s.t.} & \xi_{i} \ge 0 \quad i=1..N  \\
#             & y_{i} (w^T X_{i} + b) \ge 1 - \xi_{i}\\
# \end{array}
# \end{split}
# ```
# where
# - `X`, `y` are the `N` data points
# - `ξ` is the soft-margin loss.

# ## Define and solve the SVM

# Import the libraries.


using Ipopt, DiffOpt, LinearAlgebra, JuMP
import Random, Plots


# Construct separable, non-trivial data points.

N = 100
D = 2
Random.seed!(62)
X = vcat(randn(N ÷ 2, D), randn(N ÷ 2, D) .+ [4.0, 1.5]')
y = append!(ones(N ÷ 2), -ones(N ÷ 2));


# Let's initialize a special model that can understand sensitivities

model = Model(() -> diff_optimizer(Ipopt.Optimizer))
MOI.set(model, MOI.Silent(), true)

# Add the variables

@variable(model, ξ[1:N])
@variable(model, w[1:D])
@variable(model, b);

# Add the constraints.

@constraint(
    model,
    [i in 1:N],
    ξ[i] >= 0
);
@constraint(
    model,
    cons[i in 1:N],
    y[i] * (dot(X[i, :], w) + b) >= 1 - ξ[i]
);


# Define the objective and solve

@objective(
    model,
    Min,
    0.05 * dot(w, w) + sum(ξ),
)

optimize!(model) # solve


# We can visualize the separating hyperplane.

loss = objective_value(model)

wv = value.(w)

bv = value(b)

svm_x = [0.0, 3.0]
svm_y = (-bv .- wv[1] * svm_x )/wv[2]

p = Plots.scatter(X[:,1], X[:,2], color = [yi > 0 ? :red : :blue for yi in y], label = "")
Plots.yaxis!(p, (-2, 4.5))
Plots.plot!(p, svm_x, svm_y, label = "loss = $(round(loss, digits=2))", width=3)


# # Experiments
# Now that we've solved the SVM, we can compute the sensitivity of optimal values -- the separating hyperplane in our case -- with respect to perturbations of the problem data -- the data points -- using DiffOpt. For illustration, we've explored two questions:

# - How does a change in labels of the data points (`y=1` to `y=-1`, and vice versa) affect the position of the hyperplane? This is achieved by finding the gradient of `w`, `b` with respect to `y[i]`, the classification label of the ith data point.
# - How does a change in coordinates of the data points, `X`, affects the position of the hyperplane? This is achieved by finding gradient of `w`, `b` with respect to `X[i]`, 2D coordinates of the data points.

# ## Experiment 1: Gradient of hyperplane wrt the data point labels

# Construct perturbations in data point labels `y` without changing the data point coordinates `X`.

∇ = zeros(N)
dy = zeros(N);

# Begin differentiating the model.
# analogous to varying theta in the expression:
# ```math
# (y_{i} + \theta) (w^T X_{i} + b) \ge 1 - \xi_{i}
# ```
for i in 1:N
    dy[i] = 1.0 # set
    for j in 1:N
        MOI.set(
            model,
            DiffOpt.ForwardInConstraint(),
            cons[j],
            dy[j] * (dot(X[j,:], index.(w)) + index(b)),
        )
    end
    DiffOpt.forward(model)
    dw = MOI.get.(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        w,
    )
    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        b,
    )
    ∇[i] = norm(dw) + norm(db)
    dy[i] = 0.0
end

normalize!(∇);


# Visualize point sensitivities with respect to separating hyperplane. Note that the gradients are normalized.

p2 = Plots.scatter(
    X[:,1], X[:,2],
    color = [yi > 0 ? :red : :blue for yi in y], label = "",
    markersize = 20 * max.(∇, 0.2 * maximum(∇)),
)
Plots.yaxis!(p2, (-2, 4.5))
Plots.plot!(p2, svm_x, svm_y, label = "loss = $(round(loss, digits=2))", width=3)


# ## Experiment 2: Gradient of hyperplane wrt the data point coordinates

# Similar to previous example, construct perturbations in data points coordinates `X`.

∇ = zeros(N)
dX = zeros(N, D);

# Begin differentiating the model.
# analogous to varying theta in the expression:
# ```math
# y_{i} (w^T (X_{i} + theta) + b) \ge 1 - \xi_{i}
# ```
for i in 1:N
    dX[i, :] = ones(D)  # set
    for j in 1:N
        MOI.set(
            model,
            DiffOpt.ForwardInConstraint(),
            cons[j],
            y[j] * dot(dX[j,:], index.(w)),
        )
    end

    DiffOpt.forward(model)

    dw = MOI.get.(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        w,
    )
    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(),
        b,
    )

    ∇[i] = norm(dw) + norm(db)

    dX[i, :] = zeros(D)  # reset the change made at the beginning of the loop
end

normalize!(∇);

# We can visualize point sensitivity with respect to the separating hyperplane. Note that the gradients are normalized.

p3 = Plots.scatter(
    X[:,1], X[:,2],
    color = [yi > 0 ? :red : :blue for yi in y], label = "",
    markersize = 20 * max.(∇, 0.2 * maximum(∇)),
)
Plots.yaxis!(p3, (-2, 4.5))
Plots.plot!(p3, svm_x, svm_y, label = "loss = $(round(loss, digits=2))", width=3)
