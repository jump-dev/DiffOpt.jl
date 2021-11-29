# # Auto-tuning Hyperparameters

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/examples/autotuning-ridge.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/autotuning-ridge.ipynb)

# This example shows how to learn a hyperparameter in Ridge Regression using a gradient descent routine.
# Let the regularized regression problem be formulated as:

# ```math
# \begin{equation}
# \min_{w} \quad \frac{1}{2nd} \sum_{i=1}^{n} (w^T x_{i} - y_i)^2 + \frac{\alpha}{2d} \| w \|_2^2
# \end{equation}
# ```

# where 
# - `x`, `y` are the data points
# - `w` are the learned weights
# - `α` is the hyperparameter acting on regularization.

# The main optimization model will be formulated with JuMP.
# Using the gradient of the optimal weights with respect to the regularization parameters
# computed with DiffOpt, we can perform a gradient descent on top of the inner model
# to minimize the test loss.

using DiffOpt
using Statistics
using OSQP
using JuMP
using Plots
import Random
using LinearAlgebra


"""
    R2(y_true, y_pred)

Return the coefficient of determination R2 of the prediction in `[0,1]`.
"""
function R2(y_true, y_pred)
    u = norm(y_pred - y_true)^2  # Regression sum of squares
    v = norm(y_true .- mean(y_true))^2  # Total sum of squares
    return 1 - u/v
end

# ## Generating a noisy regression dataset

function create_problem(N, D, noise)
    w = 10 * randn(D)
    X = 10 * randn(N, D)
    y = X * w + noise * randn(N)
    l = N ÷ 2  # test train split
    return (X[1:l, :], X[l+1:N, :], y[1:l], y[l+1:N])
end

Random.seed!(42)

X_train, X_test, y_train, y_test = create_problem(1000, 200, 50);

# ## Defining the regression problem

# We implement the regularized regression problem as a function taking the problem data,
# building a JuMP model and solving it.

function fit_ridge(X, y, α, model = Model(() -> diff_optimizer(OSQP.Optimizer)))
    JuMP.empty!(model)
    set_optimizer_attribute(model, MOI.Silent(), true)
    N, D = size(X)
    @variable(model, w[1:D])
    err_term = X * w - y
    @objective(
        model,
        Min,
        1/(2 * N * D) * dot(err_term, err_term) + 1/(2 * D) * α * dot(w, w),
    )
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        error("Unexpected status: $(termination_status(model))")
    end
    regularized_loss_value = objective_value(model)
    training_loss_value = 1/(2 * N * D) * JuMP.value(dot(err_term, err_term))
    return model, w, value.(w), training_loss_value, regularized_loss_value
end

# We can solve the problem for several values of α
# to visualize the effect of regularization on the testing and training loss.

αs = 0.0:0.01:0.5
Rs = Float64[]
mse_test = Float64[]
mse_train = Float64[]
model = JuMP.Model(() -> diff_optimizer(OSQP.Optimizer))
(Ntest, D) = size(X_test)
for α in αs
    _, _, w_train, training_loss_value, _ = fit_ridge(X_train, y_train, α, model)
    y_pred = X_test * w_train
    push!(Rs, R2(y_test, y_pred))
    push!(mse_test, dot(y_pred - y_test, y_pred - y_test) / (2 * Ntest * D))
    push!(mse_train, training_loss_value)
end

# Visualize the R2 correlation metric

plot(αs, Rs, label=nothing,  xaxis="α", yaxis="R2")
title!("Test coefficient of determination R2")

# Visualize the Mean Score Error metric

plot(αs, mse_test ./ sum(mse_test), label="MSE test", xaxis = "α", yaxis="MSE", legend=(0.8,0.2))
plot!(αs, mse_train ./ sum(mse_train), label="MSE train")
title!("Normalized MSE on training and testing sets")

# ## Leveraging differentiable optimization: computing the derivative of the solution

# Using DiffOpt, we can compute `∂w_i/∂α`, the derivative of the learned solution `̂w`
# w.r.t. the regularization parameter.

function compute_dw_dparam(model, w)
    D = length(w)
    dw_dα = zeros(D)
    MOI.set(
        model, 
        DiffOpt.ForwardInObjective(),
        1/2 * dot(w, w) / D,
    )
    DiffOpt.forward(model)
    for i in 1:D
        dw_dα[i] = MOI.get(
            model,
            DiffOpt.ForwardOutVariablePrimal(), 
            w[i],
        )
    end
    return dw_dα
end

# Using `∂w_i/∂α` computed with `compute_dw_dparam`,
# we can compute the derivative of the test loss w.r.t. the parameter α
# by composing derivatives.

function d_testloss_dα(model, X_test, y_test, w, ŵ)
    N, D = size(X_test)
    dw_dα = compute_dw_dparam(model, w)
    err_term = X_test * ŵ - y_test
    return sum(eachindex(err_term)) do i
        dot(X_test[i,:], dw_dα) * err_term[i]
    end / (N * D)
end

# We can define a meta-optimizer function performing gradient descent
# on the test loss w.r.t. the regularization parameter.

function descent(α0, max_iters=100; fixed_step = 0.01, grad_tol=1e-3)
    α_s = Float64[]
    test_loss_values = Float64[]
    α = α0
    model = JuMP.Model(() -> DiffOpt.diff_optimizer(OSQP.Optimizer))
    for iter in 1:max_iters
        push!(α_s, α)
        _, w, ŵ, _,  = fit_ridge(X_train, y_train, α, model)
        err_term = X_test * ŵ - y_test
        test_loss = norm(err_term)^2 / (2 * length(X_test))
        push!(
            test_loss_values,
            test_loss,
        )
        ∂α = d_testloss_dα(model, X_test, y_test, w, ŵ)
        α -= fixed_step * ∂α
        if abs(∂α) ≤ grad_tol
            break
        end
    end
    return α_s, test_loss_values
end

ᾱ, msē = descent(0.1, 500);

# Visualize gradient descent and convergence 

plot(αs, mse_test, label="MSE test", xaxis = ("α"), legend=:topleft)
plot!(ᾱ, msē, label="learned α", lw = 2)
title!("Regularizer learning")
