# # Auto-tuning Hyperparameters

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/examples/autotuning-ridge.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/autotuning-ridge.ipynb)

# This example shows how to learn a hyperparameter in Ridge Regression using a gradient descent routine.
# Let the problem be modelled as

# ```math
# \begin{equation}
# \min_{w} \quad \frac{1}{2n} \sum_{i=1}^{n} (y_{i} - w^T x_{i})^2 + \alpha \| w \|_2^2
# \end{equation}
# ```

# where 
# - `x`, `y` are the data points
# - `w` constitutes weights of the regressing line
# - `α` is the only hyperparameter acting on regularization.

# We will try to minimize the (non-regularized) mean square error over `α` values.

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

# Create a non-trivial, noisy regression dataset

function create_problem(N, D, noise)
    w = 10 * randn(D)
    X = 10 * randn(N, D)

    # if noise=0, then there is no need of regularization and
    # alpha=0 will give the best R2 score
    y = X * w + noise * randn(N)

    l = N ÷ 2  # test train split
    return X[1:l, :], X[l+1:N, :], y[1:l], y[l+1:N]
end

Random.seed!(42)

X_train, X_test, y_train, y_test = create_problem(1000, 200, 50);


# Define a helper function for regression

function fit_ridge(X, y, α)
    model = Model(() -> diff_optimizer(OSQP.Optimizer))

    N, D = size(X)

    @variable(model, w[1:D])
    set_optimizer_attribute(model, MOI.Silent(), true)
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

    loss_value = objective_value(model)
    return model, w, loss_value, value.(w)
end

# Solve the problem for several values of α

αs = 0.0:0.01:0.5
Rs = Float64[]
mse = Float64[]
for α in αs
    _, _, _, w_train = fit_ridge(X_train, y_train, α)
    y_pred = X_test * w_train
    push!(Rs, R2(y_test, y_pred))
    push!(mse, norm(y_pred - y_test)^2 / length(y_pred))
end

# Visualize the R2 correlation metric

plot(αs, Rs, label="R2 prediction score",  xaxis = "α")

# Visualize the Mean Score Error metric

plot(αs, mse, label="MSE", xaxis = "α")

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
        # compute grad
        dw_dα[i] = MOI.get(
            model,
            DiffOpt.ForwardOutVariablePrimal(), 
            w[i],
        )
    end
    return dw_dα
end

# Define the derivative of the test loss with respect to the parameter α
function d_testloss_dα(model, X_test, y_test, w, ŵ)
    N, D = size(X_test)
    dw_dα = compute_dw_dparam(model, w)
    err_term = X_test * ŵ - y_test
    return sum(eachindex(err_term)) do i
        dot(X_test[i,:], dw_dα) * err_term[i]
    end / (N * D)
end


# Plot the gradient ∂l/∂α

∂l_∂αs_train = Float64[]
∂l_∂αs_test = Float64[]
ltrain = Float64[]
ltest = Float64[]
N, D = size(X_train)

for α in αs
    model, w, _, ŵ = fit_ridge(X_train, y_train, α)
    ∂l_∂w = α * ŵ / D - X_train' * (X_train*ŵ - y_train) / (N * D)
    # testing optimality wrt regularized model
    @show norm(∂l_∂w)
    # @assert norm(∂l_∂w) < 5e-2

    push!(
        ∂l_∂αs_test,
        d_testloss_dα(model, X_test, y_test, w, ŵ),
    )
    push!(
        ∂l_∂αs_train,
        d_testloss_dα(model, X_train, y_train, w, ŵ),
    )
    push!(
        ltrain,
        dot(X_train * ŵ - y_train, X_train * ŵ - y_train)/(2 * length(y_train)),
    )
    push!(
        ltest,
        dot(X_test * ŵ - y_test, X_test * ŵ - y_test)/(2 * length(y_test)),
    )
end

plot(αs, ∂l_∂αs_train, label="∂l/∂α",  xaxis = ("α"), title="Training")



# Define helper function for Gradient Descent

"""
    descent(α, max_iters=100)

start from initial value of regularization constant
do gradient descent on alpha
until the MSE keeps on decreasing
"""
function descent(α, max_iters=100)
    α_s = Float64[]
    mse = Float64[]
    iter = 0
    ∂α = 10
    while abs(∂α) > 0.001 && iter < max_iters
        iter += 1
        model, w, _, ŵ = fit_ridge(X_train, y_train, α)
        ∂α = ∇model(model, X_test, y_test, w, ŵ) # fetch the gradient
        α -= 0.5 * ∂α  # update by a fixed amount
        push!(α_s, α)
        y_pred = X_test * ŵ
        mse_i = sum((y_pred - y_test).^2) 
        @show ∂α
        @show mse_i
        @show α
        push!(mse, mse_i)
    end
    return α_s, mse
end

ᾱ, msē = descent(1.0, 500);

# Visualize gradient descent and convergence 

plot(αs, mse, label="MSE", xaxis = ("α"), legend=:topleft)
plot!((ᾱ), msē, label="G.D. for α", lw = 2)
