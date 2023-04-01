# # Auto-tuning Hyperparameters

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/autotuning-ridge.jl)

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

# This tutorial uses the following packages

using JuMP     # The mathematical programming modelling language
import DiffOpt # JuMP extension for differentiable optimization
import Ipopt    # Optimization solver that handles quadratic programs
import Plots   # Graphing tool
import LinearAlgebra: norm, dot
import Random

# ## Generating a noisy regression dataset

Random.seed!(42)

N = 100
D = 20
noise = 5

w_real = 10 * randn(D)
X = 10 * randn(N, D)
y = X * w_real + noise * randn(N)
l = N ÷ 2  # test train split

X_train = X[1:l, :]
X_test  = X[l+1:N, :]
y_train = y[1:l]
y_test  = y[l+1:N];

# ## Defining the regression problem

# We implement the regularized regression problem as a function taking the problem data,
# building a JuMP model and solving it.

function fit_ridge(model, X, y, α)
    JuMP.empty!(model)
    set_silent(model)
    N, D = size(X)
    @variable(model, w[1:D])
    @expression(model, err_term, X * w - y)
    @objective(
        model,
        Min,
        dot(err_term, err_term) / (2 * N * D) + α * dot(w, w) / (2 * D),
    )
    optimize!(model)
    @assert termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
    return w
end

# We can solve the problem for several values of α
# to visualize the effect of regularization on the testing and training loss.

αs = 0.00:0.01:0.50
mse_test = Float64[]
mse_train = Float64[]
model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
(Ntest, D) = size(X_test)
(Ntrain, D) = size(X_train)
for α in αs
    w = fit_ridge(model, X_train, y_train, α)
    ŵ = value.(w)
    ŷ_test = X_test * ŵ 
    ŷ_train = X_train * ŵ 
    push!(mse_test, norm(ŷ_test - y_test)^2 / (2 * Ntest * D))
    push!(mse_train, norm(ŷ_train - y_train)^2 / (2 * Ntrain * D))
end

# Visualize the Mean Score Error metric

Plots.plot(
    αs, mse_test ./ sum(mse_test),
    label="MSE test", xaxis = "α", yaxis="MSE", legend=(0.8, 0.2),
    width=3,
)
Plots.plot!(
    αs, mse_train ./ sum(mse_train),
    label="MSE train",
    linestyle=:dash,
    width=3,
)
Plots.title!("Normalized MSE on training and testing sets")

# ## Leveraging differentiable optimization: computing the derivative of the solution

# Using DiffOpt, we can compute `∂w_i/∂α`, the derivative of the learned solution `̂w`
# w.r.t. the regularization parameter.

function compute_dw_dα(model, w)
    D = length(w)
    dw_dα = zeros(D)
    MOI.set(
        model, 
        DiffOpt.ForwardObjectiveFunction(),
        dot(w, w)  / (2 * D),
    )
    DiffOpt.forward_differentiate!(model)
    for i in 1:D
        dw_dα[i] = MOI.get(
            model,
            DiffOpt.ForwardVariablePrimal(), 
            w[i],
        )
    end
    return dw_dα
end

# Using `∂w_i/∂α` computed with `compute_dw_dα`,
# we can compute the derivative of the test loss w.r.t. the parameter α
# by composing derivatives.

function d_testloss_dα(model, X_test, y_test, w, ŵ)
    N, D = size(X_test)
    dw_dα = compute_dw_dα(model, w)
    err_term = X_test * ŵ - y_test
    return sum(eachindex(err_term)) do i
        dot(X_test[i,:], dw_dα) * err_term[i]
    end / (N * D)
end

# We can define a meta-optimizer function performing gradient descent
# on the test loss w.r.t. the regularization parameter.

function descent(α0, max_iters=100; fixed_step = 0.01, grad_tol=1e-3)
    α_s = Float64[]
    ∂α_s = Float64[]
    test_loss = Float64[]
    α = α0
    N, D = size(X_test)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    for iter in 1:max_iters
        w = fit_ridge(model, X_train, y_train, α)
        ŵ = value.(w)
        err_term = X_test * ŵ - y_test
        ∂α = d_testloss_dα(model, X_test, y_test, w, ŵ)
        push!(α_s, α)
        push!(∂α_s, ∂α)
        push!(test_loss, norm(err_term)^2 / (2 * N * D))
        α -= fixed_step * ∂α
        if abs(∂α) ≤ grad_tol
            break
        end
    end
    return α_s, ∂α_s, test_loss
end

ᾱ, ∂ᾱ, msē = descent(0.10, 500)
iters = 1:length(ᾱ);

# Visualize gradient descent and convergence

Plots.plot(
    αs, mse_test,
    label="MSE test", xaxis = ("α"),
    legend=:topleft,
    width=2,
)
Plots.plot!(
    ᾱ, msē,
    label="learned α", width = 5,
    style=:dot,
)
Plots.title!("Regularizer learning")

# Visualize the convergence of α to its optimal value

Plots.plot(
    iters, ᾱ, label = nothing, color = :blue,
    xaxis = ("Iterations"), legend=:bottom,
    title = "Convergence of α"
)

# Visualize the convergence of the objective function

Plots.plot(
    iters, msē, label = nothing, color = :red,
    xaxis = ("Iterations"), legend=:bottom,
    title = "Convergence of MSE"
)

# Visualize the convergence of the derivative to zero

Plots.plot(
    iters, ∂ᾱ, label = nothing, color = :green,
    xaxis = ("Iterations"), legend=:bottom,
    title = "Convergence of ∂α"
)
