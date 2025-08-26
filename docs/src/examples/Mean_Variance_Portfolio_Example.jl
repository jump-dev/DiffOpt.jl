# # Thermal Generation Dispatch Sweep Example

# Consider the Markowitz portfolio selection problem, which allocates weights $x \in \mathbb{R}^n$ to $n$ assets so as to maximize returns subject to a variance limit $v_{\max}$:
# ```math
# \max_{x} \quad \mu^\top x
# \quad\text{s.t.}\quad
# x^\top \Sigma x \;\le\; v_{\max}, \quad
# \mathbf{1}^\top x = 1,\quad
# x \succeq 0,
# ```
# where $\mu$ is the vector of expected returns, $\Sigma$ is the covariance matrix, and $x$ must sum to 1 (fully invest the budget). An efficient conic version of this problem casts the variance limit as a second order cone constraint:

# ```math
# \| \Sigma^{1/2} x \|_{2} \;\le\; \sigma_{\max}
# ```
# where $\Sigma^{1/2}$ is the Cholesky factorization of the covariance matrix and $\sigma_{\max}$ is the standard deviation limit.

# Practitioners often care about an \emph{out-of-sample performance metric} $L(x)$ evaluated on test data or scenarios that differ from those used to form $\mu$ and $\Sigma$. To assess the impact of the risk profile in the performance evaluation, one can compute:
# ```math
# \frac{dL}{d\,\sigma_{\max}} \;=\;
# \underbrace{\frac{\partial L}{\partial x}}_{\text{(1) decision impact}}\;
# \cdot\;
# \underbrace{\frac{\partial x^*}{\partial \sigma_{\max}}}_{\text{(2) from DiffOpt.jl}},
# ```
# where $x^*(\sigma_{\max})$ is the portfolio that solves the conic Markowitz problem under a given risk limit.

# ## Define and solve the Mean-Variance Portfolio Problem for a range of risk limits

# First, import the libraries.

using Test
using JuMP
import DiffOpt
using LinearAlgebra
import SCS
using Plots
using Plots.Measures

# Fixed data

# Training data (in-sample)
Σ = [
    0.002 0.0005 0.001
    0.0005 0.003 0.0002
    0.001 0.0002 0.0025
]
μ_train = [0.05, 0.08, 0.12]

# Test data (out-of-sample)
μ_test = [0.02, -0.3, 0.1]             # simple forecast error example

# Sweep over σ_max
σ_grid = 0.002:0.002:0.06
N = length(σ_grid)

predicted_ret = zeros(N)                 # μ_train' * x*
realised_ret = zeros(N)                 # μ_test'  * x*
loss = zeros(N)                 # L(x*)
dL_dσ = zeros(N)                 # ∂L/∂σ_max  from DiffOpt

for (k, σ_val) in enumerate(σ_grid)

    ## 1) differentiable conic model
    model = DiffOpt.conic_diff_model(SCS.Optimizer)
    set_silent(model)

    ## 2) parameter σ_max
    @variable(model, σ_max in Parameter(σ_val))

    ## 3) portfolio weights
    @variable(model, x[1:3] >= 0)
    @constraint(model, sum(x) <= 1)

    ## 4) objective: maximise expected return (training data)
    @objective(model, Max, dot(μ_train, x))

    ## 5) conic variance constraint  ||L*x|| <= σ_max
    L_chol = cholesky(Symmetric(Σ)).L
    @variable(model, t >= 0)
    @constraint(model, [t; L_chol * x] in SecondOrderCone())
    @constraint(model, t <= σ_max)

    optimize!(model)

    x_opt = value.(x)
    println("Optimal portfolio weights: ", x_opt)

    ## store performance numbers
    predicted_ret[k] = dot(μ_train, x_opt)
    realised_ret[k] = dot(μ_test, x_opt)

    ## -------- reverse differentiation wrt σ_max --------
    DiffOpt.empty_input_sensitivities!(model)
    ## ∂L/∂x   (adjoint)  =  -μ_test
    DiffOpt.set_reverse_variable.(model, x, μ_test)
    DiffOpt.reverse_differentiate!(model)
    dL_dσ[k] = DiffOpt.get_reverse_parameter(model, σ_max)
end

# ## Results with Plot graphs

default(;
    size = (600, 40),
    legendfontsize = 8,
    guidefontsize = 9,
    tickfontsize = 7,
)

# (a) predicted vs realised return
plt_ret = plot(
    σ_grid,
    realised_ret;
    lw = 2,
    label = "Realised (test)",
    xlabel = "σ_max (risk limit)",
    ylabel = "Return",
    title = "Return vs risk limit",
    legend = :bottomright,
);
plot!(
    plt_ret,
    σ_grid,
    predicted_ret;
    lw = 2,
    ls = :dash,
    label = "Predicted (train)",
);

# (b) out-of-sample loss and its gradient
plt_loss = plot(
    σ_grid,
    dL_dσ;
    xlabel = "σ_max (risk limit)",
    ylabel = "∂L/∂σ_max",
    title = "Return Gradient",
    legend = false,
);

plot_all = plot(
    plt_ret,
    plt_loss;
    layout = (1, 2),
    left_margin = 5Plots.Measures.mm,
    bottom_margin = 5Plots.Measures.mm,
)

# Impact of the risk limit $\sigma_{\max}$ on Markowitz
# portfolios.  **Left:** predicted in-sample return versus
# realized out-of-sample return.  **Right:** the
# out-of-sample loss $L(x)$ together with the absolute gradient
# $|\partial L/\partial\sigma_{\max}|$ obtained from
# `DiffOpt.jl`.  The gradient tells the practitioner which
# way—and how aggressively—to adjust $\sigma_{\max}$ to reduce
# forecast error; its value is computed in one reverse-mode call
# without re-solving the optimization for perturbed risk limits.
