# # Planar Arm Example

# Inverse Kinematics (IK) computes joint angles that place a robot’s end-effector at a desired target $(x_t,y_t)$. For a 2-link planar arm with joint angles $\theta_1,\theta_2$, the end-effector position is:
# ```math
# f(\theta_1,\theta_2) = \bigl(\ell_1\cos(\theta_1) + \ell_2\cos(\theta_1+\theta_2),\,\,
# \ell_1\sin(\theta_1) + \ell_2\sin(\theta_1+\theta_2)\bigr).
# ```
# We can solve an NLP:
# ```math
# \min_{\theta_1,\theta_2} \;\; (\theta_1^2 + \theta_2^2),
# \quad\text{s.t.}\quad f(\theta_1,\theta_2) = (x_t,y_t).
# ```
# Treat $(x_t,y_t)$ as parameters. Once solved, we differentiate w.r.t. $(x_t,y_t)$ to find how small changes in the target location alter the optimal angles - the *differential kinematics*.

# ## Define and solve the 2-link planar arm problem and build sensitivity map of joint angles to target

# First, import the libraries.

using Test
using JuMP
import DiffOpt
using LinearAlgebra
using Statistics
import Ipopt
using Plots
using Plots.Measures

# Fixed data

# Arm geometry
l1 = 1.0;
l2 = 1.0;
reach = l1 + l2          # 2.0
tol = 1e-6               # numerical tolerance for feasibility
# Sampling grid in workspace
grid_res = 25 # grid resolution low for documentation compilation requirements
xs = range(-reach, reach; length = grid_res)
ys = range(-reach, reach; length = grid_res)

heat = fill(NaN, grid_res, grid_res)   # store ‖J_inv‖₂
feas = fill(0.0, grid_res, grid_res)  # feasibility mask
θ1mat = similar(heat)

function ik_angles(x, y; l1 = 1.0, l2 = 1.0, elbow_up = true)
    c2 = (x^2 + y^2 - l1^2 - l2^2) / (2 * l1 * l2)
    θ2 = elbow_up ? acos(clamp(c2, -1, 1)) : -acos(clamp(c2, -1, 1))
    k1 = l1 + l2 * cos(θ2)
    k2 = l2 * sin(θ2)
    θ1 = atan(y, x) - atan(k2, k1)
    return θ1, θ2
end

for (i, x_t) in enumerate(xs), (j, y_t) in enumerate(ys)
    global θ1mat, heat
    ## skip points outside the circular reach
    norm([x_t, y_t]) > reach && continue

    ## ---------- build differentiable NLP ----------
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, xt in Parameter(x_t))
    @variable(model, yt in Parameter(y_t))
    @variable(model, θ1)
    @variable(model, θ2)

    @objective(model, Min, θ1^2 + θ2^2)
    @constraint(model, l1 * cos(θ1) + l2 * cos(θ1 + θ2) == xt)
    @constraint(model, l1 * sin(θ1) + l2 * sin(θ1 + θ2) == yt)

    ## --- supply analytic start values ---
    θ1₀, θ2₀ = ik_angles(x_t, y_t; elbow_up = true)
    set_start_value(θ1, θ1₀)
    set_start_value(θ2, θ2₀)

    optimize!(model)
    println("Solving for target (", x_t, ", ", y_t, ")")
    ## check for optimality
    status = termination_status(model)
    println("Status: ", status)

    status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || continue

    θ1̂ = value(θ1)
    θ1mat[j, i] = θ1̂   # save pose

    ## ---- forward diff wrt  xt  (∂θ/∂x) ----
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, xt, 0.01)
    DiffOpt.forward_differentiate!(model)
    dθ1_dx = DiffOpt.get_forward_variable(model, θ1)
    dθ2_dx = DiffOpt.get_forward_variable(model, θ2)

    ## check first order approximation keeps solution close to target withing tolerance
    θ_approx = [θ1̂ + dθ1_dx, θ1̂ + dθ2_dx]
    x_approx = l1 * cos(θ_approx[1]) + l2 * cos(θ_approx[1] + θ_approx[2])
    y_approx = l1 * sin(θ_approx[1]) + l2 * sin(θ_approx[1] + θ_approx[2])
    _error = [x_approx - (x_t + 0.01), y_approx - y_t]
    println("Error in first order approximation: ", _error)
    feas[j, i] = norm(_error)

    ## ---- forward diff wrt  yt  (∂θ/∂y) ----
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, yt, 0.01)
    DiffOpt.forward_differentiate!(model)
    dθ1_dy = DiffOpt.get_forward_variable(model, θ1)
    dθ2_dy = DiffOpt.get_forward_variable(model, θ2)

    ## 2-norm of inverse Jacobian
    Jinv = [
        dθ1_dx dθ1_dy
        dθ2_dx dθ2_dy
    ]
    heat[j, i] = opnorm(Jinv)            # σ_max  of Jinv
end
# Replace nans with 0.0
heat = replace(heat, NaN => 0.0)

# ## Results with Plot graphs

default(;
    size = (1150, 350),
    legendfontsize = 8,
    guidefontsize = 9,
    tickfontsize = 7,
)

plt = heatmap(
    xs,
    ys,
    heat;
    xlabel = "x target",
    ylabel = "y target",
    clims = (0, quantile(skipmissing(heat), 0.95)),   # clip extremes
    colorbar_title = "‖∂θ/∂(x,y)‖₂",
    left_margin = 5Plots.Measures.mm,
    bottom_margin = 5Plots.Measures.mm,
);

# Overlay workspace boundary
θ = range(0, 2π; length = 200)
plot!(plt, reach * cos.(θ), reach * sin.(θ); c = :white, lw = 1, lab = "reach");

plt_feas = heatmap(
    xs,
    ys,
    feas;
    xlabel = "x target",
    ylabel = "y target",
    clims = (0, 1),
    colorbar_title = "Precision Error",
    left_margin = 5Plots.Measures.mm,
    bottom_margin = 5Plots.Measures.mm,
);

plot!(
    plt_feas,
    reach * cos.(θ),
    reach * sin.(θ);
    c = :white,
    lw = 1,
    lab = "reach",
);

plt_all = plot(
    plt,
    plt_feas;
    layout = (1, 2),
    left_margin = 5Plots.Measures.mm,
    bottom_margin = 5Plots.Measures.mm,
    legend = :bottomright,
)

# Left figure shows the spectral-norm heat-map
# $\bigl\lVert\partial\boldsymbol{\theta}/\partial(x,y)\bigr\rVert_2$
# for a two-link arm - Bright rings mark near-singular poses. Right figure shows the normalized precision error of the first order approximation derived from calculated sensitivities.
