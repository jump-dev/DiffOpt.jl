"""
    Source code for the example given in sensitivity-ananlysis-svm.md
"""

import Random
using Test
import SCS
using DiffOpt
using LinearAlgebra
using MathOptInterface

const MOI = MathOptInterface;

# optional Plot to integrate in tests
# set ENV["SVM_PLOT"] = "1" to build plots
const should_plot = get(ENV, "SVM_PLOT", nothing) == "1"
if should_plot
    using Plots
end

N = 50
D = 2
Random.seed!(rand(1:100))
X = vcat(randn(N, D), randn(N,D) .+ [4.0,1.5]')
y = append!(ones(N), -ones(N));

model = diff_optimizer(SCS.Optimizer)

# add variables
l = MOI.add_variables(model, N)
w = MOI.add_variables(model, D)
b = MOI.add_variable(model)

MOI.add_constraint(
    model,
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:N, MOI.ScalarAffineTerm.(1.0, l)), zeros(N)
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

_objective_function = MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.(ones(N), l),
                        0.0,
                    )
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), _objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

MOI.optimize!(model)

loss = MOI.get(model, MOI.ObjectiveValue())
wv = MOI.get(model, MOI.VariablePrimal(), w)
bv = MOI.get(model, MOI.VariablePrimal(), b)

if should_plot
    p = Plots.scatter(X[:,1], X[:,2], color = [yi > 0 ? :red : :blue for yi in y], label = "")
    Plots.yaxis!(p, (-2, 4.5))
    Plots.plot!(p, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
end


## Experiment 1: Gradient of hyperplane wrt the data point labels

∇ = Float64[]
dy = zeros(N)

# begin differentiating
for Xi in 1:N
    dy[Xi] = 1.0  # set

    MOI.set(
        model,
        DiffOpt.ForwardInConstraint(),
        cons,
        MOIU.vectorize(dy .* MOI.SingleVariable(b)),
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
normalize!(∇)

# point sensitivity wrt the separating hyperplane
# gradients are normalized

if should_plot
    p2 = Plots.scatter(
        X[:,1], X[:,2],
        color = [yi > 0 ? :red : :blue for yi in y], label = "",
        markersize = ∇ * 20,
    )
    Plots.yaxis!(p2, (-2, 4.5))
    Plots.plot!(p2, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
end

## Experiment 2: Gradient of hyperplane wrt the data point coordinates

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
            MOIU.vectorize(dX[:,i] .* MOI.SingleVariable(w[i])),
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
normalize!(∇)

# point sensitivity wrt the separating hyperplane
# gradients are normalized

if should_plot
    p3 = Plots.scatter(
        X[:,1], X[:,2],
        color = [yi > 0 ? :red : :blue for yi in y], label = "",
        markersize = ∇ * 20,
    )
    Plots.yaxis!(p3, (-2, 4.5))
    Plots.plot!(p3, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
end
