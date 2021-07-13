"""
    Source code for the example given in sensitivity-analysis-ridge.md
"""

import Random
import OSQP
using DiffOpt
using JuMP
using LinearAlgebra

# optional Plot to integrate in tests
# set ENV["BUILD_PLOT"] = "1" to build plots
const should_plot = get(ENV, "BUILD_PLOT", nothing) == "1"
if should_plot
    using Plots
end

function create_problem(N=100)
    m = 2*abs(randn())
    b = rand()
    X = randn(N)
    Y = m*X .+ b + 0.8*randn(N)
    
    return X, Y
end

X, Y = create_problem();

function fitRidge(X,Y,alpha=0.1)
    model = Model(() -> diff_optimizer(OSQP.Optimizer))

    # add variables
    @variable(model, w)
    @variable(model, b)
    set_optimizer_attribute(model, MOI.Silent(), true)
    
    @objective(
        model,
        Min,
        sum((Y - w*X .- b).*(Y - w*X .- b)) + alpha*(sum(w*w)+sum(b*b)),
    )

    optimize!(model)

    loss = objective_value(model)
    return model, w, b, loss, value(w), value(b)
end

model, w, b, loss_train, ŵ, b̂ = fitRidge(X, Y)

# plot the regressing line
if should_plot
    p = Plots.scatter(X, Y, label="")
    mi, ma = minimum(X), maximum(X)
    Plots.plot!(p, [mi, ma], [mi*ŵ+b̂, ma*ŵ+b̂], color=:red, label="")
end

# get the gradients
∇ = zero(X)
for i in 1:length(X)
    # MOI.set(
    #     model,
    #     DiffOpt.ForwardInObjective(), 
    #     w, 
    #     -2*(Y[i] + X[i])
    # ) 
    # MOI.set(
    #     model, 
    #     DiffOpt.ForwardInObjective(), 
    #     w,
    #     w,
    #     2*X[i]
    # )

    MOI.set(
        model, 
        DiffOpt.ForwardInObjective(), 
        MOI.ScalarQuadraticFunction(
            [MOI.ScalarAffineTerm(-2(Y[i] + X[i]), w.index)], 
            [MOI.ScalarQuadraticTerm(2X[i], w.index, w.index)], 
            0.0
        )
    )
    
    DiffOpt.forward(model)

    db = MOI.get(
        model,
        DiffOpt.ForwardOutVariablePrimal(), 
        b
    )

    ∇[i] = db
end
normalize!(∇);


# plot the data-point sensitivities
if should_plot
    p = Plots.scatter(
        X, Y,
        color=[x>0 ? :red : :blue for x in ∇],
        markersize=[25*abs(x) for x in ∇],
        label=""
    )
    mi, ma = minimum(X), maximum(X)
    Plots.plot!(p, [mi, ma], [mi*ŵ+b̂, ma*ŵ+b̂], color=:red, label="")
end

