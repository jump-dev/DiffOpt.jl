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
    MOI.Nonnegatives(N)
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
    -ones(N)
)
MOI.add_constraint(
    model,
    f,
    MOI.Nonnegatives(N)
)

objective_function = MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.(ones(N), l),
                        0.0
                    )
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

MOI.optimize!(model);

loss = MOI.get(model, MOI.ObjectiveValue())
wv = MOI.get(model, MOI.VariablePrimal(), w)
bv = MOI.get(model, MOI.VariablePrimal(), b)

if should_plot
    p = Plots.scatter(X[:,1], X[:,2], color = [yi > 0 ? :red : :blue for yi in y], label = "")
    Plots.yaxis!(p, (-2, 4.5))
    Plots.plot!(p, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
end

# constructing perturbations
ðA = zeros(2*N, N+D+1)
ðb = zeros(2*N)
ðc = zeros(N+D+1); 

∇ = Float64[]

# begin differentiating
for Xi in 1:N
    ðA[N+Xi, N+D+1] = 1.0
    
    dx, dy, ds = backward(model, ðA, ðb, ðc)
    dl, dw, db = dx[1:N], dx[N+1:N+1+D], dx[N+1+D]
    push!(∇, norm(dw)+norm(db))
    
    ðA[N+Xi, N+D+1] = 0.0
end
normalize!(∇)

# point sensitvity wrt the separating hyperplane
# gradients are normalized

if should_plot
    p2 = Plots.scatter(
        X[:,1], X[:,2], 
        color = [yi > 0 ? :red : :blue for yi in y], label = "",
        markersize = ∇ * 20
    )
    Plots.yaxis!(p2, (-2, 4.5))
    Plots.plot!(p2, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
end

# constructing perturbations
ðA = zeros(2*N, N+D+1)
ðb = zeros(2*N)
ðc = zeros(N+D+1); 

∇ = Float64[]

# begin differentiating
for Xi in 1:N
    ðA[N+Xi, N.+(1:D+1)] = ones(3)
    
    dx, dy, ds = backward(model, ðA, ðb, ðc)
    dl, dw, db = dx[1:N], dx[N+1:N+1+D], dx[N+1+D]
    push!(∇, norm(dw)+norm(db))
    
    ðA[N+Xi, N.+(1:D+1)] = zeros(3)
end
normalize!(∇)

# point sensitvity wrt the separating hyperplane
# gradients are normalized

if should_plot
    p3 = Plots.scatter(
        X[:,1], X[:,2], 
        color = [yi > 0 ? :red : :blue for yi in y], label = "",
        markersize = ∇ * 20
    )
    Plots.yaxis!(p3, (-2, 4.5))
    Plots.plot!(p3, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "loss = $(round(loss, digits=2))")
end
