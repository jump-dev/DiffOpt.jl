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

(nobs, nfeat) = size(X)

model = diff_optimizer(SCS.Optimizer) 

# add variables
l = MOI.add_variables(model, nobs)
w = MOI.add_variables(model, nfeat)
b = MOI.add_variable(model)

MOI.add_constraint(
    model,
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:nobs, MOI.ScalarAffineTerm.(1.0, l)), zeros(nobs)
    ), 
    MOI.Nonnegatives(nobs)
)

# define the whole matrix Ax, it'll be easier then
# refer https://discourse.julialang.org/t/solve-minimization-problem-where-constraint-is-the-system-of-linear-inequation-with-mathoptinterface-efficiently/23571/4
Ax = Matrix{MOI.ScalarAffineTerm{Float64}}(undef, nobs, nfeat+2)
for i in 1:nobs
    Ax[i, :] = MOI.ScalarAffineTerm.([1.0; y[i]*X[i,:]; y[i]], [l[i]; w; b])
end
terms = MOI.VectorAffineTerm.(1:nobs, Ax)
f = MOI.VectorAffineFunction(
    vec(terms),
    -ones(nobs)
)
MOI.add_constraint(
    model,
    f,
    MOI.Nonnegatives(nobs)
)

objective_function = MOI.ScalarAffineFunction(
                        MOI.ScalarAffineTerm.(ones(nobs), l),
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
ðA = zeros(2*nobs, nobs+nfeat+1)
ðb = zeros(2*nobs)
ðc = zeros(nobs+nfeat+1); # c = sum(`l`) + 0'w + 0.b

∇ = Float64[]

# begin differentiating
for Xi in 1:nobs
    ðA[nobs+Xi, nobs+nfeat+1] = 1.0
    
    dx, dy, ds = backward(model, ðA, ðb, ðc)
    dl, dw, db = dx[1:nobs], dx[nobs+1:nobs+1+nfeat], dx[nobs+1+nfeat]
    push!(∇, norm(dw)+norm(db))
    
    ðA[nobs+Xi, nobs+nfeat+1] = 0.0
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
ðA = zeros(2*nobs, nobs+nfeat+1)
ðb = zeros(2*nobs)
ðc = zeros(nobs+nfeat+1); # c = sum(`l`) + 0'w + 0.b

∇ = Float64[]

# begin differentiating
for Xi in 1:nobs
    ðA[nobs+Xi, nobs.+(1:nfeat+1)] = ones(3)
    
    dx, dy, ds = backward(model, ðA, ðb, ðc)
    dl, dw, db = dx[1:nobs], dx[nobs+1:nobs+1+nfeat], dx[nobs+1+nfeat]
    push!(∇, norm(dw)+norm(db))
    
    ðA[nobs+Xi, nobs.+(1:nfeat+1)] = zeros(3)
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
