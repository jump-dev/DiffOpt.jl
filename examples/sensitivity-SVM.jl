import Random
using Test
import SCS
import Plots
using DiffOpt
using MathOptInterface

const MOI = MathOptInterface;

Random.seed!(rand(1:100))
X = vcat(randn(30, 2), randn(30,2) .+ [4.0,1.5]')
y = append!(ones(30), -ones(30));

penalty = 5.0
(nobs, nfeat) = size(X)

model = diff_optimizer(SCS.Optimizer) 

# add variables
l = MOI.add_variables(model, nobs)
w = MOI.add_variables(model, nfeat)
b = MOI.add_variable(model)

t = MOI.add_variable(model)  # extra variable for the SOC constraint

MOI.add_constraint(
    model,
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:nobs, MOI.ScalarAffineTerm.(1.0, l)), zeros(nobs)
    ), 
    MOI.Nonnegatives(nobs)
)

MOI.add_constraint(
    model,
    MOI.VectorAffineFunction([MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, t))], [√penalty]),
    MOI.Zeros(1)
)

pen_const = MOI.add_constraint(
    model, 
    MOI.VectorAffineFunction(MOI.VectorAffineTerm.(1:(nfeat+2), MOI.ScalarAffineTerm.(1.0, vcat(t, w,b))), zeros(nfeat+2)), 
    MOI.SecondOrderCone(nfeat + 2)
)

# define the whole matrix Ax, it'll be easier then
# refer https://discourse.julialang.org/t/solve-minimization-problem-where-constraint-is-the-system-of-linear-inequation-with-mathoptinterface-efficiently/23571/4
Ax = Array{MOI.ScalarAffineTerm{Float64}}(undef, nobs, nfeat+2)
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
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

MOI.optimize!(model);

loss = MOI.get(model, MOI.ObjectiveValue())
λ = MOI.get(model, MOI.ConstraintDual(), pen_const)[1]
wv = MOI.get(model, MOI.VariablePrimal(), w)
bv = MOI.get(model, MOI.VariablePrimal(), b)

p = Plots.scatter(X[:,1], X[:,2], color = [yi > 0 ? :red : :blue for yi in y], label = "")
Plots.yaxis!(p, (-2, 4.5))
Plots.plot!(p, [0.0, 2.0], [-bv / wv[2], (-bv - 2wv[1])/wv[2]], label = "RHS = $(penalty), loss = $(round(loss, digits=2)), lambda = $(round(λ, digits=2))")

@test (wv'wv + bv*bv) ≈ penalty atol=1e-4  # sanity check

# x = model.primal_optimal
# s = MOI.get(model, MOI.ConstraintPrimal(), model.con_idx)
# dual = model.dual_optimal

dA = zeros(2*(nobs+nfeat) + 1, nobs+nfeat+1+1)
db = zeros(2*(nobs+nfeat) + 1)
dc = zeros(nobs+nfeat+1+1); # c = sum(`l`) + 0'w + 0.b + 0.t

# db[nobs+1] = 1.0   # only weighing the penalty constraint i.e. √penalty = t 
# dc[1] = 1.0

# dx, dy, ds = backward_conic!(model, dA, db, dc)

# dl, dw, db, dt = dx[1:50], dx[51:52], dx[53], dx[54]

∇ = []

# begin differentiating
for Xi in 1:nobs
    dA[nobs+5+Xi, nobs.+(1:2)] = ones(2) # set
    
    dx, dy, ds = backward_conic!(model, dA, db, dc)
    push!(∇, norm(dx) + norm(dy))
    
    dA[nobs+5+Xi, nobs.+(1:2)] = zeros(2)  # reset
end
∇ = normalize(∇);

# point sensitvity wrt the separating hyperplane
# gradients are normalized

Plots.plot!(
    Plots.scatter(
        X[:,1], X[:,2], 
        color = [yi > 0 ? :red : :blue for yi in y], label = "",
        markersize= ∇ * 25
    ),
    ylims= (-2, 4.5),
    [-bv / wv[2],
    (-bv - 2wv[1])/wv[2]], label = "penalty = $(penalty), loss = $(round(loss, digits=2)), lambda = $(round(λ, digits=2))"
)

#Plots.heatmap(reshape(∇,1,length(∇)))


