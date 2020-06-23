"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note 
Currently supports differentiating linear and quadratic programs only.
"""

const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex


mutable struct Optimizer{OT<:MOI.ModelLike}
    optimizer::OT
    primal_optimal::Array{Float64}  # solution
    dual_optimal::Array{Float64}  
    var_idx::Vector{VI}
    con_idx::Vector{CI}

    function Optimizer(optimizer_constructor)
        new(
            MOI.instantiate(optimizer_constructor, with_bridge_type=Float64),
            zeros(0),
            zeros(0),
            Vector{VI}(),
            Vector{CI}()
        )
    end
end


function MOI.add_variable(model::Optimizer)
    vi = MOI.add_variable(model.optimizer)
    push!(model.var_idx, vi)
    return vi
end


function MOI.add_variables(model::Optimizer, N::Int)
    return VI[MOI.add_variable(model) for i in 1:N]
end


function MOI.add_constraint(model::Optimizer, f::MOI.AbstractFunction, s::MOI.AbstractSet)
    ci = MOI.add_constraint(model.optimizer, f, s)
    push!(model.con_idx, ci)
    return ci
end


function MOI.add_constraints(model::Optimizer, f::Vector{F}, s::Vector{S}) where {F<:MOI.AbstractFunction, S<:MOI.AbstractSet}
    return CI[MOI.add_constraint(model, f[i], s[i]) for i in 1:size(f)[1]]
end


function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{F}, f::F) where {F <: Union{
        MOI.SingleVariable,
        MOI.ScalarAffineFunction{T},
        MOI.ScalarQuadraticFunction{T}
    }} where T
    MOI.set(model.optimizer, attr, f)
    return
end


function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    return MOI.set(model.optimizer, attr, sense)
end


function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.optimizer, attr)
end


function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices{F, S}) where {F<:MOI.AbstractFunction, S<:MOI.AbstractSet}
    return MOI.get(model.optimizer, attr)
end


function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    return MOI.get(model.optimizer, attr, ci)
end


function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    return MOI.get(model.optimizer, attr, ci)
end


function MOI.optimize!(model::Optimizer)
    MOI.optimize!(model.optimizer)

    # check status
    @assert MOI.get(model.optimizer, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
        
    # save the solution
    model.primal_optimal = MOI.get(model.optimizer, MOI.VariablePrimal(), model.var_idx)
    model.dual_optimal = MOI.get(model.optimizer, MOI.ConstraintDual(), model.con_idx)
end

    
# """
#     Method to differentiate and obtain gradients/jacobians
#     of z, λ, ν  with respect to the parameters specified in
#     in argument
# """
# function backward(params)
#     grads = []
#     LHS = create_LHS_matrix(z, λ, Q, G, h, A)

#     # compute the jacobian of (z, λ, ν) with respect to each 
#     # of the parameters recieved in the method argument
#     # for instance, to get the jacobians w.r.t vector `b`
#     # substitute db = I and set all other differential terms
#     # in the right hand side to zero. For more info refer 
#     # equation (6) of https://arxiv.org/pdf/1703.00443.pdf
#     for param in params
#         if param == "Q"
#             RHS = create_RHS_matrix(z, ones(nz, nz), zeros(nz, 1),
#                                     λ, zeros(nineq, nz), zeros(nineq,1),
#                                     ν, zeros(neq, nz), zeros(neq, 1))
#             push!(grads, LHS \ RHS)
#         elseif param == "q"
#             RHS = create_RHS_matrix(z, zeros(nz, nz), ones(nz, 1),
#                                     λ, zeros(nineq, nz), zeros(nineq,1),
#                                     ν, zeros(neq, nz), zeros(neq, 1))
#             push!(grads, LHS \ RHS)
#         elseif param == "G"
#             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1),
#                                     λ, ones(nineq, nz), zeros(nineq,1),
#                                     ν, zeros(neq, nz), zeros(neq, 1))
#             push!(grads, LHS \ RHS)
#         elseif param == "h"
#             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
#                                     λ, zeros(nineq, nz), ones(nineq,1),
#                                     ν, zeros(neq, nz), zeros(neq, 1))
#             push!(grads, LHS \ RHS)
#         elseif param == "A"
#             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
#                                     λ, zeros(nineq, nz), zeros(nineq,1),
#                                     ν, ones(neq, nz), zeros(neq, 1))
#             push!(grads, LHS \ RHS)
#         elseif param == "b"
#             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
#                                     λ, zeros(nineq, nz), zeros(nineq,1),
#                                     ν, zeros(neq, nz), ones(neq, 1))
#             push!(grads, LHS \ RHS)
#         else
#             push!(grads, [])
#         end
#     end
#     return grads
# end

"""
    Method to differentiate optimal solution `z` and return
    product of jacobian matrices (`dz / dQ`, `dz / dq`, etc) with 
    the backward pass vector `dl / dz`

    The method computes the product of 
    1. jacobian of problem solution `z*` with respect to 
        problem parameters `params` recieved as method arguments
    2. a backward pass vector `dl / dz`, where `l` can be a loss function

    Note that does not returns the actual jacobians

    For more info refer eqn(7) and eqn(8) of https://arxiv.org/pdf/1703.00443.pdf
"""
function backward!(model::Optimizer, params::Array{String}, dl_dz::Array{Float64})
    Q, q, G, h, A, b, nz, var_idx, nineq, ineq_con_idx, neq, eq_con_idx = get_problem_data(model.optimizer)

    z = model.primal_optimal

    # separate λ, ν
    λ = filter(con -> !is_equality(MOI.get(model.optimizer, MOI.ConstraintSet(), con)), model.con_idx)
    ν = filter(con -> is_equality(MOI.get(model.optimizer, MOI.ConstraintSet(), con)), model.con_idx)

    λ = [MOI.get(model.optimizer, MOI.ConstraintDual(), con) for con in λ]
    ν = [MOI.get(model.optimizer, MOI.ConstraintDual(), con) for con in ν]

    grads = []
    LHS = create_LHS_matrix(z, λ, Q, G, h, A)
    RHS = [dl_dz'; zeros(neq+nineq,1)]

    partial_grads = -(LHS \ RHS)

    dz = partial_grads[1:nz]
    if nineq > 0
        dλ = partial_grads[nz+1:nz+nineq]
    end
    if neq > 0
        dν = partial_grads[nz+nineq+1:nz+nineq+neq]
    end
    
    for param in params
        if param == "Q"
            push!(grads, 0.5 * (dz * z' + z * dz'))
        elseif param == "q"
            push!(grads, dz)
        elseif param == "G"
            push!(grads, Diagonal(λ) * dλ * z' - λ * dz')
        elseif param == "h"
            push!(grads, -Diagonal(λ) * dλ)
        elseif param == "A"
            push!(grads, dν * z' - ν * dz')
        elseif param == "b"
            push!(grads, -dν)
        else
            push!(grads, [])
        end
    end
    return grads
end