"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note 
Currently supports differentiating linear and quadratic programs only.
"""

const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex

# use `<:SUPPORTED_OBJECTIVES` if you need to specify `Type` - specified by F/S
# use `::SUPPORTED_OBJECTIVES` if you need to specify a variable/object of that `Type` - specified by f/s

const SUPPORTED_OBJECTIVES = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64}
}

const SUPPORTED_SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64}, 
    MOI.EqualTo{Float64}
}

const SUPPORTED_SCALAR_FUNCTIONS = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64}
}

function diff_optimizer(optimizer_constructor)::Optimizer 
    return Optimizer(MOI.instantiate(optimizer_constructor, with_bridge_type=Float64))
end


mutable struct Optimizer{OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    primal_optimal::Array{Float64}  # solution
    dual_optimal::Array{Float64}  
    var_idx::Vector{VI}
    con_idx::Vector{CI}

    function Optimizer(optimizer_constructor::OT) where {OT <: MOI.ModelLike}
        new{OT}(
            optimizer_constructor,
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

function MOI.add_constraint(model::Optimizer, f::SUPPORTED_SCALAR_FUNCTIONS, s::SUPPORTED_SCALAR_SETS)
    ci = MOI.add_constraint(model.optimizer, f, s)
    push!(model.con_idx, ci)
    return ci
end

# TODO: support more sets here
function MOI.add_constraint(model::Optimizer, vf::MOI.VectorAffineFunction{Float64}, s::MOI.Zeros)
    ci = MOI.add_constraint(model.optimizer, vf, s)
    push!(model.con_idx, ci)   # adding it as a whole here; need to define a method to extract matrices
    return ci
end

function MOI.add_constraints(model::Optimizer, f::Vector{F}, s::Vector{S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return CI[MOI.add_constraint(model, f[i], s[i]) for i in 1:size(f)[1]]
end


function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{<: SUPPORTED_OBJECTIVES}, f::SUPPORTED_OBJECTIVES)
    MOI.set(model.optimizer, attr, f)
end


function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    return MOI.set(model.optimizer, attr, sense)
end


function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.optimizer, attr)
end


function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::MOI.ConstraintIndex{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::MOI.ConstraintIndex{F, S}, s::S) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.set(model.optimizer,attr,ci,s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::MOI.ConstraintIndex{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::MOI.ConstraintIndex{F, S}, f::F) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.set(model.optimizer,attr,ci,f)
end


function MOI.optimize!(model::Optimizer)
    MOI.optimize!(model.optimizer)

    # do not fail. interferes with MOI.Tests.linear12test
    if MOI.get(model.optimizer, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
        # save the solution
        model.primal_optimal = MOI.get(model.optimizer, MOI.VariablePrimal(), model.var_idx)
        model.dual_optimal = MOI.get(model.optimizer, MOI.ConstraintDual(), model.con_idx)
    else
        @warn "problem status: ", MOI.get(model.optimizer, MOI.TerminationStatus())
    end
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

# `MOI.supports` methods

function MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})
    return true
end

function MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}})
    return true
end

function MOI.supports(::Optimizer, ::MOI.AbstractModelAttribute)
    return true
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction)
    return false
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<: SUPPORTED_OBJECTIVES})
    return true
end

MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{<: SUPPORTED_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{<: SUPPORTED_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorAffineFunction{Float64}}, ::Type{MOI.Zeros}) = true


MOI.get(model::Optimizer, ::MOI.SolveTime) = model.optimizer.solve_time

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.optimizer)
    empty!(model.primal_optimal)
    empty!(model.dual_optimal)
    empty!(model.var_idx)
    empty!(model.con_idx)
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.optimizer) &&
           isempty(model.primal_optimal) &&
           isempty(model.dual_optimal) &&
           isempty(model.var_idx) &&
           isempty(model.con_idx)
end

# TODO: this'll be needed if we specify params using Variable attributes
MOIU.supports_default_copy_to(model::Optimizer, copy_names::Bool) = !copy_names

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOIU.default_copy_to(model.optimizer, src, copy_names)
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    return MOI.get(model.optimizer, MOI.TerminationStatus())
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::MOI.VariableIndex, value::Float64)
    MOI.set(model.optimizer, MOI.VariablePrimalStart(), vi, value)
end

function MOI.supports(model::Optimizer, ::MOI.VariablePrimalStart,
                      ::Type{MOI.VariableIndex})
    return MOI.supports(model.optimizer, MOI.VariablePrimalStart(), MOI.VariableIndex)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model.optimizer, attr)
    return MOI.get(model.optimizer, attr, vi)
end

function MOI.add_constraint(model::Optimizer, f::MOI.SingleVariable, s::SUPPORTED_SCALAR_SETS)
    ci = MOI.add_constraint(model.optimizer, f, s)
    push!(model.con_idx, ci)
    return ci
end

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F <: MOI.AbstractScalarFunction, S <: SUPPORTED_SCALAR_SETS}
    filter!(e -> e≠ci, model.con_idx)
    MOI.delete(model.optimizer, ci) 
end

function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
    ci::CI{F,S}) where {F <: MOI.AbstractScalarFunction, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci)
end

function MOI.is_valid(model::Optimizer, v::VI)
    return v in model.var_idx
end

function MOI.is_valid(model::Optimizer, con::CI)
    return con in model.con_idx
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual, ci::CI{F,S}) where {F <: MOI.AbstractScalarFunction, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintDual(), ci)
end

# helper method to check if a constraint contains a Variable
function _constraint_contains(model::Optimizer, v::VI, ci::CI{F,S}) where {F <: MOI.AbstractScalarFunction, S <: SUPPORTED_SCALAR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    if typeof(func) == MOI.SingleVariable
        return v == func.variable
    elseif typeof(func) == MOI.ScalarAffineFunction{Float64}
        for term in func.terms
            if term.variable_index == v
                return true
            end
        end
        return false
    end
    return false   # default
end

function MOI.delete(model::Optimizer, v::VI)
    # remove those constraints that depend on this Variable
    filter!(ci -> !_constraint_contains(model, v, ci), model.con_idx)

    # delete in inner solver 
    MOI.delete(model.optimizer, v) 
end
