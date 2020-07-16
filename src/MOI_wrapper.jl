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
    MOI.EqualTo{Float64},
    MOI.Interval{Float64}
}

const SUPPORTED_SCALAR_FUNCTIONS = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64}
}

const SUPPORTED_VECTOR_FUNCTIONS = Union{
    MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
}

const SUPPORTED_VECTOR_SETS = Union{
    MOI.Zeros, 
    MOI.Nonpositives,
    MOI.Nonnegatives,
    MOI.SecondOrderCone
}


function diff_optimizer(optimizer_constructor)::Optimizer 
    return Optimizer(MOI.instantiate(optimizer_constructor, with_bridge_type=Float64))
end


mutable struct Optimizer{OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    primal_optimal::Vector{Float64}  # solution
    dual_optimal::Vector{Union{Vector{Float64}, Float64}}  # refer - https://github.com/AKS1996/DiffOpt.jl/issues/21#issuecomment-651130485
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

function MOI.add_constraint(model::Optimizer, vf::SUPPORTED_VECTOR_FUNCTIONS, s::SUPPORTED_VECTOR_SETS)
    ci = MOI.add_constraint(model.optimizer, vf, s)
    push!(model.con_idx, ci)   # adding it as a whole here; need to define a method to extract matrices
    return ci
end

function MOI.add_constraints(model::Optimizer, f::Vector{F}, s::Vector{S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return CI{F, S}[MOI.add_constraint(model, f[i], s[i]) for i in 1:size(f)[1]]
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

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}, s::S) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.set(model.optimizer,attr,ci,s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.set(model.optimizer,attr,ci,f)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}, s::S) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.set(model.optimizer,attr,ci,s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.set(model.optimizer, attr, ci, f)
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

MOI.supports_constraint(::Optimizer, ::Type{<: SUPPORTED_SCALAR_FUNCTIONS}, ::Type{<: SUPPORTED_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{<: SUPPORTED_VECTOR_FUNCTIONS}, ::Type{<: SUPPORTED_VECTOR_SETS}) = true
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{CI{F, S}}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS} = true
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{CI{F, S}}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS} = true

MOI.get(model::Optimizer, attr::MOI.SolveTime) = MOI.get(model.optimizer, attr)

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

# now supports name too
MOIU.supports_default_copy_to(model::Optimizer, copy_names::Bool) = true #!copy_names

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOIU.default_copy_to(model.optimizer, src, copy_names)
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    return MOI.get(model.optimizer, MOI.TerminationStatus())
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::VI, value::Float64)
    MOI.set(model.optimizer, MOI.VariablePrimalStart(), vi, value)
end

function MOI.supports(model::Optimizer, ::MOI.VariablePrimalStart,
                      ::Type{MOI.VariableIndex})
    return MOI.supports(model.optimizer, MOI.VariablePrimalStart(), MOI.VariableIndex)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(model.optimizer, attr)
    return MOI.get(model.optimizer, attr, vi)
end

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    filter!((≠ci), model.con_idx)
    MOI.delete(model.optimizer, ci) 
end

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    filter!((≠ci), model.con_idx)
    MOI.delete(model.optimizer, ci) 
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.is_valid(model::Optimizer, v::VI)
    return v in model.var_idx
end

function MOI.is_valid(model::Optimizer, con::CI)
    return con in model.con_idx
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintBasisStatus, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintBasisStatus(), ci)
end

# helper methods to check if a constraint contains a Variable
function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.SingleVariable, S}) where {S <: SUPPORTED_SCALAR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return v == func.variable
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.ScalarAffineFunction{Float64}, S}) where {S <: SUPPORTED_SCALAR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return any(term -> v == term.variable_index, func.terms)
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VectorOfVariables, S}) where {S <: SUPPORTED_VECTOR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return v in func.variables
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VectorAffineFunction{Float64}, S}) where {S <: SUPPORTED_VECTOR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return any(term -> v == term.scalar_term.variable_index, func.terms)
end


function MOI.delete(model::Optimizer, v::VI)
    # remove those constraints that depend on this Variable
    filter!(ci -> !_constraint_contains(model, v, ci), model.con_idx)

    # delete in inner solver 
    MOI.delete(model.optimizer, v) 

    # delete from var_idx
    filter!(e -> e ≠ v, model.var_idx)
end

# for array deletion
function MOI.delete(model::Optimizer, indices::Vector{VI})
    for i in indices
        MOI.delete(model, i)
    end
end

function MOI.modify(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.AbstractFunctionModification
)
    MOI.modify(
        model.optimizer, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        chg
    )
end

function MOI.modify(model::Optimizer, ci::CI{F, S}, chg::MOI.AbstractFunctionModification) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    MOI.modify(model.optimizer, ci, chg)
end

function MOI.modify(model::Optimizer, ci::CI{F, S}, chg::MOI.AbstractFunctionModification) where {F<:MOI.VectorAffineFunction{Float64}, S <: SUPPORTED_VECTOR_SETS}
    MOI.modify(model.optimizer, ci, chg)
end

"""
    Method to differentiate optimal solution `x`, `y`, `s`
"""
function backward_conic!(model::Optimizer, dA::Array{Float64,2}, db::Array{Float64}, dc::Array{Float64})
    if MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
        # @assert MOI.get(model.optimizer, MOI.SolverName()) == "SCS"
        # MOIU.load_variables(model.optimizer.model.optimizer, nz)
        
        # for con in model.con_idx
        #     func = MOI.get(model.optimizer, MOI.ConstraintFunction(), con)
        #     set = MOI.get(model.optimizer, MOI.ConstraintSet(), con)
        #     MOIU.load_constraint(model.optimizer.model.optimizer, con, func, set)
        # end
        
        # # now SCS>data shud be allocated
        # A = sparse(
        #     model.optimizer.model.optimizer.data.I, 
        #     model.optimizer.model.optimizer.data.J, 
        #     model.optimizer.model.optimizer.data.V 
        # )
        # b = model.optimizer.model.optimizer.data.b 
        # c = model.optimizer.model.optimizer.data.c 

        # TODO: remove this static hack
        A = [
            0.0   0.0   1.0;
            0.0  -1.0   0.0;
            0.0   0.0  -1.0;
            -1.0  0.0   0.0;
            0.0  -1.0   0.0;
        ]
        b = [1.0; -0.707107; 0.0; 0.0; 0.0]
        c = [1.0; 0.0; 0.0]

        # get x,y,s
        x = model.primal_optimal
        s = MOI.get(model, MOI.ConstraintPrimal(), model.con_idx)
        y = model.dual_optimal

        # pre-compute quantities for the derivative
        m, n = size(A)
        N = m + n + 1
        cones = MOI.get(model, MOI.ConstraintSet(), model.con_idx)
        z = (x, y - s, [1.0])
        u, v, w = z

        Q = [
            zeros(n,n)   A'           c;
            -A           zeros(m,m)   b;
            -c'          -b'          zeros(1,1)
        ]

        Dπv = Dπ(cones, v)

        M = ((Q - Matrix{Float64}(I, size(Q))) * BlockDiagonal([Matrix{Float64}(I, length(u), length(u)), Dπv, Matrix{Float64}(I,1,1)])) + Matrix{Float64}(I, size(Q))

        πz = vcat([u, π(cones, v), max.(w,0.0)]...)

        dQ = [
            zeros(n,n)    dA'          dc;
            -dA           zeros(m,m)   db;
            -dc'         -db'          zeros(1,1)
        ]

        RHS = dQ * πz
        if norm(RHS) <= 1e-4
            dz = zeros(Float64, size(RHS))
        else
            dz = lsqr(M, RHS)
        end

        du, dv, dw = dz[1:n], dz[n+1:n+m], dz[n+m+1]
        dx = du - x * dw
        dy = Dπv * dv - vcat(y...) * dw
        ds = Dπv * dv - dv - vcat(s...) * dw
        return -dx, -dy, -ds
    else
        @error "problem status: ", MOI.get(model.optimizer, MOI.TerminationStatus())
    end    
end

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true

function MOI.get(model::Optimizer, ::MOI.VariableName, v::VI)
    return MOI.get(model.optimizer, MOI.VariableName(), v)
end

function MOI.set(model::Optimizer, ::MOI.VariableName, v::VI, name::String)
    MOI.set(model.optimizer, MOI.VariableName(), v, name)
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    return MOI.get(model.optimizer, MOI.VariableIndex, name)
end

function MOI.set(model::Optimizer, ::MOI.ConstraintName, con::CI, name::String)
    MOI.set(model.optimizer, MOI.ConstraintName(), con, name)
end

function MOI.get(model::Optimizer, ::Type{MOI.ConstraintIndex}, name::String)
    return MOI.get(model.optimizer, MOI.ConstraintIndex, name)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, con::CI)
    return MOI.get(model.optimizer, MOI.ConstraintName(), con)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ::Type{CI{F, S}}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintName(), CI{F,S})
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ::Type{CI{VF, VS}}) where {VF<:SUPPORTED_VECTOR_FUNCTIONS, VS<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintName(), CI{VF,VS})
end

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    MOI.set(model.optimizer, MOI.Name(), name)
end

function MOI.get(model::Optimizer, ::Type{CI{F, S}}, name::String) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, CI{F,S}, name)
end

function MOI.get(model::Optimizer, ::Type{CI{VF, VS}}, name::String) where {VF<:SUPPORTED_VECTOR_FUNCTIONS, VS<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, CI{VF,VS}, name)
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Union{Real, Nothing})
    MOI.set(model.optimizer, MOI.TimeLimitSec(), value)
end
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return MOI.get(model.optimizer, MOI.TimeLimitSec())
end

MOI.supports(model::Optimizer, ::MOI.Silent) = MOI.supports(model.optimizer, MOI.Silent())

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    MOI.set(model.optimizer, MOI.Silent(), value)
end

function MOI.get(model::Optimizer, ::MOI.Silent)
    return MOI.get(model.optimizer, MOI.Silent())
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    return MOI.get(model.optimizer, MOI.SolverName())
end
