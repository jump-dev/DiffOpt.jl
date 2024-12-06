# Copyright (c) 2020: Andrew Rosemberg and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# NonLinearProgram.jl
module NonLinearProgram

import DiffOpt
import JuMP
import MathOptInterface as MOI
using SparseArrays

include("nlp_utilities.jl")

Base.@kwdef struct Cache
    primal_vars::Vector{Float64}
    params::Vector{Float64}
    evaluator
    cons
end

Base.@kwdef struct ForwCache
    Δs::Matrix{Float64}
    sp::Vector{Float64}
end

Base.@kwdef struct ReverseCache
    Δs_T::Matrix{Float64}
    sp_T::Vector{Float64}
end

"""
    DiffOpt.NonLinearProgram.Model <: DiffOpt.AbstractModel

Model to differentiate nonlinear programs.

The forward differentiation computes the Jacobian product for selected variables
with respect to specified parameters.

The reverse differentiation computes the Jacobian transpose product for dual and primal
variables with respect to the parameters.

# Key Components
- Forward differentiation: Partial derivatives of variables of interest.
- Reverse differentiation: Transpose Jacobian computation.

"""
mutable struct Model <: DiffOpt.AbstractModel
    model::JuMP.Model          # JuMP optimization model
    cache::Union{Nothing, Cache} # Caches to hold evaluator and constraints
    forw_grad_cache::Union{Nothing, ForwCache} # Cache for forward sensitivity results
    back_grad_cache::Union{Nothing, ReverseCache} # Cache for reverse sensitivity results
    diff_time::Float64
end

function Model(model::JuMP.Model)
    return Model(
        model,
        nothing,
        nothing,
        nothing,
        NaN,
    )
end

function MOI.is_empty(model::Model)
    return model.cache === nothing
end

function MOI.empty!(model::Model)
    model.cache = nothing
    model.forw_grad_cache = nothing
    model.back_grad_cache = nothing
    model.diff_time = NaN
    return
end

MOI.get(model::Model, ::DiffOpt.DifferentiateTimeSec) = model.diff_time

function _cache_evaluator!(model::Model, primal_vars, params)
    if model.cache !== nothing
        return model.cache
    end
    evaluator, cons = create_evaluator(model.model; x=[primal_vars; params])
    model.cache = Cache(primal_vars=primal_vars, params=params, evaluator=evaluator, cons=cons)
    return model.cache
end

function DiffOpt.forward_differentiate!(model::Model; focus_vars, focus_duals, params)
    model.diff_time = @elapsed begin
        # Retrieve primal variables and cache the evaluator
        primal_vars = all_primal_vars(model.model)
        vars_idx = [findall(x -> x == i, primal_vars)[1] for i in focus_vars]
        
        cache = _cache_evaluator!(model, primal_vars, params)
        
        dual_index = [findall(x -> x == i, cache.cons)[1] for i in focus_duals]
        leq_locations, geq_locations = find_inequealities(cache.cons)
        num_ineq = length(leq_locations) + length(geq_locations)
        num_primal = length(primal_vars)
        
        # Compute sensitivities
        Δs, sp = compute_sensitivity(cache.evaluator, cache.cons, Δp; primal_vars=primal_vars, params=params)
        Δs_focus = Δs[vars_idx, :]
        Δs_duals_focus = Δs[num_primal + num_ineq + dual_index, :]
        
        model.forw_grad_cache = ForwCache(Δs=[Δs_focus; Δs_duals_focus], sp=sp)
    end
    return nothing
end

function DiffOpt.reverse_differentiate!(model::Model)
    model.diff_time = @elapsed begin
        # Not implemented, placeholder for reverse sensitivity logic
        # Use transpose Jacobian logic based on evaluator and derivatives
        throw(NotImplementedError("Reverse differentiation not yet implemented for NonLinearProgram"))
    end
    return nothing
end

function MOI.get(
    model::Model,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    if model.forw_grad_cache === nothing
        error("Forward differentiation has not been performed yet.")
    end
    Δs = model.forw_grad_cache.Δs
    return Δs[vi.value]
end

end # module NonLinearProgram


# module NonLinearProgram

# using JuMP
# using MathOptInterface
# import MathOptInterface: ConstraintSet, CanonicalConstraintFunction
# using SparseArrays
# using LinearAlgebra
# import JuMP.index
# include("nlp_utilities.jl")

# Base.@kwdef struct Cache
#     K::SparseArrays.SparseMatrixCSC{Float64,Int}
#     evaluator::MOI.Nonlinear.Model
#     rows::Vector{ConstraintRef}
# end

# # TODO: What is the forward cache when we want the sentsitivities wrt primal,dual,slack and bound dual variables?
# Base.@kwdef struct ForwCache
#     du::Vector{Float64}
#     dv::Vector{Float64}
#     dw::Vector{Float64}
# end

# # TODO: What is the reverse cache when we want the sentsitivities wrt primal,dual,slack and bound dual variables?
# Base.@kwdef struct ReverseCache
#     g::Vector{Float64}
#     πz::Vector{Float64}
# end

# # TODO: What models are supported?
# function MOI.supports(
#     ::MOI.Utilities.GenericModel{T},
#     ::MOI.ObjectiveFunction{F},
# ) where {T<:MOI.Utilities.ModelLike,F<:MOI.AbstractFunction}
#     return F === MOI.ScalarNonlinearFunction{T}
# end

# """
#     Diffopt.NonLinearProgram.Model <: DiffOpt.AbstractModel

# Model to differentiate conic programs.

# The forward differentiation computes the product of the derivative (Jacobian) at
# the non-linear program parameters `p`, to the perturbations `dp`.

# The reverse differentiation computes the product of the transpose of the
# derivative (Jacobian) at the non-linear program parameters `p`, to the
# perturbations `dx`, `dy`, `ds`, `dv`.

# For theoretical background, refer to XXX.
# """
# mutable struct Model <: DiffOpt.AbstractModel
#     # storage for problem data in matrix form
#     model::Form{Float64}
#     # includes maps from matrix indices to problem data held in `optimizer`
#     # also includes KKT matrices
#     # also includes the solution
#     gradient_cache::Union{Nothing,Cache}

#     # caches for sensitivity output
#     # result from solving KKT/residualmap linear systems
#     # this allows keeping the same `gradient_cache`
#     # if only sensitivy input changes
#     forw_grad_cache::Union{Nothing,ForwCache}
#     back_grad_cache::Union{Nothing,ReverseCache}

#     # sensitivity input cache using MOI like sparse format
#     input_cache::DiffOpt.InputCache

#     x::Vector{Float64} # Primal
#     s::Vector{Float64} # Slack
#     y::Vector{Float64} # Dual
#     v::Vector{Float64} # Bound Dual
#     diff_time::Float64
# end

# function Model()
#     return Model(
#         Form{Float64}(),
#         nothing,
#         nothing,
#         nothing,
#         DiffOpt.InputCache(),
#         Float64[],
#         Float64[],
#         Float64[],
#         Float64[],
#         NaN,
#     )
# end

# function MOI.is_empty(model::Model)
#     return MOI.is_empty(model.model)
# end

# function MOI.empty!(model::Model)
#     MOI.empty!(model.model)
#     model.gradient_cache = nothing
#     model.forw_grad_cache = nothing
#     model.back_grad_cache = nothing
#     empty!(model.input_cache)
#     empty!(model.x)
#     empty!(model.s)
#     empty!(model.y)
#     model.diff_time = NaN
#     return
# end

# MOI.get(model::Model, ::DiffOpt.DifferentiateTimeSec) = model.diff_time

# # TODO: what constraints are supported?
# function MOI.supports_constraint(
#     model::Model,
#     F::Type{MOI.VectorAffineFunction{Float64}},
#     ::Type{S},
# ) where {S<:MOI.AbstractVectorSet}
#     if DiffOpt.add_set_types(model.model.constraints.sets, S)
#         push!(model.model.constraints.caches, Tuple{F,S}[])
#         push!(model.model.constraints.are_indices_mapped, BitSet())
#     end
#     return MOI.supports_constraint(model.model, F, S)
# end

# # TODO: what is this?
# function MOI.set(
#     model::Model,
#     ::MOI.ConstraintPrimalStart,
#     ci::MOI.ConstraintIndex,
#     value,
# )
#     MOI.throw_if_not_valid(model, ci)
#     return DiffOpt._enlarge_set(
#         model.s,
#         MOI.Utilities.rows(model.model.constraints, ci),
#         value,
#     )
# end

# # TODO: what is this?
# function MOI.set(
#     model::Model,
#     ::MOI.ConstraintDualStart,
#     ci::MOI.ConstraintIndex,
#     value,
# )
#     MOI.throw_if_not_valid(model, ci)
#     return DiffOpt._enlarge_set(
#         model.y,
#         MOI.Utilities.rows(model.model.constraints, ci),
#         value,
#     )
# end

# function _gradient_cache(model::Model)
#     if model.gradient_cache !== nothing
#         return model.gradient_cache
#     end

#     evaluator, cons = create_evaluator(model; x=[primal_vars; params])

#     model.gradient_cache =
#         Cache(; M = M, vp = vp, Dπv = Dπv, A = A, b = b, c = c)

#     return model.gradient_cache
# end

# function DiffOpt.forward_differentiate!(model::Model)
#     model.diff_time = @elapsed begin
#         gradient_cache = _gradient_cache(model)
#         M = gradient_cache.M
#         vp = gradient_cache.vp
#         Dπv = gradient_cache.Dπv
#         x = model.x
#         y = model.y
#         s = model.s
#         A = gradient_cache.A
#         b = gradient_cache.b
#         c = gradient_cache.c

#         objective_function = DiffOpt._convert(
#             MOI.ScalarAffineFunction{Float64},
#             model.input_cache.objective,
#         )
#         sparse_array_obj = DiffOpt.sparse_array_representation(
#             objective_function,
#             length(c),
#         )
#         dc = sparse_array_obj.terms

#         db = zeros(length(b))
#         DiffOpt._fill(
#             S -> false,
#             gradient_cache,
#             model.input_cache,
#             model.model.constraints.sets,
#             db,
#         )
#         (lines, cols) = size(A)
#         nz = SparseArrays.nnz(A)
#         dAi = zeros(Int, 0)
#         dAj = zeros(Int, 0)
#         dAv = zeros(Float64, 0)
#         sizehint!(dAi, nz)
#         sizehint!(dAj, nz)
#         sizehint!(dAv, nz)
#         DiffOpt._fill(
#             S -> false,
#             gradient_cache,
#             model.input_cache,
#             model.model.constraints.sets,
#             dAi,
#             dAj,
#             dAv,
#         )
#         dA = SparseArrays.sparse(dAi, dAj, dAv, lines, cols)

#         m = size(A, 1)
#         n = size(A, 2)
#         N = m + n + 1
#         # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
#         (u, v, w) = (x, y - s, 1.0)

#         # g = dQ * Π(z/|w|) = dQ * [u, vp, 1.0]
#         RHS = [
#             dA' * vp + dc
#             -dA * u + db
#             -LinearAlgebra.dot(dc, u) - LinearAlgebra.dot(db, vp)
#         ]

#         dz = if LinearAlgebra.norm(RHS) <= 1e-400 # TODO: parametrize or remove
#             RHS .= 0 # because M is square
#         else
#             IterativeSolvers.lsqr(M, RHS)
#         end

#         du, dv, dw = dz[1:n], dz[n+1:n+m], dz[n+m+1]
#         model.forw_grad_cache = ForwCache(du, dv, [dw])
#     end
#     return nothing
#     # dx = du - x * dw
#     # dy = Dπv * dv - y * dw
#     # ds = Dπv * dv - dv - s * dw
#     # return -dx, -dy, -ds
# end

# function DiffOpt.reverse_differentiate!(model::Model)
#     model.diff_time = @elapsed begin
#         gradient_cache = _gradient_cache(model)
#         M = gradient_cache.M
#         vp = gradient_cache.vp
#         Dπv = gradient_cache.Dπv
#         x = model.x
#         y = model.y
#         s = model.s
#         A = gradient_cache.A
#         b = gradient_cache.b
#         c = gradient_cache.c

#         dx = zeros(length(c))
#         for (vi, value) in model.input_cache.dx
#             dx[vi.value] = value
#         end
#         dy = zeros(length(b))
#         ds = zeros(length(b))

#         m = size(A, 1)
#         n = size(A, 2)
#         N = m + n + 1
#         # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
#         (u, v, w) = (x, y - s, 1.0)

#         # dz = D \phi (z)^T (dx,dy,dz)
#         dz = [
#             dx
#             Dπv' * (dy + ds) - ds
#             -x' * dx - y' * dy - s' * ds
#         ]

#         g = if LinearAlgebra.norm(dz) <= 1e-4 # TODO: parametrize or remove
#             dz .= 0 # because M is square
#         else
#             IterativeSolvers.lsqr(M, dz)
#         end

#         πz = [
#             u
#             vp
#             1.0
#         ]

#         # TODO: very important
#         # contrast with:
#         # http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-109.pdf
#         # pg 97, cap 7.4.2

#         model.back_grad_cache = ReverseCache(g, πz)
#     end
#     return nothing
#     # dQ = - g * πz'
#     # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
#     # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
#     # dc = - dQ[1:n, end] + dQ[end, 1:n]'
#     # return dA, db, dc
# end

# function MOI.get(model::Model, ::DiffOpt.ReverseObjectiveFunction)
#     g = model.back_grad_cache.g
#     πz = model.back_grad_cache.πz
#     dc = DiffOpt.lazy_combination(-, πz, g, length(g), eachindex(model.x))
#     return DiffOpt.VectorScalarAffineFunction(dc, 0.0)
# end

# function MOI.get(
#     model::Model,
#     ::DiffOpt.ForwardVariablePrimal,
#     vi::MOI.VariableIndex,
# )
#     i = vi.value
#     du = model.forw_grad_cache.du
#     dw = model.forw_grad_cache.dw
#     return -(du[i] - model.x[i] * dw[])
# end

# function DiffOpt._get_db(
#     model::Model,
#     ci::MOI.ConstraintIndex{F,S},
# ) where {F<:MOI.AbstractVectorFunction,S}
#     i = MOI.Utilities.rows(model.model.constraints, ci) # vector
#     # i = ci.value
#     n = length(model.x) # columns in A
#     # Since `b` in https://arxiv.org/pdf/1904.09043.pdf is the constant in the right-hand side and
#     # `b` in MOI is the constant on the left-hand side, we have the opposite sign here
#     # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
#     g = model.back_grad_cache.g
#     πz = model.back_grad_cache.πz
#     # `g[end] * πz[n .+ i] - πz[end] * g[n .+ i]`
#     return DiffOpt.lazy_combination(-, πz, g, length(g), n .+ i)
# end

# function DiffOpt._get_dA(
#     model::Model,
#     ci::MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
# )
#     i = MOI.Utilities.rows(model.model.constraints, ci) # vector
#     # i = ci.value
#     n = length(model.x) # columns in A
#     m = length(model.y) # lines in A
#     # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
#     g = model.back_grad_cache.g
#     πz = model.back_grad_cache.πz
#     #return DiffOpt.lazy_combination(-, g, πz, n .+ i, 1:n)
#     return g[n.+i] * πz[1:n]' - πz[n.+i] * g[1:n]'
# end

# function MOI.get(
#     model::Model,
#     attr::MOI.ConstraintFunction,
#     ci::MOI.ConstraintIndex,
# )
#     return MOI.get(model.model, attr, ci)
# end

# end
