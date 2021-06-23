"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note
Currently supports differentiating linear and quadratic programs only.
"""


"""
    diff_optimizer(optimizer_constructor)::Optimizer

Creates a `DiffOpt.Optimizer`, which is an MOI layer with an internal optimizer
and other utility methods. Results (primal, dual and slack values) are obtained
by querying the internal optimizer instantiated using the
`optimizer_constructor`. These values are required for find jacobians with respect to problem data.

One define a differentiable model by using any solver of choice. Example:

```julia
julia> using DiffOpt, GLPK

julia> model = diff_optimizer(GLPK.Optimizer)
julia> model.add_variable(x)
julia> model.add_constraint(...)

julia> _backward_quad(model)  # for convex quadratic models

julia> _backward_quad(model)  # for convex conic models
```
"""
function diff_optimizer(optimizer_constructor)::Optimizer
    return Optimizer(MOI.instantiate(optimizer_constructor, with_bridge_type=Float64))
end

Base.@kwdef struct QPCache
    problem_data::Tuple{
        SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # Q, q
        SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # G, h
        SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # A, b
        Int, Vector{VI}, # nz, var_list
        Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}, # nineq_le, le_con_idx
        Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}, # nineq_ge, ge_con_idx
        Int, Vector{MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}}, # nineq_sv_le, le_con_sv_idx
        Int, Vector{MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}}, # nineq_sv_ge, ge_con_sv_idx
        Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}, # neq, eq_con_idx
        Int, Vector{MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}}, # neq_sv, eq_con_sv_idx
    }
    inequality_duals::Vector{Float64}
    equality_duals::Vector{Float64}
    var_primals::Vector{Float64}
    lhs::SparseMatrixCSC{Float64, Int}
    index_map::MOIU.IndexMap
end

const CONIC_FORM = MatOI.GeometricConicForm{
    Float64,
    MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing},
    Vector{Float64}}

Base.@kwdef struct ConicCache
    M::SparseMatrixCSC{Float64, Int}
    vp::Vector
    Dπv::BlockDiagonals.BlockDiagonal{Float64, Matrix{Float64}}
    xys::NTuple{3, Vector{Float64}}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    c::Vector{Float64}
    index_map::MOIU.IndexMap
    conic_form::CONIC_FORM
end

const CACHE_TYPE = Union{
    Nothing,
    QPCache,
    ConicCache,
}

Base.@kwdef struct QPForwBackCache
    dz::Vector{Float64}
    dλ::Vector{Float64}
    dν::Vector{Float64}
end
Base.@kwdef struct ConicForwCache
    du::Vector{Float64}
    dv::Vector{Float64}
    dw::Vector{Float64}
end
Base.@kwdef struct ConicBackCache
    g::Vector{Float64}
    πz::Vector{Float64}
end

const CACHE_FORW_TYPE = Union{
    Nothing,
    QPForwBackCache,
    ConicForwCache,
}
const CACHE_BACK_TYPE = Union{
    Nothing,
    QPForwBackCache,
    ConicBackCache,
}

const MOIDD = MOI.Utilities.DoubleDicts

Base.@kwdef mutable struct DiffInputCache
    dx::Dict{VI, Float64} = Dict{VI, Float64}()# dz for QP
    # ds
    # dy #= [d\lambda, d\nu] for QP
    # FIXME Would it be possible to have a DoubleDict where the value depends
    #       on the function type ? Here, we need to create two dicts to have
    #       concrete value types.
    # `scalar_constraints` and `vector_constraints` includes `A` and `b` for CPs
    # or `G` and `h` for QPs
    scalar_constraints::MOIDD.DoubleDict{MOI.ScalarAffineFunction{Float64}} = MOIDD.DoubleDict{MOI.ScalarAffineFunction{Float64}}() # also includes G for QPs
    vector_constraints::MOIDD.DoubleDict{MOI.VectorAffineFunction{Float64}} = MOIDD.DoubleDict{MOI.VectorAffineFunction{Float64}}() # also includes G for QPs
    objective::Union{Nothing,MOI.AbstractScalarFunction} = nothing
end

abstract type AbstractDiffAttribute end
Base.broadcastable(attribute::AbstractDiffAttribute) = Ref(attribute)

"""
    ForwardInObjective

A `MOI.AbstractModelAttribute` to set input data to forward differentiation, that
is, problem input data.
The possible values are any `MOI.AbstractScalarFunction`.
A `MOI.ScalarQuadraticFunction` can only be used in linearly constrained
quadratic models.

For instance, if the objective contains `θ * (x + 2y)`, for the purpose of
computinig the derivative with respect to `θ`, the following should be set:
```julia
fx = MOI.SingleVariable(x)
fy = MOI.SingleVariable(y)
MOI.set(model, DiffOpt.ForwardInObjective(), 1.0 * fx + 2.0 * fy)
```
where `x` and `y` are the relevant `MOI.VariableIndex`.
"""
struct ForwardInObjective <: MOI.AbstractModelAttribute end

"""
    ForwardInConstraint

A `MOI.AbstractConstraintAttribute` to set input data to forward differentiation, that
is, problem input data.

For instance, if the scalar constraint of index `ci` contains `θ * (x + 2y)`,
for the purpose of computinig the derivative with respect to `θ`, the following
should be set:
```julia
fx = MOI.SingleVariable(x)
fy = MOI.SingleVariable(y)
MOI.set(model, DiffOpt.ForwardInConstraint(), ci, 1.0 * fx + 2.0 * fy)
```
"""
struct ForwardInConstraint <: MOI.AbstractConstraintAttribute end


"""
    ForwardOutVariablePrimal

A `MOI.AbstractVariableAttribute` to get output data from forward
differentiation, that is, problem solution.

For instance, to get the tangent of the variable of index `vi` corresponding to
the tangents given to `ForwardInObjective` and `ForwardInConstraint`, do the
following:
```julia
MOI.get(model, DiffOpt.ForwardOutVariablePrimal(), vi)
```
"""
struct ForwardOutVariablePrimal <: MOI.AbstractVariableAttribute end
MOI.is_set_by_optimize(::ForwardOutVariablePrimal) = true

"""
    BackwardInVariablePrimal

A `MOI.AbstractVariableAttribute` to set input data to backward
differentiation, that is, problem solution.

For instance, to set the tangent of the variable of index `vi`, do the
following:
```julia
MOI.set(model, DiffOpt.BackwardInVariablePrimal(), x)
```
"""
struct BackwardInVariablePrimal <: MOI.AbstractVariableAttribute end

"""
    BackwardOut{T}

A AbstractDiffAttribute to get output data to backward differentiation, that
is, problem solution.
The solution data includes:
[`LinearObjective`](@ref), [`ConstraintConstant`](@ref),
[`ConstraintCoefficient`](@ref) and [`QuadraticObjective`](@ref).
The latter can only be used in linearly constrained quadratic models.

```julia
MOI.get(model, DiffOpt.BackwardOut{DiffOpt.LinearObjective}(), x)
```
"""
struct BackwardOut{T} <: AbstractDiffAttribute end

abstract type AbstractDiffInnerAttribute end

"""
    LinearObjective

An attribute to set input and get output differentials from forward and backward
differentiation related to the linear objective coefficient associated to an
`MOI.VariableIndex`.
"""
struct LinearObjective <: AbstractDiffInnerAttribute end #(var)

"""
    QuadraticObjective

An attribute to set input and get output differentials from forward and backward
differentiation related to the quadratic objective coefficient associated to a pair
of `MOI.VariableIndex`'s.
"""
struct QuadraticObjective <: AbstractDiffInnerAttribute end #(var, var)

"""
    ConstraintConstant

An attribute to set input and get output differentials from forward and backward
differentiation related to the constant term associated to a `MOI.ConstraintIndex`.
"""
struct ConstraintConstant <: AbstractDiffInnerAttribute end #(con)

"""
    ConstraintCoefficient

An attribute to set input and get output differentials from forward and backward
differentiation related to the linear coefficient associated to a pair:
`MOI.VariableIndex` and `MOI.ConstraintIndex`.
"""
struct ConstraintCoefficient <: AbstractDiffInnerAttribute end #(var, con)

mutable struct Optimizer{OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT

    # storage for problem data in matrix form
    # includes maps from matrix indices to problem data held in `optimizer`
    # also includes KKT matrices
    # also includes the solution
    gradient_cache::CACHE_TYPE

    # caches for sensitivity output
    # result from solving KKT/residualmap linear systems
    # this allows keeping the same `gradient_cache`
    # if only sensitivy input changes
    forw_grad_cache::CACHE_FORW_TYPE
    back_grad_cache::CACHE_BACK_TYPE

    # sensitivity input cache using MOI like sparse format
    input_cache::DiffInputCache

    function Optimizer(optimizer_constructor::OT) where {OT <: MOI.ModelLike}
        new{OT}(
            optimizer_constructor,
            nothing,
            nothing,
            nothing,
            DiffInputCache(),
        )
    end
end

function MOI.get(model::Optimizer, ::ForwardOutVariablePrimal, vi::VI)
    return _get_dx(model, vi)
end
_get_dx(model::Optimizer, vi) = _get_dx(model.forw_grad_cache, model.gradient_cache, vi)
function _get_dx(cache::QPForwBackCache, g_cache::QPCache, vi)
    i = g_cache.index_map[vi].value
    return cache.dz[i]
end
function _get_dx(f_cache::ConicForwCache, g_cache::ConicCache, vi)
    i = g_cache.index_map[vi].value
    du = f_cache.du
    dw = f_cache.dw
    x = g_cache.xys[1]
    return - (du[i] - x[i] * dw[])
end

function MOI.get(model::Optimizer, ::BackwardInVariablePrimal, vi::VI)
    return get(model.input_cache.dx, vi, 0.0)
end
function MOI.set(model::Optimizer, ::BackwardInVariablePrimal, vi::VI, val)
    model.input_cache.dx[vi] = val
    return
end

function MOI.get(model::Optimizer, ::BackwardOut{LinearObjective}, vi::VI)
    return _get_dc(model, vi)
end
_get_dc(model::Optimizer, vi) = _get_dc(model.back_grad_cache, model.gradient_cache, vi)
function _get_dc(b_cache::ConicBackCache, g_cache::ConicCache, vi)
    i = g_cache.index_map[vi].value
    g = b_cache.g
    πz = b_cache.πz
    dQ_i_end = - g[i] * πz[end]
    dQ_end_i = - g[end] * πz[i]
    return - dQ_i_end + dQ_i_end
end
function _get_dc(b_cache::QPForwBackCache, g_cache::QPCache, vi)
    i = g_cache.index_map[vi].value
    dz = b_cache.dz
    return dz[i]
end

function MOI.get(model::Optimizer, ::ForwardInObjective)
    return model.input_cache.objective
end
function MOI.set(model::Optimizer, ::ForwardInObjective, objective)
    model.input_cache.objective = objective
    return
end

function MOI.get(model::Optimizer,
    ::BackwardOut{QuadraticObjective}, vi1::VI, vi2::VI)
    return _get_dQ(model, vi1, vi2)
end
_get_dQ(model::Optimizer, vi1, vi2) = _get_dQ(model.back_grad_cache, model.gradient_cache, vi1, vi2)
function _get_dQ(b_cache::ConicBackCache, g_cache::ConicCache, vi1, vi2)
    error("Quadratic function not availablein conic model differentiation")
end
function _get_dQ(b_cache::QPForwBackCache, g_cache::QPCache, vi1, vi2)
    i = g_cache.index_map[vi1].value
    j = g_cache.index_map[vi2].value
    z = g_cache.var_primals
    dz = b_cache.dz
    return 0.5 * (dz[i] * z[j] + z[i] * dz[j])
end

function MOI.get(model::Optimizer,
    ::BackwardOut{ConstraintConstant}, ci::CI{F,S}) where {F,S}
    return _get_db(model, ci)
end
_get_db(model::Optimizer, ci) = _get_db(model.back_grad_cache, model.gradient_cache, ci)
function _get_db(b_cache::ConicBackCache, g_cache::ConicCache, ci::CI{F,S}
) where {F<:MOI.AbstractVectorFunction,S}
    cf = g_cache.conic_form
    _ci = g_cache.index_map[ci]
    i = MatOI.rows(cf, _ci) # vector
    # i = g_cache.index_map[ci].value
    (x, _, _) = g_cache.xys
    n = length(x) # columns in A
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    g = b_cache.g
    πz = b_cache.πz
    dQ_ni_end = - g[n .+ i] * πz[end]
    dQ_end_ni = - g[end] * πz[n .+ i]
    return - dQ_ni_end + dQ_end_ni
end
function _get_db(b_cache::ConicBackCache, g_cache::ConicCache, ci::CI{F,S}
) where {F<:MOI.AbstractScalarFunction,S}
    i = g_cache.index_map[ci].value
    (x, _, _) = g_cache.xys
    n = length(x) # columns in A
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    g = b_cache.g
    πz = b_cache.πz
    dQ_ni_end = - g[n+i] * πz[end]
    dQ_end_ni = - g[end] * πz[n+i]
    return - dQ_ni_end + dQ_end_ni
end
function _get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S}
    i = g_cache.index_map[ci].value
    # dh = -Diagonal(λ) * dλ
    dλ = b_cache.dλ
    λ = g_cache.inequality_duals
    return - λ[i] * dλ[i]
end
function _get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S<:MOI.EqualTo}
    i = g_cache.index_map[ci].value
    dν = b_cache.dν
    return - dν[i]
end

function MOI.get(model::Optimizer,
    ::ForwardInConstraint, ci::CI{MOI.ScalarAffineFunction{T},S}
) where {T,S}
    return get(model.input_cache.scalar_constraints, ci, zero(MOI.ScalarAffineFunction{T}))
end
function MOI.get(model::Optimizer,
    ::ForwardInConstraint, ci::CI{MOI.VectorAffineFunction{T},S}
) where {T,S}
    func = get(model.input_cache.vector_constraints, ci, nothing)
    if func === nothing
        set = MOI.get(model, MOI.ConstraintSet(), ci)
        dim = MOI.dimension(set)
        return MOI.Utilities.zero_with_output_dimension(MOI.VectorAffineFunction{T}, dim)
    else
        return func
    end
end
function MOI.set(model::Optimizer,
    ::ForwardInConstraint,
    ci::CI{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    model.input_cache.scalar_constraints[ci] = func
    return
end
function MOI.set(model::Optimizer,
    ::ForwardInConstraint,
    ci::CI{MOI.VectorAffineFunction{T},S},
    func::MOI.VectorAffineFunction{T},
) where {T,S}
    model.input_cache.vector_constraints[ci] = func
    return
end

function MOI.get(model::Optimizer,
    ::BackwardOut{ConstraintCoefficient}, vi::VI, ci::CI{F,S}) where {F,S}
    return _get_dA(model, vi, ci)
end
_get_dA(model::Optimizer, vi, ci) = _get_dA(model.back_grad_cache, model.gradient_cache, vi, ci)
function _get_dA(b_cache::ConicBackCache, g_cache::ConicCache, vi, ci::CI{F,S}
) where {F<:MOI.AbstractScalarFunction,S}
    j = g_cache.index_map[vi].value
    i = g_cache.index_map[ci].value
    (x, y, _) = g_cache.xys
    n = length(x) # columns in A
    m = length(y) # lines in A
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    g = b_cache.g
    πz = b_cache.πz
    dQ_i_nj =  - g[i] * πz[n+j]
    dQ_nj_i =  - g[n+j] * πz[i]
    return - dQ_i_nj + dQ_nj_i
end
function _get_dA(b_cache::ConicBackCache, g_cache::ConicCache, vi, ci::CI{F,S}
) where {F<:MOI.AbstractVectorFunction,S}
    cf = g_cache.conic_form
    _ci = g_cache.index_map[ci]
    i = MatOI.rows(cf, _ci) # vector
    j = g_cache.index_map[vi].value
    # i = g_cache.index_map[ci].value
    (x, y, _) = g_cache.xys
    n = length(x) # columns in A
    m = length(y) # lines in A
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    g = b_cache.g
    πz = b_cache.πz
    dQ_i_nj =  - g[i] * πz[n+j]
    dQ_nj_i =  - g[n+j] * πz[i]
    return - dQ_i_nj .+ dQ_nj_i
end
# quadratic matrix indexes are split by type either == or (<=/>=)
function _get_dA(b_cache::QPForwBackCache, g_cache::QPCache, vi, ci::CI{F,S}
) where {F, S<:MOI.EqualTo}
    j = g_cache.index_map[vi].value
    i = g_cache.index_map[ci].value
    z = g_cache.var_primals
    dz = b_cache.dz
    ν = g_cache.equality_duals
    dν = b_cache.dν
    # this is the previously implemented
    return dν[i] * z[j] - ν[i] * dz[j]
    # from the paper, teh correct solution should be this
    # and thec correct fix is probably correcting the signs of the duals
    # since MOI standard is different from text book shadow prices
    # return dν[i] * z[j] + ν[i] * dz[j]
end
function _get_dA(b_cache::QPForwBackCache, g_cache::QPCache, vi, ci::CI{F,S}
) where {F, S}
    j = g_cache.index_map[vi].value
    i = g_cache.index_map[ci].value
    z = g_cache.var_primals
    dz = b_cache.dz
    λ = g_cache.inequality_duals
    dλ = b_cache.dλ
    # this is the previously implemented
    # return Diagonal(λ) * dλ * z' - λ * dz')
    return λ[i] * (dλ[i] * z[j]) - λ[i] * dz[j]
    # from the paper, the correct solution should be this
    # and the correct fix is probably correcting the signs of the duals
    # since MOI standard is different from text book shadow prices
    # return λ[i] * (dλ[i] * z[j] + λ[i] * dz[j])
end


function MOI.optimize!(model::Optimizer)
    model.gradient_cache = nothing
    MOI.optimize!(model.optimizer)

    # do not fail. interferes with MOI.Tests.linear12test
    if !in(MOI.get(model.optimizer, MOI.TerminationStatus()),  (MOI.LOCALLY_SOLVED, MOI.OPTIMAL))
        @warn "problem status: $(MOI.get(model.optimizer, MOI.TerminationStatus()))"
        return
    end

    return
end


const _QP_SET_TYPES = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    # MOI.Interval{Float64},
}

const _QP_FUNCTION_TYPES = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
}

const QP_OBJECTIVE_TYPES = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.SingleVariable,
}

"""
    backward(model::Optimizer)

Wrapper method for the backward pass.
This method will consider as input a currently solved problem and differentials
with respect to the solution set with the [`BackwardIn`](@ref) attribute.
The output problem data differentials can be queried with the
attribute [`BackwardOut`](@ref).
"""
function backward(model::Optimizer)
    if _qp_supported(model.optimizer)
        return _backward_quad(model)
    elseif !_is_qp_obj(model)
        return _backward_conic(model)
    else
        error("Non-supported model")
    end
end

"""
    forward(model::Optimizer)

Wrapper method for the forward pass.
This method will consider as input a currently solved problem and
differentials with respect to problem data set with
the [`ForwardInObjective`](@ref) and  [`ForwardInConstraint`](@ref) attributes.
The output solution differentials can be queried with the attribute
[`ForwardOutVariablePrimal`](@ref).
"""
function forward(model::Optimizer)
    if _qp_supported(model.optimizer)
        return _forward_quad(model)
    elseif !_is_qp_obj(model)
        return _forward_conic(model)
    else
        error("Non-supported model")
    end
end

function _is_qp_obj(model)
    MOI.get(model.optimizer, MOI.ObjectiveFunctionType()) <: MOI.ScalarQuadraticFunction{Float64}
end

_qp_supported(::Type{F}, ::Type{S}) where {F <: _QP_FUNCTION_TYPES, S <: _QP_SET_TYPES} = true
_qp_supported(::Type{F}, ::Type{S}) where {F, S} = false
function _qp_supported(model)
    con_types = MOI.get(model, MOI.ListOfConstraints())
    for (func, set) in con_types
        if !_qp_supported(func, set)
            return false
        end
    end
    return true
end

"""
    _backward_quad(model::Optimizer)

Method to differentiate optimal solution `z` and return
product of jacobian matrices (`dz / dQ`, `dz / dq`, etc) with
the backward pass vector `dl / dz`

The method computes the product of
1. jacobian of problem solution `z*` with respect to
    problem parameters set with the [`BackwardIn`](@ref)
2. a backward pass vector `dl / dz`, where `l` can be a loss function

Note that this method *does not returns* the actual jacobians.

For more info refer eqn(7) and eqn(8) of https://arxiv.org/pdf/1703.00443.pdf
"""
function _backward_quad(model::Optimizer)

    if model.gradient_cache === nothing
        build_quad_diff_cache!(model)
    end
    (
        Q, q, G, h, A, b, nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = model.gradient_cache.problem_data
    z = model.gradient_cache.var_primals
    λ = model.gradient_cache.inequality_duals
    ν = model.gradient_cache.equality_duals
    LHS = model.gradient_cache.lhs

    index_map = model.gradient_cache.index_map
    dl_dz = zeros(length(z))
    for (vi, val) in model.input_cache.dx
        inner_index = index_map[vi].value
        dl_dz[inner_index] = val
    end

    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    RHS = [dl_dz; zeros(neq + neq_sv + nineq_total)]

    partial_grads = if norm(Q) ≈ 0
        -lsqr(LHS, RHS)
    else
        -LHS \ RHS
    end

    dz = partial_grads[1:nz]
    dλ = partial_grads[nz+1:nz+nineq_total]
    dν = partial_grads[nz+nineq_total+1:nz+nineq_total+neq+neq_sv]

    model.back_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
    # dQ = 0.5 * (dz * z' + z * dz')
    # dq = dz
    # dG = Diagonal(λ) * (dλ * z' + λ * dz') # was: Diagonal(λ) * dλ * z' - λ * dz')
    # dh = -Diagonal(λ) * dλ
    # dA = dν * z'+ ν * dz' # was: dν * z' - ν * dz'
    # db = -dν
    # todo, check MOI signs for dA and dG
end

"""
    _forward_quad(model::Optimizer)
"""
function _forward_quad(model::Optimizer)
    if model.gradient_cache === nothing
        build_quad_diff_cache!(model)
    end
    (
        Q, q, G, h, A, b, nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = model.gradient_cache.problem_data
    z = model.gradient_cache.var_primals
    λ = model.gradient_cache.inequality_duals
    ν = model.gradient_cache.equality_duals
    LHS = model.gradient_cache.lhs
    index_map = model.gradient_cache.index_map

    objective_function = _convert(MOI.ScalarQuadraticFunction{Float64}, model.input_cache.objective)
    sparse_array_obj = sparse_array_representation(objective_function, LinearAlgebra.checksquare(Q), index_map)
    dQ = sparse_array_obj.quadratic_terms
    dq = sparse_array_obj.affine_terms

    db = zeros(length(b))
    _fill(isequal(MOI.EqualTo{Float64}), model, _QPForm(), db)

    dh = zeros(length(h))
    _fill(!isequal(MOI.EqualTo{Float64}), model, _QPForm(), dh)

    nz = nnz(A)
    (lines, cols) = size(A)
    dAi = zeros(Int, 0)
    dAj = zeros(Int, 0)
    dAv = zeros(Float64, 0)
    sizehint!(dAi, nz)
    sizehint!(dAj, nz)
    sizehint!(dAv, nz)
    _fill(isequal(MOI.EqualTo{Float64}), model, _QPForm(), dAi, dAj, dAv)
    dA = sparse(dAi, dAj, dAv, lines, cols)

    nz = nnz(G)
    (lines, cols) = size(G)
    dGi = zeros(Int, 0)
    dGj = zeros(Int, 0)
    dGv = zeros(Float64, 0)
    sizehint!(dGi, nz)
    sizehint!(dGj, nz)
    sizehint!(dGv, nz)
    _fill(!isequal(MOI.EqualTo{Float64}), model, _QPForm(), dGi, dGj, dGv)
    dG = sparse(dGi, dGj, dGv, lines, cols)


    RHS = [
        dQ * z + dq + dG' * λ + dA' * ν
        λ .* (dG * z) - λ .* dh
        dA * z - db
    ]

    partial_grads = if norm(Q) ≈ 0
        -lsqr(LHS, RHS)
    else
        -LHS \ RHS
    end

    nv = length(z)
    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    dz = partial_grads[1:nv]
    dλ = partial_grads[nv+1:nv+nineq_total]
    dν = partial_grads[nv+nineq_total+1:nv+nineq_total+neq+neq_sv]

    model.forw_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
end



"""
    π(v::Vector{Float64}, model::MOI.ModelLike, conic_form::MatOI.GeometricConicForm, index_map::MOIU.IndexMap)

Given a `model`, its `conic_form` and the `index_map` from the indices of
`model` to the indices of `conic_form`, find the projection of the vectors `v`
of length equal to the number of rows in the conic form onto the cartesian
product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function π(v::Vector{T}, model::MOI.ModelLike, conic_form::MatOI.GeometricConicForm, index_map::MOIU.IndexMap) where T
    return map_rows(model, conic_form, index_map, Flattened{T}()) do ci, r
        MOSD.projection_on_set(
            MOSD.DefaultDistance(),
            v[r],
            MOI.dual_set(MOI.get(model, MOI.ConstraintSet(), ci))
        )
    end
end


"""
    Dπ(v::Vector{Float64}, model, conic_form::MatOI.GeometricConicForm, index_map::MOIU.IndexMap)

Given a `model`, its `conic_form` and the `index_map` from the indices of
`model` to the indices of `conic_form`, find the gradient of the projection of
the vectors `v` of length equal to the number of rows in the conic form onto the
cartesian product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function Dπ(v::Vector{T}, model::MOI.ModelLike, conic_form::MatOI.GeometricConicForm, index_map::MOIU.IndexMap) where T
    return BlockDiagonals.BlockDiagonal(
        map_rows(model, conic_form, index_map, Nested{Matrix{T}}()) do ci, r
            MOSD.projection_gradient_on_set(
                MOSD.DefaultDistance(),
                v[r],
                MOI.dual_set(MOI.get(model, MOI.ConstraintSet(), ci)),
            )
        end
    )
end

# See the docstring of `map_rows`.
struct Nested{T} end
struct Flattened{T} end

# Store in `x` the values `y` corresponding to the rows `r` and the `k`th
# constraint.
function _assign_mapped!(x, y, r, k, ::Nested)
    x[k] = y
end
function _assign_mapped!(x, y, r, k, ::Flattened)
    x[r] = y
end

# Map the rows corresponding to `F`-in-`S` constraints and store it in `x`.
function _map_rows!(f::Function, x::Vector, model, conic_form::MatOI.GeometricConicForm, index_map::MOIU.DoubleDicts.IndexWithType{F, S}, map_mode, k) where {F, S}
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
        r = MatOI.rows(conic_form, index_map[ci])
        k += 1
        _assign_mapped!(x, f(ci, r), r, k, map_mode)
    end
    return k
end

# Allocate a vector for storing the output of `map_rows`.
_allocate_rows(conic_form, ::Nested{T}) where {T} = Vector{T}(undef, length(conic_form.dimension))
_allocate_rows(conic_form, ::Flattened{T}) where {T} = Vector{T}(undef, length(conic_form.b))

"""
    map_rows(f::Function, model, conic_form::MatOI.GeometricConicForm, index_map::MOIU.IndexMap, map_mode::Union{Nested{T}, Flattened{T}})

Given a `model`, its `conic_form`, the `index_map` from the indices of `model`
to the indices of `conic_form` and `map_mode` of type `Nested` (resp.
`Flattened`), return a `Vector{T}` of length equal to the number of cones (resp.
rows) in the conic form where the value for the index (resp. rows) corresponding
to each cone is equal to `f(ci, r)` where `ci` is the corresponding constraint
index in `model` and `r` is a `UnitRange` of the corresponding rows in the conic
form.
"""
function map_rows(f::Function, model, conic_form::MatOI.GeometricConicForm, index_map::MOIU.IndexMap, map_mode::Union{Nested, Flattened})
    x = _allocate_rows(conic_form, map_mode)
    k = 0
    for (F, S) in MOI.get(model, MOI.ListOfConstraints())
        # Function barrier for type unstability of `F` and `S`
        # `conmap` is a `MOIU.DoubleDicts.MainIndexDoubleDict`, we index it at `F, S`
        # which returns a `MOIU.DoubleDicts.IndexWithType{F, S}` which is type stable.
        # If we have a small number of different constraint types and many
        # constraint of each type, this mostly removes type unstabilities
        # as most the time is in `_map_rows!` which is type stable.
        k = _map_rows!(f, x, model, conic_form, index_map.conmap[F, S], map_mode, k)
    end
    return x
end

function _check_termination_status(model::Optimizer)
    if !in(
        MOI.get(model, MOI.TerminationStatus()), (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
        )
        error("problem status: ", MOI.get(model.optimizer, MOI.TerminationStatus()))
    end
end

"""
    _forward_conic(model::Optimizer)

Method to compute the product of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c`  to the perturbations `dA`, `db`, `dc`.
This is similar to [`forward`](@ref).

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
function _forward_conic(model::Optimizer)
    _check_termination_status(model)

    if model.gradient_cache === nothing
        build_conic_diff_cache!(model)
    end

    M = model.gradient_cache.M
    vp = model.gradient_cache.vp
    Dπv = model.gradient_cache.Dπv
    (x, y, s) = model.gradient_cache.xys
    A = model.gradient_cache.A
    b = model.gradient_cache.b
    c = model.gradient_cache.c
    index_map = model.gradient_cache.index_map

    objective_function = _convert(MOI.ScalarAffineFunction{Float64}, model.input_cache.objective)
    sparse_array_obj = sparse_array_representation(objective_function, length(c), index_map)
    dc = sparse_array_obj.terms

    db = zeros(length(b))
    _fill(model, model.gradient_cache.conic_form, db)
    (lines, cols) = size(A)
    nz = nnz(A)
    dAi = zeros(Int, 0)
    dAj = zeros(Int, 0)
    dAv = zeros(Float64, 0)
    sizehint!(dAi, nz)
    sizehint!(dAj, nz)
    sizehint!(dAv, nz)
    _fill(model, model.gradient_cache.conic_form, dAi, dAj, dAv)
    dA = sparse(dAi, dAj, dAv, lines, cols)

    m = size(A, 1)
    n = size(A, 2)
    N = m + n + 1
    # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
    (u, v, w) = (x, y - s, 1.0)

    # g = dQ * Π(z/|w|) = dQ * [u, vp, 1.0]
    RHS = [dA' * vp + dc; -dA * u + db; -dc ⋅ u - db ⋅ vp]

    dz = if norm(RHS) <= 1e-400 # TODO: parametrize or remove
        RHS .= 0 # because M is square
    else
        lsqr(M, RHS)
    end

    du, dv, dw = dz[1:n], dz[n+1:n+m], dz[n+m+1]
    model.forw_grad_cache = ConicForwCache(du, dv, [dw])
    return nothing
    # dx = du - x * dw
    # dy = Dπv * dv - y * dw
    # ds = Dπv * dv - dv - s * dw
    # return -dx, -dy, -ds
end

# Just a hack that will be removed once we use `MOIU.MatrixOfConstraints`
struct _QPForm end
MatOI.rows(::_QPForm, ci::MOI.ConstraintIndex) = ci.value

function _fill(model::Optimizer, conic_form, args...)
    _fill(S -> true, model, conic_form, args...)
end
function _fill(filter::Function, model::Optimizer, conic_form, args...)
    conmap = model.gradient_cache.index_map.conmap
    varmap = model.gradient_cache.index_map.varmap
    for (F, S) in MOI.get(model, MOI.ListOfConstraints())
        filter(S) || continue
        if F == MOI.ScalarAffineFunction{Float64}
            _fill(args..., conic_form, conmap[F,S], varmap, model.input_cache.scalar_constraints[F,S])
        elseif F == MOI.VectorAffineFunction{Float64}
            _fill(args..., conic_form, conmap[F,S], varmap, model.input_cache.vector_constraints[F,S])
        end
    end
    return
end

function _fill(vector::Vector, conic_form, constraint_map, variable_map, dict)
    for (ci, func) in dict
        r = MatOI.rows(conic_form, constraint_map[ci])
        vector[r] = MOI.constant(func)
    end
end
function _fill(I::Vector, J::Vector, V::Vector, conic_form, constraint_map, variable_map, dict)
    for (ci, func) in dict
        r = MatOI.rows(conic_form, constraint_map[ci])
        for term in func.terms
            _push_term(I, J, V, r, term, variable_map)
        end
    end
end
function _push_term(I::Vector, J::Vector, V::Vector, r::Integer, term::MOI.ScalarAffineTerm, variable_map)
    push!(I, r)
    push!(J, variable_map[term.variable_index].value)
    push!(V, term.coefficient)
end
function _push_term(I::Vector, J::Vector, V::Vector, r::UnitRange, term::MOI.VectorAffineTerm, variable_map)
    _push_term(I, J, V, r[term.output_index], term.scalar_term, variable_map)
end

"""
    _backward_conic(model::Optimizer, dx::Vector{Float64}, dy::Vector{Float64}, ds::Vector{Float64})

Method to compute the product of the transpose of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c`  to the perturbations `dx`, `dy`, `ds`.
This is similar to [`backward`](@ref).

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
function _backward_conic(model::Optimizer)
    _check_termination_status(model)

    if model.gradient_cache === nothing
        build_conic_diff_cache!(model)
    end

    M = model.gradient_cache.M
    vp = model.gradient_cache.vp
    Dπv = model.gradient_cache.Dπv
    (x, y, s) = model.gradient_cache.xys
    A = model.gradient_cache.A
    b = model.gradient_cache.b
    c = model.gradient_cache.c

    index_map = model.gradient_cache.index_map
    dx = zeros(length(c))
    for (vi, val) in model.input_cache.dx
        inner_index = index_map[vi].value
        dx[inner_index] = val
    end
    dy = zeros(length(b))
    ds = zeros(length(b))

    m = size(A, 1)
    n = size(A, 2)
    N = m + n + 1
    # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
    (u, v, w) = (x, y - s, 1.0)

    # dz = D \phi (z)^T (dx,dy,dz)
    dz = [
        dx
        Dπv' * (dy + ds) - ds
        - x' * dx - y' * dy - s' * ds
    ]

    g = if norm(dz) <= 1e-4 # TODO: parametrize or remove
        dz .= 0 # because M is square
    else
        lsqr(M, dz)
    end

    πz = [
        u
        vp
        1.0
    ]

    # TODO: very important
    # contrast with:
    # http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-109.pdf
    # pg 97, cap 7.4.2

    model.back_grad_cache = ConicBackCache(g, πz)
    return nothing
    # dQ = - g * πz'
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    # dc = - dQ[1:n, end] + dQ[end, 1:n]'
    # return dA, db, dc
end
