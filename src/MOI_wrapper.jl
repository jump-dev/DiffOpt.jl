"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note
Currently supports differentiating linear and quadratic programs only.
"""

const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex

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
    MOI.SecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,
}

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

julia> backward_quad(model)  # for convex quadratic models

julia> backward_quad(model)  # for convex conic models
```
"""
function diff_optimizer(optimizer_constructor)::Optimizer
    return Optimizer(MOI.instantiate(optimizer_constructor, with_bridge_type=Float64))
end

# TODO: arragen this in fields
Base.@kwdef struct QPCache
    problem_data::Tuple{
        Matrix{Float64}, Vector{Float64}, # Q, q
        Matrix{Float64}, Vector{Float64}, # G, h
        Matrix{Float64}, Vector{Float64}, # A, b
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
    lhs::SparseMatrixCSC{Float64, Int}
    index_map::MOIU.IndexMap
end
Base.@kwdef struct ConicCache
    M::SparseMatrixCSC{Float64, Int}
    vp::Vector
    Dπv::BlockDiagonals.BlockDiagonal{Float64, Matrix{Float64}}
    xys::NTuple{3, Vector{Float64}}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    c::Vector{Float64}
    c::Vector{Float64}
    index_map::MOIU.IndexMap
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
# TODO bechmark access to these, ^^, type unstable type
# possibly add fields for all of them

const MOIDD = MOI.Utilities.DoubleDicts

# TODO: possibly use sparse vector/MatrixCSC format for better performance
# will need to keep indices correct
Base.@kwdef struct DiffInputCache
    dx::Dict{VI, Float64}
    dA::MOIDD.DoubleDict{Dict{VI,Float64}} # also includes G for QPs
    db::MOIDD.DoubleDict{Float64} # also includes h for QPs
    dc::Dict{VI, Float64}
    dQ::Dict{Tuple{VI, VI}, Float64}
end

struct ForwardDiffIn{T} <: MOI.AbstractModelAttribute end
struct ForwardDiffOut{T} <: MOI.AbstractModelAttribute end
struct BackwardDiffIn{T} <: MOI.AbstractModelAttribute end
struct BackwardDiffOut{T} <: MOI.AbstractModelAttribute end

struct LinearObjective <: MOI.AbstractModelAttribute end #(var)
struct QuadraticObjective <: MOI.AbstractModelAttribute end #(var, var)
struct ConstraintConstant <: MOI.AbstractModelAttribute end #(con)
struct ConstraintCoefficient <: MOI.AbstractModelAttribute end #(var, con)

get_dx(model::Optimizer, vi) = get_dx(model.forw_grad_cache, model.gradient_cache, vi)
function get_dx(cache::ConicForwCache, g_cache::ConicCache, vi)
    i = g_cache.index_map[vi].value
    return cache.dz[i]
end
function get_dx(f_cache::QPForwBackCache, g_cache::QPCache, vi)
    i = g_cache.index_map[vi].value
    du = f_cache.du
    dw = f_cache.dw
    x = g_cache.xys[1]
    return du[i] - x[i] * dw
end

get_dc(model::Optimizer, vi) = get_dc(model.forw_grad_cache, model.gradient_cache, vi)
function get_dc(b_cache::ConicBackCache, g_cache::ConicCache, vi)
    i = g_cache.index_map[vi].value
    g = b_cache.g
    πz = b_cache.πz
    dQ_i_end = - g[i] * πz[end]
    dQ_end_i = - g[end] * πz[i]
    return - dQ_i_end + dQ_i_end
end
function get_dc(b_cache::QPForwBackCache, g_cache::QPCache, i)
    i = g_cache.index_map[vi].value
    dz = b_cache.dz
    return dz[i]
end

get_dQ(model::Optimizer, vi1, vi2) = get_dQ(model.forw_grad_cache, model.gradient_cache, vi1, vi2)
function get_dQ(b_cache::ConicBackCache, g_cache::ConicCache, vi1, vi2)
    error("Quadratic function not availablein conic model differentiation")
end
function get_dQ(b_cache::QPForwBackCache, g_cache::QPCache, vi1, vi2)
    i = g_cache.index_map[vi1].value
    j = g_cache.index_map[vi2].value
    # TODO move z to cache
    z = model.primal_optimal
    dz = b_cache.dz
    return 0.5 * (dz[i] * z[j] + z[i] * dz[j])
end

get_dA(model::Optimizer, vi, ci) = get_dA(model.forw_grad_cache, model.gradient_cache, vi, ci)
function get_dA(b_cache::ConicBackCache, g_cache::ConicCache, vi, ci)
    j = g_cache.index_map[vi].value
    i = g_cache.index_map[ci].value
    (x, y, _) = g_cache.xys
    n = length(x) # columns in A
    m = length(y) # lines in A
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    dQ_i_nj =  - g[i] * πz[n+j]
    dQ_nj_i =  - g[n+j] * πz[i]
    return - dQ_i_nj + dQ_nj_i
end
function get_dA(b_cache::QPForwBackCache, g_cache::QPCache, vi, ci::CI{F,S}
) where {F, S<:MOI.EqualTo}
    j = g_cache.index_map[vi].value
    i = g_cache.index_map[ci].value
    # TODO move z to cache
    z = model.primal_optimal
    dz = b_cache.dz
    ν = g_cache.equality_duals
    dν = b_cache.dν
    return dν[i] * z[j] + ν[i] * dz[j]
end
function get_dA(b_cache::QPForwBackCache, g_cache::QPCache, vi, ci::CI{F,S}
) where {F, S}
    j = g_cache.index_map[vi].value
    i = g_cache.index_map[ci].value
    # TODO move z to cache
    z = model.primal_optimal
    dz = b_cache.dz
    λ = g_cache.inequality_duals
    dλ = b_cache.dλ
    return λ[i] * (dλ[i] * z[j] + λ[i] * dz[j])
end

get_db(model::Optimizer, ci) = get_db(model.forw_grad_cache, model.gradient_cache, ci)
function get_db(b_cache::ConicBackCache, g_cache::ConicCache, ci)
    i = g_cache.index_map[ci].value
    (x, _, _) = g_cache.xys
    n = length(x) # columns in A
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    dQ_ni_end = - g[n+i] * πz[end]
    dQ_end_ni = - g[end] * πz[n+i]
    return - dQ_ni_end + dQ_end_ni
end
function get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S<:MOI.EqualTo}
    i = g_cache.index_map[ci].value
    # dh = -Diagonal(λ) * dλ
    dλ = b_cache.dλ
    λ = g_cache.inequality_duals
    return - λ[i] * dλ[i]
end
function get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S}
    i = g_cache.index_map[ci].value
    dν = b_cache.dν
    return - dν[i]
end

# TODO: vector/batched version of these methods for better performance
function MOI.get(model::Optimizer, ::ForwardDiffOut{VariablePrimal}, vi::VI)
    return get_dx(model, vi)
end
function MOI.get(model::Optimizer, ::BackwardDiffIn{VariablePrimal}, vi::VI)
    return get(model.input_cache.dx, vi, 0.0)
end
function MOI.set(model::Optimizer, ::BackwardDiffIn{VariablePrimal}, vi::VI, val)
    model.input_cache.dx[vi] = val
    return
end

function MOI.get(model::Optimizer, ::BackwardDiffOut{LinearObjective}, vi::VI)
    return get_dc(model, vi)
end
function MOI.get(model::Optimizer, ::ForwardDiffIn{LinearObjective}, vi::VI)
    return get(model.input_cache.dc, vi, 0.0)
end
function MOI.set(model::Optimizer, ::ForwardDiffIn{LinearObjective}, vi::VI, val)
    model.input_cache.dc[vi] = val
    return
end

function MOI.get(model::Optimizer,
    ::BackwardDiffOut{QuadraticObjective}, vi1::VI, vi2::VI)
    return get_dQ(model, vi1, vi2)
end
function MOI.get(model::Optimizer,
    ::ForwardDiffIn{QuadraticObjective}, vi1::VI, vi2::VI)
    tuple = ifelse(index(vi1) <= index(vi2), (vi1, vi2), (vi2, vi1))
    return get(model.input_cache.dQ, tuple, 0.0)
end
function MOI.set(model::Optimizer,
    ::ForwardDiffIn{QuadraticObjective}, vi1::VI, vi2::VI, val)
    tuple = ifelse(index(vi1) <= index(vi2), (vi1, vi2), (vi2, vi1))
    model.input_cache.dQ[tuple] = val
    return
end

# TODO: deal with vector constraints
function MOI.get(model::Optimizer,
    ::BackwardDiffOut{ConstraintConstant}, ci::CI{F,S}) where {F,S}
    return get_db(model, ci)
end
function MOI.get(model::Optimizer,
    ::ForwardDiffIn{ConstraintConstant}, ci::CI{F,S}) where {F,S}
    return get(model.input_cache.db, ci, 0.0)
end
function MOI.set(model::Optimizer,
    ::ForwardDiffIn{ConstraintConstant}, ci::CI{F,S}, val) where {F,S}
    model.input_cache.db[ci] = val
    return
end

# TODO: deal with vector constraints
function MOI.get(model::Optimizer,
    ::BackwardDiffOut{ConstraintCoefficient}, vi::VI, ci::CI{F,S}) where {F,S}
    return get_dA(model, ci, vi)
end
function MOI.get(model::Optimizer,
    ::ForwardDiffIn{ConstraintCoefficient}, vi::VI, ci::CI{F,S}) where {F,S}
    dict = get(model.input_cache.dA, ci, nothing)
    if dict === nothing
        return 0.0
    else
        return get(dict, vi, 0.0)
    end
end
function MOI.set(model::Optimizer,
    ::ForwardDiffIn{ConstraintCoefficient}, vi::VI, ci::CI{F,S}, val) where {F,S}
    dict = get(model.input_cache.dA, ci, nothing)
    if dict === nothing
        model.input_cache.dA[ci] = Dict(vi => val)
    else
        dict[vi] = val
    end
    return
end

mutable struct Optimizer{OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT
    primal_optimal::Vector{Float64}  # solution
    var_idx::Vector{VI}
    # this one is needed in case we want to send a different perturbation
    gradient_cache::CACHE_TYPE
    # these are needed to query projection results sparsely
    forw_grad_cache::CACHE_FORW_TYPE
    back_grad_cache::CACHE_BACK_TYPE
    input_cache::DiffInputCache

    function Optimizer(optimizer_constructor::OT) where {OT <: MOI.ModelLike}
        new{OT}(
            optimizer_constructor,
            zeros(0),
            Vector{VI}(),
            nothing,
            nothing,
            nothing,
            DiffInputCache(),
        )
    end
end

function MOI.add_variable(model::Optimizer)
    model.gradient_cache = nothing
    vi = MOI.add_variable(model.optimizer)
    push!(model.var_idx, vi)
    return vi
end

function MOI.add_variables(model::Optimizer, N::Int)
    model.gradient_cache = nothing
    return VI[MOI.add_variable(model) for i in 1:N]
end

function MOI.add_constraint(model::Optimizer, f::SUPPORTED_SCALAR_FUNCTIONS, s::SUPPORTED_SCALAR_SETS)
    model.gradient_cache = nothing
    return MOI.add_constraint(model.optimizer, f, s)
end

function MOI.add_constraint(model::Optimizer, vf::SUPPORTED_VECTOR_FUNCTIONS, s::SUPPORTED_VECTOR_SETS)
    model.gradient_cache = nothing
    return MOI.add_constraint(model.optimizer, vf, s)
end

function MOI.add_constraints(model::Optimizer, f::AbstractVector{F}, s::AbstractVector{S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    model.gradient_cache = nothing
    return CI{F, S}[MOI.add_constraint(model, f[i], s[i]) for i in eachindex(f)]
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{<: SUPPORTED_OBJECTIVES}, f::SUPPORTED_OBJECTIVES)
    model.gradient_cache = nothing
    MOI.set(model.optimizer, attr, f)
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.gradient_cache = nothing
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
    model.gradient_cache = nothing
    return MOI.set(model.optimizer, attr, ci, s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    model.gradient_cache = nothing
    return MOI.set(model.optimizer, attr, ci, f)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}, s::S) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    model.gradient_cache = nothing
    return MOI.set(model.optimizer, attr, ci, s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    model.gradient_cache = nothing
    return MOI.set(model.optimizer, attr, ci, f)
end

function MOI.optimize!(model::Optimizer)
    model.gradient_cache = nothing
    MOI.optimize!(model.optimizer)

    # do not fail. interferes with MOI.Tests.linear12test
    if !in(MOI.get(model.optimizer, MOI.TerminationStatus()),  (MOI.LOCALLY_SOLVED, MOI.OPTIMAL))
        @warn "problem status: $(MOI.get(model.optimizer, MOI.TerminationStatus()))"
        return
    end
    # save the solution
    model.primal_optimal = MOI.get(model.optimizer, MOI.VariablePrimal(), model.var_idx)
    return
end


const QP_SET_TYPES = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const QP_FUNCTION_TYPES = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
}

const QP_OBJECTIVE_TYPES = Union{
    MathOptInterface.ScalarAffineFunction{Float64},
    MathOptInterface.ScalarQuadraticFunction{Float64},
}

"""
    backward(model::Optimizer, params...)

Wrapper method for the backward pass. Checks the model/problem type and dispatches a call to
[`backward_quad`](@ref) or [`backward_conic`](@ref) accordingly.
"""
function backward(model::Optimizer, params...)
    # check if the model is a QP
    isQP = qp_supported(model.optimizer)

    # check objective
    if isQP && !(MOI.get(model.optimizer, MOI.ObjectiveFunctionType()) <: QP_OBJECTIVE_TYPES)
        isQP = false
    end

    if isQP
        return backward_quad(model, params...)
    else
        return backward_conic(model, params...)
    end
end

qp_supported(::F, ::S) where {F <: QP_FUNCTION_TYPES, S <: QP_SET_TYPES} = true
qp_supported(::F, ::S) where {F, S} = false
function qp_supported(model)
    con_types = MOI.get(model, MOI.ListOfConstraints())
    for (func, set) in con_types
        if !qp_supported(func, set)
            return false
        end
    end
    return true
end

"""
    backward_quad(model::Optimizer, params::Array{String}, dl_dz::Array{Float64})

Method to differentiate optimal solution `z` and return
product of jacobian matrices (`dz / dQ`, `dz / dq`, etc) with
the backward pass vector `dl / dz`

The method computes the product of
1. jacobian of problem solution `z*` with respect to
    problem parameters `params` recieved as method arguments
2. a backward pass vector `dl / dz`, where `l` can be a loss function

Note that this method *does not returns* the actual jacobians.

For more info refer eqn(7) and eqn(8) of https://arxiv.org/pdf/1703.00443.pdf
"""
function backward_quad(model::Optimizer, dl_dz::Vector{Float64})
    z = model.primal_optimal

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
    λ = model.gradient_cache.inequality_duals
    ν = model.gradient_cache.equality_duals
    LHS = model.gradient_cache.lhs

    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    RHS = [dl_dz; zeros(neq + neq_sv + nineq_total)]
    partial_grads = -LHS \ RHS

    dz = partial_grads[1:nz]
    dλ = partial_grads[nz+1:nz+nineq_total]
    dν = partial_grads[nz+nineq_total+1:nz+nineq_total+neq+neq_sv]

    model.back_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
    # dQ = 0.5 * (dz * z' + z * dz')
    # dq = dz
    # dG = Diagonal(λ) * (dλ * z' + λ * dz') # THIS WAS WRONG
    # dh = -Diagonal(λ) * dλ
    # dA = dν * z'+ ν * dz'
    # db = -dν
end

function forward_quad(model::Optimizer,
    dQ, dq, dG, dh, dA, db
)
    z = model.primal_optimal
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
    λ = model.gradient_cache.inequality_duals
    ν = model.gradient_cache.equality_duals
    LHS = model.gradient_cache.lhs

    RHS = [
        dQ * z + dq + dG' * λ + dA' * ν
        λ .* (dG * z) - λ .* dh
        dA * z - db
    ]

    partial_grads = - LHS \ RHS

    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    dz = partial_grads[1:nz]
    dλ = partial_grads[nz+1:nz+nineq_total]
    dν = partial_grads[nz+nineq_total+1:nz+nineq_total+neq+neq_sv]

    model.forw_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
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
    empty!(model.var_idx)
    model.gradient_cache = nothing
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.optimizer) &&
           isempty(model.primal_optimal) &&
           model.gradient_cache === nothing
           isempty(model.var_idx)
end

# now supports name too
MOIU.supports_default_copy_to(model::Optimizer, copy_names::Bool) = true #!copy_names

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    model.gradient_cache = nothing
    return MOIU.default_copy_to(model.optimizer, src, copy_names)
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    return MOI.get(model.optimizer, MOI.TerminationStatus())
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::VI, value::Float64)
    model.gradient_cache = nothing
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
    model.gradient_cache = nothing
    MOI.delete(model.optimizer, ci)
end

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    model.gradient_cache = nothing
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

MOI.is_valid(model::Optimizer, con::CI) = MOI.is_valid(model.optimizer, con)

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
    model.gradient_cache = nothing
    # delete in inner solver
    MOI.delete(model.optimizer, v)

    # delete from var_idx
    filter!(x -> x ≠ v, model.var_idx)
end

# for array deletion
function MOI.delete(model::Optimizer, indices::Vector{VI})
    model.gradient_cache = nothing
    for i in indices
        MOI.delete(model, i)
    end
end

function MOI.modify(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.AbstractFunctionModification
)
    model.gradient_cache = nothing
    MOI.modify(
        model.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        chg
    )
end

function MOI.modify(model::Optimizer, ci::CI{F, S}, chg::MOI.AbstractFunctionModification) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    model.gradient_cache = nothing
    MOI.modify(model.optimizer, ci, chg)
end

function MOI.modify(model::Optimizer, ci::CI{F, S}, chg::MOI.AbstractFunctionModification) where {F<:MOI.VectorAffineFunction{Float64}, S <: SUPPORTED_VECTOR_SETS}
    model.gradient_cache = nothing
    MOI.modify(model.optimizer, ci, chg)
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
    forward_conic(model::Optimizer, dA::Matrix{Float64}, db::Vector{Float64}, dc::Vector{Float64})

Method to compute the product of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c`  to the perturbations `dA`, `db`, `dc`.
This is similar to [`forward_quad`](@ref).

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
function forward_conic(
    model::Optimizer,
    dA::AbstractMatrix{<:Real}, db::AbstractVector{<:Real}, dc::AbstractVector{<:Real}
)
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

    m = size(A, 1)
    n = size(A, 2)
    N = m + n + 1
    # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
    (u, v, w) = (x, y - s, 1.0)

    # g = dQ * Π(z/|w|) = dQ * [u, vp, 1.0]
    RHS = [dA' * vp + dc; -dA * u + db; -dc ⋅ u - db ⋅ vp]

    dz = if norm(RHS) <= 1e-4 # TODO: parametrize or remove
        RHS .= 0 # because M is square
    else
        lsqr(M, RHS)
    end

    du, dv, dw = dz[1:n], dz[n+1:n+m], dz[n+m+1]
    model.forw_grad_cache = ConicForwCache(du, dv, dw)
    return nothing
    # dx = du - x * dw
    # dy = Dπv * dv - y * dw
    # ds = Dπv * dv - dv - s * dw
    # return -dx, -dy, -ds
end

"""
    backward_conic(model::Optimizer, dx::Vector{Float64}, dy::Vector{Float64}, ds::Vector{Float64})

Method to compute the product of the transpose of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c`  to the perturbations `dx`, `dy`, `ds`.
This is similar to [`backward_quad`](@ref).

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
function backward_conic(
    model::Optimizer,
    dx::AbstractMatrix{<:Real}, dy::AbstractVector{<:Real}, ds::AbstractVector{<:Real}
)
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

    model.back_grad_cache = ConicBackCache(g, πz)
    return nothing
    # dQ = - g * πz'
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    # dc = - dQ[1:n, end] + dQ[end, 1:n]'
    # return dA, db, dc
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
