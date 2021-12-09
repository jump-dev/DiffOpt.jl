"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note
Currently supports differentiating linear and quadratic programs only.
"""

Base.@kwdef struct QPCache
    problem_data::Tuple{
        SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # Q, q
        SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # G, h
        SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # A, b
        Int, Vector{VI}, # nz, var_list
        Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}, # nineq_le, le_con_idx
        Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}, # nineq_ge, ge_con_idx
        Int, Vector{MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}}, # nineq_sv_le, le_con_sv_idx
        Int, Vector{MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}}, # nineq_sv_ge, ge_con_sv_idx
        Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}, # neq, eq_con_idx
        Int, Vector{MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}}, # neq_sv, eq_con_sv_idx
    }
    inequality_duals::Vector{Float64}
    equality_duals::Vector{Float64}
    var_primals::Vector{Float64}
    lhs::SparseMatrixCSC{Float64, Int}
end

Base.@kwdef struct ConicCache
    M::SparseMatrixCSC{Float64, Int}
    vp::Vector
    Dπv::BlockDiagonals.BlockDiagonal{Float64, Matrix{Float64}}
    xys::NTuple{3, Vector{Float64}}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    c::Vector{Float64}
    cones::ProductOfSets{Float64}
end

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

function Base.empty!(cache::DiffInputCache)
    empty!(cache.dx)
    empty!(cache.scalar_constraints)
    empty!(cache.vector_constraints)
    cache.objective = nothing
    return
end

"""
    ForwardInObjective <: MOI.AbstractModelAttribute

A `MOI.AbstractModelAttribute` to set input data to forward differentiation, that
is, problem input data.
The possible values are any `MOI.AbstractScalarFunction`.
A `MOI.ScalarQuadraticFunction` can only be used in linearly constrained
quadratic models.

For instance, if the objective contains `θ * (x + 2y)`, for the purpose of
computing the derivative with respect to `θ`, the following should be set:
```julia
MOI.set(model, DiffOpt.ForwardInObjective(), 1.0 * x + 2.0 * y)
```
where `x` and `y` are the relevant `MOI.VariableIndex`.
"""
struct ForwardInObjective <: MOI.AbstractModelAttribute end

"""
    ForwardInConstraint <: MOI.AbstractConstraintAttribute

A `MOI.AbstractConstraintAttribute` to set input data to forward differentiation, that
is, problem input data.

For instance, if the scalar constraint of index `ci` contains `θ * (x + 2y) <= 5θ`,
for the purpose of computing the derivative with respect to `θ`, the following
should be set:
```julia
MOI.set(model, DiffOpt.ForwardInConstraint(), ci, 1.0 * x + 2.0 * y - 5.0)
```
Note that we use `-5` as the `ForwardInConstraint` sets the tangent of the
ConstraintFunction so we consider the expression `θ * (x + 2y - 5)`.
"""
struct ForwardInConstraint <: MOI.AbstractConstraintAttribute end


"""
    ForwardOutVariablePrimal <: MOI.AbstractVariableAttribute

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
    BackwardInVariablePrimal <: MOI.AbstractVariableAttribute

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
    BackwardOutObjective <: MOI.AbstractModelAttribute

A `MOI.AbstractModelAttribute` to get output data to backward differentiation,
that is, problem input data.

For instance, to get the tangent of the objective function corresponding to
the tangent given to `BackwardInVariablePrimal`, do the
following:
```julia
func = MOI.get(model, DiffOpt.BackwardOutObjective)
```
Then, to get the sensitivity of the linear term with variable `x`, do
```julia
JuMP.coefficient(func, x)
```
To get the sensitivity with respect to the quadratic term with variables `x`
and `y`, do either
```julia
JuMP.coefficient(func, x, y)
```
or
```julia
DiffOpt.quad_sym_half(func, x, y)
```

!!! warning
    These two lines are **not** equivalent in case `x == y`, see
    [`quad_sym_half`](@ref) for the details on the difference between these two
    functions.
"""
struct BackwardOutObjective <: MOI.AbstractModelAttribute end
MOI.is_set_by_optimize(::BackwardOutObjective) = true

"""
    BackwardOutConstraint

An `MOI.AbstractConstraintAttribute` to get output data to backward differentiation, that
is, problem input data.

For instance, if the following returns `x + 2y + 5`, it means that the tangent
has coordinate `1` for the coefficient of `x`, coordinate `2` for the
coefficient of `y` and `5` for the function constant.
If the constraint is of the form `func == constant` or `func <= constant`,
the tangent for the constant on the right-hand side is `-5`.
```julia
MOI.get(model, DiffOpt.BackwardOutConstraint(), ci)
```
"""
struct BackwardOutConstraint <: MOI.AbstractConstraintAttribute end
MOI.is_set_by_optimize(::BackwardOutConstraint) = true

"""
    @enum ProgramClassCode QUADRATIC CONIC AUTOMATIC

Program class used by DiffOpt. DiffOpt implements differentiation of two
different program class:
1) Quadratic Program (QP): quadratic objective and linear constraints and
2) Conic Program (CP): linear objective and conic constraints.

`AUTOMATIC` which means that the class will be automatically selected given the
problem data: if any constraint is conic, CP is used and QP is used otherwise.
See [`ProgramClass`](@ref).
"""
@enum ProgramClassCode QUADRATIC CONIC AUTOMATIC

abstract type DiffModel <: MOI.ModelLike end

mutable struct QPDiff <: DiffModel
    # storage for problem data in matrix form
    # includes maps from matrix indices to problem data held in `optimizer`
    # also includes KKT matrices
    # also includes the solution
    gradient_cache::Union{Nothing,QPCache}

    # caches for sensitivity output
    # result from solving KKT/residualmap linear systems
    # this allows keeping the same `gradient_cache`
    # if only sensitivy input changes
    forw_grad_cache::Union{Nothing,QPForwBackCache}
    back_grad_cache::Union{Nothing,QPForwBackCache}

    # sensitivity input cache using MOI like sparse format
    input_cache::DiffInputCache
end
function QPDiff()
    return QPDiff(nothing, nothing, nothing, DiffInputCache())
end

mutable struct ConicDiff <: DiffModel
    # storage for problem data in matrix form
    # includes maps from matrix indices to problem data held in `optimizer`
    # also includes KKT matrices
    # also includes the solution
    gradient_cache::Union{Nothing,ConicCache}

    # caches for sensitivity output
    # result from solving KKT/residualmap linear systems
    # this allows keeping the same `gradient_cache`
    # if only sensitivy input changes
    forw_grad_cache::Union{Nothing,ConicForwCache}
    back_grad_cache::Union{Nothing,ConicBackCache}

    # sensitivity input cache using MOI like sparse format
    input_cache::DiffInputCache
end
function ConicDiff()
    return ConicDiff(nothing, nothing, nothing, DiffInputCache())
end

function MOI.set(model::DiffModel, ::ForwardInObjective, objective)
    model.input_cache.objective = objective
    return
end
function MOI.set(model::DiffModel, ::BackwardInVariablePrimal, vi::VI, val)
    model.input_cache.dx[vi] = val
    return
end
function MOI.set(model::DiffModel,
    ::ForwardInConstraint,
    ci::CI{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    model.input_cache.scalar_constraints[ci] = func
    return
end
function MOI.set(model::DiffModel,
    ::ForwardInConstraint,
    ci::CI{MOI.VectorAffineFunction{T},S},
    func::MOI.VectorAffineFunction{T},
) where {T,S}
    model.input_cache.vector_constraints[ci] = func
    return
end

function MOI.get(model::DiffModel, ::ForwardOutVariablePrimal, vi::VI)
    return _get_dx(model, vi)
end
_get_dx(model::DiffModel, vi) = _get_dx(model.forw_grad_cache, model.gradient_cache, vi)
function _get_dx(cache::QPForwBackCache, g_cache::QPCache, vi)
    return cache.dz[vi.value]
end
function _get_dx(f_cache::ConicForwCache, g_cache::ConicCache, vi)
    i = vi.value
    du = f_cache.du
    dw = f_cache.dw
    x = g_cache.xys[1]
    return - (du[i] - x[i] * dw[])
end

function lazy_combination(op::F, α, a, β, b) where {F<:Function}
    return LazyArrays.ApplyArray(
        op,
        LazyArrays.@~(α .* a),
        LazyArrays.@~(β .* b),
    )
end
# Workaround for Julia v1.0
@static if VERSION < v"1.6"
    _view(x, I) = x[I]
else
    _view(x, I) = view(x, I)
end
function lazy_combination(op::F, α, a, β, b, I::UnitRange) where {F<:Function}
    return lazy_combination(op, α, _view(a, I), β, _view(b, I))
end
function lazy_combination(op::F, a, b, i::Integer, args::Vararg{Any,N}) where {F<:Function,N}
    return lazy_combination(op, a[i], b, b[i], a, args...)
end
function lazy_combination(op::F, a, b, i::UnitRange, I::UnitRange) where {F<:Function}
    return lazy_combination(op, _view(a, i), b, _view(b, i), a, I)
end

function MOI.get(model::DiffModel, ::BackwardOutObjective)
    return _back_obj(model.back_grad_cache, model.gradient_cache)
end
function _back_obj(b_cache::ConicBackCache, g_cache::ConicCache)
    g = b_cache.g
    πz = b_cache.πz
    dc = lazy_combination(-, πz, g, length(g))
    return VectorScalarAffineFunction(dc, 0.0)
end
function _back_obj(b_cache::QPForwBackCache, g_cache::QPCache)
    ∇z = b_cache.dz
    z = g_cache.var_primals
    # `∇z * z' + z * ∇z'` doesn't work, see
    # https://github.com/JuliaArrays/LazyArrays.jl/issues/178
    dQ = LazyArrays.@~ (∇z .* z' + z .* ∇z') / 2.0
    return MatrixScalarQuadraticFunction(
        VectorScalarAffineFunction(b_cache.dz, 0.0),
        dQ,
    )
end

_lazy_affine(vector, constant::Number) = VectorScalarAffineFunction(vector, constant)
_lazy_affine(matrix, vector) = MatrixVectorAffineFunction(matrix, vector)
function MOI.get(model::DiffModel, ::BackwardOutConstraint, ci::CI)
    return _lazy_affine(_get_dA(model, ci), _get_db(model, ci))
end
_get_db(model::DiffModel, ci) = _get_db(model.back_grad_cache, model.gradient_cache, ci)
function _get_db(b_cache::ConicBackCache, g_cache::ConicCache, ci::CI{F,S}
) where {F<:MOI.AbstractVectorFunction,S}
    cf = g_cache.cones
    i = MOI.Utilities.rows(cf, ci) # vector
    # i = ci.value
    (x, _, _) = g_cache.xys
    n = length(x) # columns in A
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    g = b_cache.g
    πz = b_cache.πz
    return lazy_combination(-, πz, g, length(g), n .+ i)
end
function _get_db(b_cache::ConicBackCache, g_cache::ConicCache, ci::CI{F,S}
) where {F<:MOI.AbstractScalarFunction,S}
    i = ci.value
    (x, _, _) = g_cache.xys
    n = length(x) # columns in A
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    g = b_cache.g
    πz = b_cache.πz
    dQ_ni_end = - g[n+i] * πz[end]
    dQ_end_ni = - g[end] * πz[n+i]
    return - dQ_ni_end + dQ_end_ni
end
_neg_if_gt(x, ::Type{<:MOI.LessThan}) = x
_neg_if_gt(x, ::Type{<:MOI.GreaterThan}) = -x
function _get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S}
    i = ci.value
    # dh = -Diagonal(λ) * dλ
    dλ = b_cache.dλ
    λ = g_cache.inequality_duals
    return _neg_if_gt(λ[i] * dλ[i], S)
end
function _get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S<:MOI.EqualTo}
    i = ci.value
    dν = b_cache.dν
    return dν[i]
end

_get_dA(model::DiffModel, ci) = _get_dA(model.back_grad_cache, model.gradient_cache, ci)
function _get_dA(b_cache::ConicBackCache, g_cache::ConicCache, ci::CI{F,S}
) where {F<:MOI.AbstractScalarFunction,S}
    j = vi.value
    i = ci.value
    (x, y, _) = g_cache.xys
    n = length(x) # columns in A
    m = length(y) # lines in A
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    g = b_cache.g
    πz = b_cache.πz
    return lazy_combination(-, g, πz, i, n .+ (1:n))
end
function _get_dA(b_cache::ConicBackCache, g_cache::ConicCache, ci::CI{F,S}
) where {F<:MOI.AbstractVectorFunction,S}
    cf = g_cache.cones
    i = MOI.Utilities.rows(cf, ci) # vector
    # i = ci.value
    (x, y, _) = g_cache.xys
    n = length(x) # columns in A
    m = length(y) # lines in A
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    g = b_cache.g
    πz = b_cache.πz
    return lazy_combination(-, g, πz, i, n .+ (1:n))
end
# quadratic matrix indexes are split by type either == or (<=/>=)
function _get_dA(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F, S<:MOI.EqualTo}
    i = ci.value
    z = g_cache.var_primals
    dz = b_cache.dz
    ν = g_cache.equality_duals
    dν = b_cache.dν
    return lazy_combination(+, dν[i], z, ν[i], dz)
end
function _get_dA(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S}
    i = ci.value
    z = g_cache.var_primals
    dz = b_cache.dz
    λ = g_cache.inequality_duals
    dλ = b_cache.dλ
    l = _neg_if_gt(λ[i], S)
    return lazy_combination(+, l * dλ[i], z, l * λ[i], dz)
end

"""
    backward(model::QPDiff)

Method to differentiate optimal solution `z` and return
product of jacobian matrices (`dz / dQ`, `dz / dq`, etc) with
the backward pass vector `dl / dz`

The method computes the product of
1. jacobian of problem solution `z*` with respect to
    problem parameters set with the [`BackwardInVariablePrimal`](@ref)
2. a backward pass vector `dl / dz`, where `l` can be a loss function

Note that this method *does not returns* the actual jacobians.

For more info refer eqn(7) and eqn(8) of https://arxiv.org/pdf/1703.00443.pdf
"""
function backward(model::QPDiff)
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

    dl_dz = zeros(length(z))
    for (vi, value) in model.input_cache.dx
        dl_dz[vi.value] = value
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
    forward(model::QPDiff)
"""
function forward(model::QPDiff)
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

    objective_function = _convert(MOI.ScalarQuadraticFunction{Float64}, model.input_cache.objective)
    sparse_array_obj = sparse_array_representation(objective_function, LinearAlgebra.checksquare(Q))
    dQ = sparse_array_obj.quadratic_terms
    dq = sparse_array_obj.affine_terms

    # The user sets the constraint function in the sense `func`-in-`set` while
    # `db` and `dh` corresponds to the tangents of the set constants. Therefore,
    # we should multiply the constant by `-1`. For `GreaterThan`, we needed to
    # multiply by `-1` to transform it to `LessThan` so it cancels out.
    db = zeros(length(b))
    _fill(isequal(MOI.EqualTo{Float64}), (::Type{MOI.EqualTo{Float64}}) -> true, model.gradient_cache, model.input_cache, _QPSets(), db)
    dh = zeros(length(h))
    _fill(!isequal(MOI.EqualTo{Float64}), !isequal(MOI.GreaterThan{Float64}), model.gradient_cache, model.input_cache, _QPSets(), dh)

    nz = nnz(A)
    (lines, cols) = size(A)
    dAi = zeros(Int, 0)
    dAj = zeros(Int, 0)
    dAv = zeros(Float64, 0)
    sizehint!(dAi, nz)
    sizehint!(dAj, nz)
    sizehint!(dAv, nz)
    _fill(isequal(MOI.EqualTo{Float64}), isequal(MOI.GreaterThan{Float64}), model.gradient_cache, model.input_cache, _QPSets(), dAi, dAj, dAv)
    dA = sparse(dAi, dAj, dAv, lines, cols)

    nz = nnz(G)
    (lines, cols) = size(G)
    dGi = zeros(Int, 0)
    dGj = zeros(Int, 0)
    dGv = zeros(Float64, 0)
    sizehint!(dGi, nz)
    sizehint!(dGj, nz)
    sizehint!(dGv, nz)
    _fill(!isequal(MOI.EqualTo{Float64}), isequal(MOI.GreaterThan{Float64}), model.gradient_cache, model.input_cache, _QPSets(), dGi, dGj, dGv)
    dG = sparse(dGi, dGj, dGv, lines, cols)


    RHS = [
        dQ * z + dq + dG' * λ + dA' * ν
        λ .* (dG * z) - λ .* dh
        dA * z - db
    ]

    partial_grads = if norm(Q) ≈ 0
        -lsqr(LHS', RHS)
    else
        -_linsolve(LHS', RHS)
    end


    nv = length(z)
    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    dz = partial_grads[1:nv]
    dλ = partial_grads[nv+1:nv+nineq_total]
    dν = partial_grads[nv+nineq_total+1:nv+nineq_total+neq+neq_sv]

    model.forw_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
end

_linsolve(A, b) = A \ b
# See https://github.com/JuliaLang/julia/issues/32668
_linsolve(A, b::SparseVector) = A \ Vector(b)


"""
    π(v::Vector{Float64}, model::MOI.ModelLike, cones::ProductOfSets)

Given a `model`, its `cones` and the `index_map` from the indices of
`model` to the indices of `cones`, find the projection of the vectors `v`
of length equal to the number of rows in the conic form onto the cartesian
product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function π(v::Vector{T}, model::MOI.ModelLike, cones::ProductOfSets, index_map::MOIU.IndexMap) where T
    return map_rows(model, cones, index_map, Flattened{T}()) do ci, r
        MOSD.projection_on_set(
            MOSD.DefaultDistance(),
            v[r],
            MOI.dual_set(MOI.get(model, MOI.ConstraintSet(), ci))
        )
    end
end


"""
    Dπ(v::Vector{Float64}, model, cones::ProductOfSets, index_map::MOIU.IndexMap)

Given a `model`, its `cones` and the `index_map` from the indices of
`model` to the indices of `cones`, find the gradient of the projection of
the vectors `v` of length equal to the number of rows in the conic form onto the
cartesian product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function Dπ(v::Vector{T}, model::MOI.ModelLike, cones::ProductOfSets, index_map::MOIU.IndexMap) where T
    return BlockDiagonals.BlockDiagonal(
        map_rows(model, cones, index_map, Nested{Matrix{T}}()) do ci, r
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
function _map_rows!(f::Function, x::Vector, model, cones::ProductOfSets, index_map::MOIU.DoubleDicts.IndexDoubleDictInner{F, S}, map_mode, k) where {F, S}
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
        r = MOI.Utilities.rows(cones, index_map[ci])
        k += 1
        _assign_mapped!(x, f(ci, r), r, k, map_mode)
    end
    return k
end

# Allocate a vector for storing the output of `map_rows`.
_allocate_rows(cones, ::Nested{T}) where {T} = Vector{T}(undef, length(cones.dimension))
_allocate_rows(cones, ::Flattened{T}) where {T} = Vector{T}(undef, MOI.dimension(cones))

"""
    map_rows(f::Function, model, cones::ProductOfSets, index_map::MOIU.IndexMap, map_mode::Union{Nested{T}, Flattened{T}})

Given a `model`, its `cones`, the `index_map` from the indices of `model`
to the indices of `cones` and `map_mode` of type `Nested` (resp.
`Flattened`), return a `Vector{T}` of length equal to the number of cones (resp.
rows) in the conic form where the value for the index (resp. rows) corresponding
to each cone is equal to `f(ci, r)` where `ci` is the corresponding constraint
index in `model` and `r` is a `UnitRange` of the corresponding rows in the conic
form.
"""
function map_rows(f::Function, model, cones::ProductOfSets, index_map::MOIU.IndexMap, map_mode::Union{Nested, Flattened})
    x = _allocate_rows(cones, map_mode)
    k = 0
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        # Function barrier for type unstability of `F` and `S`
        # `con_map` is a `MOIU.DoubleDicts.MainIndexDoubleDict`, we index it at `F, S`
        # which returns a `MOIU.DoubleDicts.IndexWithType{F, S}` which is type stable.
        # If we have a small number of different constraint types and many
        # constraint of each type, this mostly removes type unstabilities
        # as most the time is in `_map_rows!` which is type stable.
        k = _map_rows!(f, x, model, cones, index_map.con_map[F, S], map_mode, k)
    end
    return x
end

"""
    forward(model::ConicDiff)

Method to compute the product of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c`  to the perturbations `dA`, `db`, `dc`.
This is similar to [`forward`](@ref).

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
function forward(model::ConicDiff)
    M = model.gradient_cache.M
    vp = model.gradient_cache.vp
    Dπv = model.gradient_cache.Dπv
    (x, y, s) = model.gradient_cache.xys
    A = model.gradient_cache.A
    b = model.gradient_cache.b
    c = model.gradient_cache.c

    objective_function = _convert(MOI.ScalarAffineFunction{Float64}, model.input_cache.objective)
    sparse_array_obj = sparse_array_representation(objective_function, length(c))
    dc = sparse_array_obj.terms

    db = zeros(length(b))
    _fill(S -> false, model.gradient_cache, model.input_cache, model.gradient_cache.cones, db)
    (lines, cols) = size(A)
    nz = nnz(A)
    dAi = zeros(Int, 0)
    dAj = zeros(Int, 0)
    dAv = zeros(Float64, 0)
    sizehint!(dAi, nz)
    sizehint!(dAj, nz)
    sizehint!(dAv, nz)
    _fill(S -> false, model.gradient_cache, model.input_cache, model.gradient_cache.cones, dAi, dAj, dAv)
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
struct _QPSets end
MOI.Utilities.rows(::_QPSets, ci::MOI.ConstraintIndex) = ci.value

function _fill(neg::Function, gradient_cache, input_cache, cones, args...)
    _fill(S -> true, neg, gradient_cache, input_cache, cones, args...)
end
function _fill(filter::Function, neg::Function, gradient_cache, input_cache, cones, args...)
    for (F, S) in keys(input_cache.scalar_constraints.dict)
        filter(S) || continue
        _fill(args..., neg(S), cones, input_cache.scalar_constraints[F,S])
    end
    for (F, S) in keys(input_cache.vector_constraints.dict)
        _fill(args..., neg(S), cones, input_cache.vector_constraints[F,S])
    end
    return
end

function _fill(vector::Vector, neg::Bool, cones, dict)
    for (ci, func) in dict
        r = MOI.Utilities.rows(cones, ci)
        vector[r] = neg ? -MOI.constant(func) : MOI.constant(func)
    end
end
function _fill(I::Vector, J::Vector, V::Vector, neg::Bool, cones, dict)
    for (ci, func) in dict
        r = MOI.Utilities.rows(cones, ci)
        for term in func.terms
            _push_term(I, J, V, neg, r, term)
        end
    end
end
function _push_term(I::Vector, J::Vector, V::Vector, neg::Bool, r::Integer, term::MOI.ScalarAffineTerm)
    push!(I, r)
    push!(J, term.variable.value)
    push!(V, neg ? -term.coefficient : term.coefficient)
end
function _push_term(I::Vector, J::Vector, V::Vector, neg::Bool, r::UnitRange, term::MOI.VectorAffineTerm)
    _push_term(I, J, V, neg, r[term.output_index], term.scalar_term)
end

"""
    backward(model::ConicDiff)

Method to compute the product of the transpose of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c`  to the perturbations `dx`, `dy`, `ds`.
This is similar to [`backward`](@ref).

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
function backward(model::ConicDiff)
    M = model.gradient_cache.M
    vp = model.gradient_cache.vp
    Dπv = model.gradient_cache.Dπv
    (x, y, s) = model.gradient_cache.xys
    A = model.gradient_cache.A
    b = model.gradient_cache.b
    c = model.gradient_cache.c

    dx = zeros(length(c))
    for (vi, value) in model.input_cache.dx
        dx[vi.value] = value
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
