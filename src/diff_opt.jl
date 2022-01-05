"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note
Currently supports differentiating linear and quadratic programs only.
"""

Base.@kwdef struct QPCache
    inequality_duals::Vector{Float64}
    equality_duals::Vector{Float64}
    var_primals::Vector{Float64}
    lhs::SparseMatrixCSC{Float64, Int}
end

Base.@kwdef struct ConicCache
    M::SparseMatrixCSC{Float64, Int}
    vp::Vector
    Dπv::BlockDiagonals.BlockDiagonal{Float64, Matrix{Float64}}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    c::Vector{Float64}
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

MOI.supports_incremental_interface(::DiffModel) = true

MOI.is_valid(model::DiffModel, idx::MOI.Index) = MOI.is_valid(model.model, idx)

function MOI.add_variables(model::DiffModel, n)
    return MOI.add_variables(model.model, n)
end

function MOI.Utilities.pass_nonvariable_constraints(
    dest::DiffModel,
    src::MOI.ModelLike,
    idxmap::MOIU.IndexMap,
    constraint_types,
)
    MOI.Utilities.pass_nonvariable_constraints(dest.model, src, idxmap, constraint_types)
end

function MOI.Utilities.final_touch(model::DiffModel, index_map)
    MOI.Utilities.final_touch(model.model, index_map)
end

function MOI.add_constraint(model::DiffModel, func::MOI.AbstractFunction, set::MOI.AbstractSet)
    return MOI.add_constraint(model.model, func, set)
end

function _enlarge_set(vec::Vector, idx, value)
    m = last(idx)
    if length(vec) < m
        n = length(vec)
        resize!(vec, m)
        fill!(view(vec, (n+1):m), NaN)
        vec[idx] = value
    end
    return
end

function MOI.set(model::DiffModel, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value)
    MOI.throw_if_not_valid(model, vi)
    _enlarge_set(model.x, vi.value, value)
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

"""
    π(v::Vector{Float64}, model::MOI.ModelLike, cones::ProductOfSets)

Given a `model`, its `cones`, find the projection of the vectors `v`
of length equal to the number of rows in the conic form onto the cartesian
product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function π(v::Vector{T}, model::MOI.ModelLike, cones::ProductOfSets) where T
    return map_rows(model, cones, Flattened{T}()) do ci, r
        MOSD.projection_on_set(
            MOSD.DefaultDistance(),
            v[r],
            MOI.dual_set(MOI.get(model, MOI.ConstraintSet(), ci))
        )
    end
end


"""
    Dπ(v::Vector{Float64}, model, cones::ProductOfSets)

Given a `model`, its `cones`, find the gradient of the projection of
the vectors `v` of length equal to the number of rows in the conic form onto the
cartesian product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function Dπ(v::Vector{T}, model::MOI.ModelLike, cones::ProductOfSets) where T
    return BlockDiagonals.BlockDiagonal(
        map_rows(model, cones, Nested{Matrix{T}}()) do ci, r
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
function _map_rows!(f::Function, x::Vector, model, cones::ProductOfSets, ::Type{F}, ::Type{S}, map_mode, k) where {F, S}
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
        r = MOI.Utilities.rows(cones, ci)
        k += 1
        _assign_mapped!(x, f(ci, r), r, k, map_mode)
    end
    return k
end

# Allocate a vector for storing the output of `map_rows`.
_allocate_rows(cones, ::Nested{T}) where {T} = Vector{T}(undef, length(cones.dimension))
_allocate_rows(cones, ::Flattened{T}) where {T} = Vector{T}(undef, MOI.dimension(cones))

"""
    map_rows(f::Function, model, cones::ProductOfSets, map_mode::Union{Nested{T}, Flattened{T}})

Given a `model`, its `cones` and `map_mode` of type `Nested` (resp.
`Flattened`), return a `Vector{T}` of length equal to the number of cones (resp.
rows) in the conic form where the value for the index (resp. rows) corresponding
to each cone is equal to `f(ci, r)` where `ci` is the corresponding constraint
index in `model` and `r` is a `UnitRange` of the corresponding rows in the conic
form.
"""
function map_rows(f::Function, model, cones::ProductOfSets, map_mode::Union{Nested, Flattened})
    x = _allocate_rows(cones, map_mode)
    k = 0
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        # Function barrier for type unstability of `F` and `S`
        # `con_map` is a `MOIU.DoubleDicts.MainIndexDoubleDict`, we index it at `F, S`
        # which returns a `MOIU.DoubleDicts.IndexWithType{F, S}` which is type stable.
        # If we have a small number of different constraint types and many
        # constraint of each type, this mostly removes type unstabilities
        # as most the time is in `_map_rows!` which is type stable.
        k = _map_rows!(f, x, model, cones, F, S, map_mode, k)
    end
    return x
end

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


function MOI.supports(model::DiffModel, attr::MOI.AbstractModelAttribute)
    return MOI.supports(model.model, attr)
end

function MOI.set(model::DiffModel, attr::MOI.AbstractModelAttribute, value)
    MOI.set(model.model, attr, value)
end

function MOI.get(model::DiffModel, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.model, attr)
end
