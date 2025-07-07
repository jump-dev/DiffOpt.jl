# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# Constructs a Differentiable Optimizer model from a MOI Optimizer.
# Supports `forward_differentiate!` and `reverse_differentiate!` methods for solving and differentiating the model respectectively.

# ## Note
# Currently supports differentiating linear and quadratic programs only.

const MOIDD = MOI.Utilities.DoubleDicts

Base.@kwdef mutable struct InputCache
    dx::Dict{MOI.VariableIndex,Float64} = Dict{MOI.VariableIndex,Float64}()# dz for QP
    dy::Dict{MOI.ConstraintIndex,Float64} = Dict{MOI.ConstraintIndex,Float64}()
    # Dual sensitivity currently only works for NonLinearProgram
    # ds
    # dy #= [d\lambda, d\nu] for QP
    # FIXME Would it be possible to have a DoubleDict where the value depends
    #       on the function type ? Here, we need to create two dicts to have
    #       concrete value types.
    # `scalar_constraints` and `vector_constraints` includes `A` and `b` for CPs
    # or `G` and `h` for QPs
    parameter_constraints::Dict{MOI.ConstraintIndex,Float64} =
        Dict{MOI.ConstraintIndex,Float64}() # Specifically for NonLinearProgram
    scalar_constraints::MOIDD.DoubleDict{MOI.ScalarAffineFunction{Float64}} =
        MOIDD.DoubleDict{MOI.ScalarAffineFunction{Float64}}() # also includes G for QPs
    vector_constraints::MOIDD.DoubleDict{MOI.VectorAffineFunction{Float64}} =
        MOIDD.DoubleDict{MOI.VectorAffineFunction{Float64}}() # also includes G for QPs
    objective::Union{Nothing,MOI.AbstractScalarFunction} = nothing
    factorization::Union{Nothing,Function} = nothing
end

function Base.empty!(cache::InputCache)
    empty!(cache.dx)
    empty!(cache.dy)
    empty!(cache.parameter_constraints)
    empty!(cache.scalar_constraints)
    empty!(cache.vector_constraints)
    cache.objective = nothing
    return
end

"""
    reverse_differentiate!(model::MOI.ModelLike)

Wrapper method for the backward pass / reverse differentiation.
This method will consider as input a currently solved problem and differentials
with respect to the solution set with the [`ReverseVariablePrimal`](@ref) attribute.
The output problem data differentials can be queried with the
attributes [`ReverseObjectiveFunction`](@ref) and [`ReverseConstraintFunction`](@ref).
"""
function reverse_differentiate! end

"""
    forward_differentiate!(model::Optimizer)

Wrapper method for the forward pass.
This method will consider as input a currently solved problem and
differentials with respect to problem data set with
the [`ForwardObjectiveFunction`](@ref) and  [`ForwardConstraintFunction`](@ref) attributes.
The output solution differentials can be queried with the attribute
[`ForwardVariablePrimal`](@ref).
"""
function forward_differentiate! end

"""
    empty_input_sensitivities!(model::MOI.ModelLike)

Empty the input sensitivities of the model.
Sets to zero all the sensitivities set by the user with method such as:
- `MOI.set(model, DiffOpt.ReverseVariablePrimal(), variable_index, value)`
- `MOI.set(model, DiffOpt.ForwardObjectiveFunction(), expression)`
- `MOI.set(model, DiffOpt.ForwardConstraintFunction(), index, expression)`
"""
function empty_input_sensitivities! end

"""
    ForwardObjectiveFunction <: MOI.AbstractModelAttribute

A `MOI.AbstractModelAttribute` to set input data to forward differentiation, that
is, problem input data.
The possible values are any `MOI.AbstractScalarFunction`.
A `MOI.ScalarQuadraticFunction` can only be used in linearly constrained
quadratic models.

For instance, if the objective contains `θ * (x + 2y)`, for the purpose of
computing the derivative with respect to `θ`, the following should be set:
```julia
MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x + 2.0 * y)
```
where `x` and `y` are the relevant `MOI.VariableIndex`.
"""
struct ForwardObjectiveFunction <: MOI.AbstractModelAttribute end

"""
    NonLinearKKTJacobianFactorization <: MOI.AbstractModelAttribute

A `MOI.AbstractModelAttribute` to set which factorization method to use for the
nonlinear KKT Jacobian matrix, necessary for the implict function
diferentiation for `NonLinearProgram` models.

The function will be called with the following signature:
```julia
function factorization(M::SparseMatrixCSC{T<Real}, # The matrix to factorize
    model::NonLinearProgram.Model (can be ignored - useful for inertia correction)
)
```

* `M` is the matrix to factorize.
* `model` is the nonlinear model data that generated `M`. This can be used for
some factorization techniques such as LU with inertia correction.

Can be set by the user to use a custom factorization function:

```julia
MOI.set(model, DiffOpt.NonLinearKKTJacobianFactorization(), factorization)
```
"""
struct NonLinearKKTJacobianFactorization <: MOI.AbstractModelAttribute end

"""
    ForwardConstraintFunction <: MOI.AbstractConstraintAttribute

A `MOI.AbstractConstraintAttribute` to set input data to forward differentiation, that
is, problem input data.

For instance, if the scalar constraint of index `ci` contains `θ * (x + 2y) <= 5θ`,
for the purpose of computing the derivative with respect to `θ`, the following
should be set:
```julia
MOI.set(model, DiffOpt.ForwardConstraintFunction(), ci, 1.0 * x + 2.0 * y - 5.0)
```
Note that we use `-5` as the `ForwardConstraintFunction` sets the tangent of the
ConstraintFunction so we consider the expression `θ * (x + 2y - 5)`.
"""
struct ForwardConstraintFunction <: MOI.AbstractConstraintAttribute end

"""
    ForwardConstraintSet <: MOI.AbstractConstraintAttribute

A `MOI.AbstractConstraintAttribute` to set input data to forward differentiation, that
is, problem input data.

Currently, this only works for the set `MOI.Parameter`.
"""
struct ForwardConstraintSet <: MOI.AbstractConstraintAttribute end

"""
    ForwardVariablePrimal <: MOI.AbstractVariableAttribute

A `MOI.AbstractVariableAttribute` to get output data from forward
differentiation, that is, problem solution.

For instance, to get the tangent of the variable of index `vi` corresponding to
the tangents given to `ForwardObjectiveFunction` and `ForwardConstraintFunction`, do the
following:
```julia
MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi)
```
"""
struct ForwardVariablePrimal <: MOI.AbstractVariableAttribute end

MOI.is_set_by_optimize(::ForwardVariablePrimal) = true

"""
    ReverseVariablePrimal <: MOI.AbstractVariableAttribute

A `MOI.AbstractVariableAttribute` to set input data to
reverse differentiation, that is, problem solution.

For instance, to set the tangent of the variable of index `vi`, do the
following:
```julia
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x)
```
"""
struct ReverseVariablePrimal <: MOI.AbstractVariableAttribute end

"""
    ReverseConstraintDual <: MOI.AbstractConstraintAttribute

A `MOI.AbstractConstraintAttribute` to set input data from reverse differentiation.

For instance, to set the sensitivity `value` with respect to the dual variable of constraint
with index `ci` do the following:
```julia
MOI.set(model, DiffOpt.ReverseConstraintDual(), ci, value)
```
"""
struct ReverseConstraintDual <: MOI.AbstractConstraintAttribute end

"""
    ForwardConstraintDual <: MOI.AbstractConstraintAttribute

A `MOI.AbstractConstraintAttribute` to get output data from forward differentiation for the dual variable.

For instance, to get the sensitivity of the dual of constraint of index `ci` with respect to the parameter perturbation, do the following:

```julia
MOI.get(model, DiffOpt.ForwardConstraintDual(), ci)
```
"""
struct ForwardConstraintDual <: MOI.AbstractConstraintAttribute end

MOI.is_set_by_optimize(::ForwardConstraintDual) = true

"""
    ReverseObjectiveFunction <: MOI.AbstractModelAttribute

A `MOI.AbstractModelAttribute` to get output data to reverse differentiation,
that is, problem input data.

For instance, to get the tangent of the objective function corresponding to
the tangent given to `ReverseVariablePrimal`, do the
following:
```julia
func = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
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
struct ReverseObjectiveFunction <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::ReverseObjectiveFunction) = true

"""
    ReverseConstraintFunction

An `MOI.AbstractConstraintAttribute` to get output data to reverse differentiation, that
is, problem input data.

For instance, if the following returns `x + 2y + 5`, it means that the tangent
has coordinate `1` for the coefficient of `x`, coordinate `2` for the
coefficient of `y` and `5` for the function constant.
If the constraint is of the form `func == constant` or `func <= constant`,
the tangent for the constant on the right-hand side is `-5`.
```julia
MOI.get(model, DiffOpt.ReverseConstraintFunction(), ci)
```
"""
struct ReverseConstraintFunction <: MOI.AbstractConstraintAttribute end

MOI.is_set_by_optimize(::ReverseConstraintFunction) = true

"""
    ReverseConstraintSet

An `MOI.AbstractConstraintAttribute` to get output data to reverse differentiation, that
is, problem input data.

Currently, this only works for the set `MOI.Parameter`.
"""
struct ReverseConstraintSet <: MOI.AbstractConstraintAttribute end

MOI.is_set_by_optimize(::ReverseConstraintSet) = true

"""
    DifferentiateTimeSec()

A model attribute for the total elapsed time (in seconds) for computing
the differentiation information.
"""
struct DifferentiateTimeSec <: MOI.AbstractModelAttribute end

MOI.attribute_value_type(::DifferentiateTimeSec) = Float64

MOI.is_set_by_optimize(::DifferentiateTimeSec) = true

"""
    abstract type AbstractModel <: MOI.ModelLike end

Model supporting [`forward_differentiate!`](@ref) and
[`reverse_differentiate!`](@ref).
"""
abstract type AbstractModel <: MOI.ModelLike end

function empty_input_sensitivities!(model::AbstractModel)
    empty!(model.input_cache)
    return
end

MOI.supports_incremental_interface(::AbstractModel) = true

function MOI.is_valid(model::AbstractModel, idx::MOI.Index)
    return MOI.is_valid(model.model, idx)
end

function MOI.add_variable(model::AbstractModel)
    return MOI.add_variable(model.model)
end

function MOI.add_variables(model::AbstractModel, n)
    return MOI.add_variables(model.model, n)
end

# TODO: add support for add_constrained_variable(s) and supports_

function MOI.Utilities.pass_nonvariable_constraints(
    dest::AbstractModel,
    src::MOI.ModelLike,
    idxmap::MOI.Utilities.IndexMap,
    constraint_types,
)
    return MOI.Utilities.pass_nonvariable_constraints(
        dest.model,
        src,
        idxmap,
        constraint_types,
    )
end

function MOI.Utilities.final_touch(model::AbstractModel, index_map)
    return MOI.Utilities.final_touch(model.model, index_map)
end

function MOI.supports_constraint(
    model::AbstractModel,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(model.model, F, S)
end

function MOI.add_constraint(
    model::AbstractModel,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet,
)
    return MOI.add_constraint(model.model, func, set)
end

function _enlarge_set(vec::Vector, idx, value)
    m = last(idx)
    if length(vec) < m
        n = length(vec)
        resize!(vec, m)
        fill!(view(vec, (n+1):m), NaN)
    end
    vec[idx] = value
    return
end

# The following `supports` methods are needed because
# `MOI.set(::MOI.ModelLike, ::SlackBridgePrimalDualStart, ::SlackBridge, ::Nothing)`
# checks that the model supports these starting value attributes.
function MOI.supports(
    ::AbstractModel,
    ::MOI.VariablePrimalStart,
    ::Type{<:MOI.VariableIndex},
)
    return true
end

function MOI.supports(
    ::AbstractModel,
    ::Union{MOI.ConstraintDualStart,MOI.ConstraintPrimalStart},
    ::Type{<:MOI.ConstraintIndex},
)
    return true
end

function MOI.get(
    model::AbstractModel,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    return model.x[vi.value]
end

function MOI.set(
    model::AbstractModel,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value,
)
    MOI.throw_if_not_valid(model, vi)
    return _enlarge_set(model.x, vi.value, value)
end

function MOI.set(model::AbstractModel, ::ForwardObjectiveFunction, objective)
    model.input_cache.objective = objective
    return
end

function MOI.set(
    model::AbstractModel,
    ::NonLinearKKTJacobianFactorization,
    factorization::Function,
)
    model.input_cache.factorization = factorization
    return
end

function MOI.set(
    model::AbstractModel,
    ::ReverseVariablePrimal,
    vi::MOI.VariableIndex,
    val,
)
    model.input_cache.dx[vi] = val
    return
end

function MOI.set(
    model::AbstractModel,
    ::ReverseConstraintDual,
    vi::MOI.ConstraintIndex,
    val,
)
    model.input_cache.dy[vi] = val
    return
end

function MOI.set(
    model::AbstractModel,
    ::ForwardConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    if MOI.supports_add_constrained_variable(model.model, MOI.Parameter{T})
        error(
            "The model with type $(typeof(model)) does support Parameters, so setting ForwardConstraintFunction fails.",
        )
    end
    model.input_cache.scalar_constraints[ci] = func
    return
end

function MOI.set(
    model::AbstractModel,
    ::ForwardConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.VectorAffineFunction{T},S},
    func::MOI.VectorAffineFunction{T},
) where {T,S}
    if MOI.supports_add_constrained_variable(model.model, MOI.Parameter{T})
        error(
            "The model with type $(typeof(model)) does support Parameters, so setting ForwardConstraintFunction fails.",
        )
    end
    model.input_cache.vector_constraints[ci] = func
    return
end

function MOI.set(
    model::AbstractModel,
    ::ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    set::MOI.Parameter{T},
) where {T}
    if !MOI.supports_add_constrained_variable(
        model.model,
        MOI.Parameter{T},
    )
        error(
            "The model with type $(typeof(model)) does not support Parameters",
        )
    end
    model.input_cache.parameter_constraints[ci] = set.value
    return
end

function lazy_combination(op::F, α, a, β, b) where {F<:Function}
    return LazyArrays.ApplyArray(
        op,
        LazyArrays.@~(α .* a),
        LazyArrays.@~(β .* b),
    )
end

function lazy_combination(
    op::F,
    α,
    a,
    β,
    b,
    I::AbstractUnitRange,
) where {F<:Function}
    return lazy_combination(op, α, view(a, I), β, view(b, I))
end

function lazy_combination(
    op::F,
    a,
    b,
    i::Integer,
    args::Vararg{Any,N},
) where {F<:Function,N}
    return lazy_combination(op, a[i], b, b[i], a, args...)
end

function lazy_combination(
    op::F,
    a,
    b,
    i::AbstractUnitRange,
    I::AbstractUnitRange,
) where {F<:Function}
    return lazy_combination(op, view(a, i), b, view(b, i), a, I)
end

function _lazy_affine(vector, constant::Number)
    return VectorScalarAffineFunction(vector, constant)
end

_lazy_affine(matrix, vector) = MatrixVectorAffineFunction(matrix, vector)

function _get_db end

function _get_dA end

function MOI.get(
    model::AbstractModel,
    ::ReverseConstraintFunction,
    ci::MOI.ConstraintIndex,
)
    return _lazy_affine(_get_dA(model, ci), _get_db(model, ci))
end

"""
    π(v::Vector{Float64}, model::MOI.ModelLike, cones::ProductOfSets)

Given a `model`, its `cones`, find the projection of the vectors `v`
of length equal to the number of rows in the conic form onto the cartesian
product of the cones corresponding to these rows.
For more info, refer to https://github.com/matbesancon/MathOptSetDistances.jl
"""
function π(v::Vector{T}, model::MOI.ModelLike, cones::ProductOfSets) where {T}
    return map_rows(model, cones, Flattened{T}()) do ci, r
        return MOSD.projection_on_set(
            MOSD.DefaultDistance(),
            v[r],
            MOI.dual_set(MOI.get(model, MOI.ConstraintSet(), ci)),
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
function Dπ(v::Vector{T}, model::MOI.ModelLike, cones::ProductOfSets) where {T}
    return BlockDiagonals.BlockDiagonal(
        map_rows(model, cones, Nested{Matrix{T}}()) do ci, r
            return MOSD.projection_gradient_on_set(
                MOSD.DefaultDistance(),
                v[r],
                MOI.dual_set(MOI.get(model, MOI.ConstraintSet(), ci)),
            )
        end,
    )
end

# See the docstring of `map_rows`.
struct Nested{T} end

struct Flattened{T} end

# Store in `x` the values `y` corresponding to the rows `r` and the `k`th
# constraint.
function _assign_mapped!(x, y, r, k, ::Nested)
    return x[k] = y
end

function _assign_mapped!(x, y, r, k, ::Flattened)
    return x[r] = y
end

# Map the rows corresponding to `F`-in-`S` constraints and store it in `x`.
function _map_rows!(
    f::Function,
    x::Vector,
    model,
    cones::ProductOfSets,
    ::Type{F},
    ::Type{S},
    map_mode,
    k,
) where {F,S}
    for ci in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
        r = MOI.Utilities.rows(cones, ci)
        k += 1
        _assign_mapped!(x, f(ci, r), r, k, map_mode)
    end
    return k
end

# Allocate a vector for storing the output of `map_rows`.
function _allocate_rows(cones, ::Nested{T}) where {T}
    return Vector{T}(undef, length(cones.dimension))
end

function _allocate_rows(cones, ::Flattened{T}) where {T}
    return Vector{T}(undef, MOI.dimension(cones))
end

"""
    map_rows(f::Function, model, cones::ProductOfSets, map_mode::Union{Nested{T}, Flattened{T}})

Given a `model`, its `cones` and `map_mode` of type `Nested` (resp.
`Flattened`), return a `Vector{T}` of length equal to the number of cones (resp.
rows) in the conic form where the value for the index (resp. rows) corresponding
to each cone is equal to `f(ci, r)` where `ci` is the corresponding constraint
index in `model` and `r` is a `UnitRange` of the corresponding rows in the conic
form.
"""
function map_rows(
    f::Function,
    model,
    cones::ProductOfSets,
    map_mode::Union{Nested,Flattened},
)
    x = _allocate_rows(cones, map_mode)
    k = 0
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        # Function barrier for type unstability of `F` and `S`
        # `con_map` is a `MOI.Utilities.DoubleDicts.MainIndexDoubleDict`, we index it at `F, S`
        # which returns a `MOI.Utilities.DoubleDicts.IndexWithType{F, S}` which is type stable.
        # If we have a small number of different constraint types and many
        # constraint of each type, this mostly removes type unstabilities
        # as most the time is in `_map_rows!` which is type stable.
        k = _map_rows!(f, x, model, cones, F, S, map_mode, k)
    end
    return x
end

function _fill(neg::Function, gradient_cache, input_cache, cones, args...)
    return _fill(S -> true, neg, gradient_cache, input_cache, cones, args...)
end

function _fill(
    filter::Function,
    neg::Function,
    gradient_cache,
    input_cache,
    cones,
    args...,
)
    for (F, S) in keys(input_cache.scalar_constraints.dict)
        filter(S) || continue
        _fill(args..., neg(S), cones, input_cache.scalar_constraints[F, S])
    end
    for (F, S) in keys(input_cache.vector_constraints.dict)
        _fill(args..., neg(S), cones, input_cache.vector_constraints[F, S])
    end
    return
end

function _fill(vector::Vector, neg::Bool, cones, dict)
    for (ci, func) in dict
        r = MOI.Utilities.rows(cones, ci)
        vector[r] = neg ? -MOI.constant(func) : MOI.constant(func)
    end
    return
end

function _fill(I::Vector, J::Vector, V::Vector, neg::Bool, cones, dict)
    for (ci, func) in dict
        r = MOI.Utilities.rows(cones, ci)
        for term in func.terms
            _push_term(I, J, V, neg, r, term)
        end
    end
    return
end

function _push_term(
    I::Vector,
    J::Vector,
    V::Vector,
    neg::Bool,
    r::Integer,
    term::MOI.ScalarAffineTerm,
)
    push!(I, r)
    push!(J, term.variable.value)
    return push!(V, neg ? -term.coefficient : term.coefficient)
end

function _push_term(
    I::Vector,
    J::Vector,
    V::Vector,
    neg::Bool,
    r::AbstractUnitRange,
    term::MOI.VectorAffineTerm,
)
    return _push_term(I, J, V, neg, r[term.output_index], term.scalar_term)
end

function MOI.supports(model::AbstractModel, attr::MOI.AbstractModelAttribute)
    return MOI.supports(model.model, attr)
end

function MOI.set(model::AbstractModel, attr::MOI.AbstractModelAttribute, value)
    return MOI.set(model.model, attr, value)
end

function MOI.get(model::AbstractModel, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.model, attr)
end
