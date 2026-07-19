# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache_type = Float64, with_outer_cache = true)

Create a JuMP model with a differentiable optimizer. The optimizer is created
using `optimizer_constructor`. This model will try to select the proper
differentiable optimization method based on the problem structure.

See also: [`nonlinear_diff_model`](@ref), [`conic_diff_model`](@ref), [`quadratic_diff_model`](@ref).
"""
function diff_model(
    optimizer_constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
    return JuMP.direct_model(inner)
end

"""
    nonlinear_diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache_type = Float64, with_outer_cache = true)

Create a JuMP model with a differentiable optimizer for nonlinear programs.
The optimizer is created using `optimizer_constructor`.

See also: [`conic_diff_model`](@ref), [`quadratic_diff_model`](@ref), [`diff_model`](@ref).
"""
function nonlinear_diff_model(
    optimizer_constructor;
    with_parametric_opt_interface = false,
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
    MOI.set(inner, ModelConstructor(), NonLinearProgram.Model)
    return JuMP.direct_model(inner)
end

"""
    conic_diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache_type = Float64, with_outer_cache = true)

Create a JuMP model with a differentiable optimizer for conic programs.
The optimizer is created using `optimizer_constructor`.

See also: [`nonlinear_diff_model`](@ref), [`quadratic_diff_model`](@ref), [`diff_model`](@ref).
"""
function conic_diff_model(
    optimizer_constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
    MOI.set(inner, ModelConstructor(), ConicProgram.Model)
    return JuMP.direct_model(inner)
end

"""
    quadratic_diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache_type = Float64, with_outer_cache = true)

Create a JuMP model with a differentiable optimizer for quadratic programs.
The optimizer is created using `optimizer_constructor`.

See also: [`nonlinear_diff_model`](@ref), [`conic_diff_model`](@ref), [`diff_model`](@ref).
"""
function quadratic_diff_model(
    optimizer_constructor;
    with_bridge_type = Float64,
    with_cache_type = Float64,
    with_outer_cache = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_bridge_type,
        with_cache_type,
        with_outer_cache,
    )
    MOI.set(inner, ModelConstructor(), QuadraticProgram.Model)
    return JuMP.direct_model(inner)
end

"""
    set_forward_parameter(model::JuMP.Model, variable::JuMP.VariableRef, value::Number)

Set the value of a parameter input sensitivity for forward mode.

Equivalent to `set_attribute(variable, DiffOpt.ForwardParameterValue(), value)`.
"""
function set_forward_parameter(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    JuMP.check_belongs_to_model(variable, model)
    return MOI.set(
        JuMP.backend(model),
        ForwardConstraintSet(),
        JuMP.index(ParameterRef(variable)),
        MOI.Parameter(value),
    )
end

"""
    get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a parameter output sensitivity for reverse mode.

Equivalent to `get_attribute(variable, DiffOpt.ReverseParameterValue())`.
"""
function get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)
    JuMP.check_belongs_to_model(variable, model)
    return MOI.get(
        JuMP.backend(model),
        ReverseConstraintSet(),
        JuMP.index(ParameterRef(variable)),
    ).value
end

"""
    set_reverse_variable(model::JuMP.Model, variable::JuMP.VariableRef, value::Number)

Set the value of a variable input sensitivity for reverse mode.

Equivalent to `set_attribute(variable, DiffOpt.ReverseVariablePrimal(), value)`.
"""
function set_reverse_variable(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    JuMP.check_belongs_to_model(variable, model)
    return MOI.set(
        JuMP.backend(model),
        ReverseVariablePrimal(),
        JuMP.index(variable),
        value,
    )
end

"""
    set_reverse_constraint_dual(model::JuMP.Model, con_ref::JuMP.ConstraintRef, value::Number)

Set the value of a constraint dual input sensitivity for reverse mode.

Equivalent to `set_attribute(con_ref, DiffOpt.ReverseConstraintDual(), value)`.
"""
function set_reverse_constraint_dual(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef,
    value::Number,
)
    JuMP.check_belongs_to_model(con_ref, model)
    return MOI.set(
        JuMP.backend(model),
        ReverseConstraintDual(),
        JuMP.index(con_ref),
        value,
    )
end

"""
    get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a variable output sensitivity for forward mode.

Equivalent to `get_attribute(variable, DiffOpt.ForwardVariablePrimal())`.
"""
function get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)
    JuMP.check_belongs_to_model(variable, model)
    return _moi_get_result(
        JuMP.backend(model),
        ForwardVariablePrimal(),
        JuMP.index(variable),
    )
end

"""
    set_reverse_objective(model::JuMP.Model, value::Number)

Set the value of the objective input sensitivity for reverse mode.

Equivalent to `set_attribute(model, DiffOpt.ReverseObjectiveValue(), value)`.
"""
function set_reverse_objective(model::JuMP.Model, value::Number)
    return MOI.set(model, ReverseObjectiveValue(), value)
end

"""
    get_forward_objective(model::JuMP.Model)

Get the value of the objective output sensitivity for forward mode.

Equivalent to `get_attribute(model, DiffOpt.ForwardObjectiveValue())`.
"""
function get_forward_objective(model::JuMP.Model)
    return MOI.get(model, ForwardObjectiveValue())
end

"""
    set_forward_objective_function(model::JuMP.Model, func)

Set the function to be used for forward mode differentiation of the objective.

Equivalent to `set_attribute(model, DiffOpt.ForwardObjectiveFunction(), func)`.
"""
function set_forward_objective_function(
    model::JuMP.Model,
    func::JuMP.AbstractJuMPScalar,
)
    return MOI.set(
        JuMP.backend(model),
        ForwardObjectiveFunction(),
        JuMP.moi_function(func),
    )
end

function set_forward_objective_function(model::JuMP.Model, value::Number)
    return MOI.set(
        JuMP.backend(model),
        ForwardObjectiveFunction(),
        JuMP.moi_function(JuMP.AffExpr(value)),
    )
end

"""
    set_forward_constraint_function(model::JuMP.Model, con_ref::JuMP.ConstraintRef, func)

Set the function to be used for forward mode differentiation of a constraint.

Equivalent to `set_attribute(con_ref, DiffOpt.ForwardConstraintFunction(), func)`.
"""
function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        M,
        <:MOI.ConstraintIndex{<:MOI.AbstractScalarFunction},
    },
    func::JuMP.AbstractJuMPScalar,
) where {M}
    JuMP.check_belongs_to_model(con_ref, model)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        JuMP.moi_function(func),
    )
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        M,
        <:MOI.ConstraintIndex{<:MOI.AbstractScalarFunction},
    },
    value::Number,
) where {M}
    return set_forward_constraint_function(model, con_ref, JuMP.AffExpr(value))
end

# Similar to `JuMP.set_start_value` for vector `ConstraintRef` in
# JuMP/src/constraints.jl
function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        <:JuMP.AbstractModel,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
    },
    value::AbstractArray{<:JuMP.AbstractJuMPScalar},
)
    JuMP.check_belongs_to_model(con_ref, model)
    JuMP.check_belongs_to_model.(value, model)
    v = JuMP.vectorize(value, con_ref.shape)
    return MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        JuMP.moi_function(v),
    )
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        <:JuMP.AbstractModel,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
    },
    value::AbstractArray{<:Number},
)
    return set_forward_constraint_function(model, con_ref, JuMP.AffExpr.(value))
end

"""
    get_forward_constraint_dual(model::JuMP.Model, con_ref::JuMP.ConstraintRef)

Get the value of a constraint dual output sensitivity for forward mode.

Equivalent to `get_attribute(con_ref, DiffOpt.ForwardConstraintDual())`.
"""
function get_forward_constraint_dual(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef,
)
    JuMP.check_belongs_to_model(con_ref, model)
    moi_func = MOI.get(
        JuMP.backend(model),
        ForwardConstraintDual(),
        JuMP.index(con_ref),
    )
    return JuMP.jump_function(model, moi_func)
end

"""
    get_reverse_objective_function(model::JuMP.Model)

Get the function to be used for reverse mode differentiation of the objective.

Equivalent to `get_attribute(model, DiffOpt.ReverseObjectiveFunction())`.
"""
function get_reverse_objective_function(model::JuMP.Model)
    func = MOI.get(JuMP.backend(model), ReverseObjectiveFunction())
    return JuMP.jump_function(model, func)
end

"""
    get_reverse_constraint_function(model::JuMP.Model, con_ref::JuMP.ConstraintRef)

Get the function to be used for reverse mode differentiation of a constraint.

Equivalent to `get_attribute(con_ref, DiffOpt.ReverseConstraintFunction())`.
"""
function get_reverse_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef,
)
    JuMP.check_belongs_to_model(con_ref, model)
    moi_func = MOI.get(
        JuMP.backend(model),
        ReverseConstraintFunction(),
        JuMP.index(con_ref),
    )
    return JuMP.jump_function(model, moi_func)
end
