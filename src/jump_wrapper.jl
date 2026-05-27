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

Equivalent to `set_attribute(variable, DiffOpt.ForwardParameterValue(), value)`,
which is the preferred form going forward.
"""
function set_forward_parameter(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    Base.depwarn(
        "`DiffOpt.set_forward_parameter(model, variable, value)` is deprecated; use `set_attribute(variable, DiffOpt.ForwardParameterValue(), value)`.",
        :set_forward_parameter,
    )
    return JuMP.set_attribute(variable, ForwardParameterValue(), value)
end

"""
    get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a parameter output sensitivity for reverse mode.

Equivalent to `get_attribute(variable, DiffOpt.ReverseParameterValue())`,
which is the preferred form going forward.
"""
function get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)
    Base.depwarn(
        "`DiffOpt.get_reverse_parameter(model, variable)` is deprecated; use `get_attribute(variable, DiffOpt.ReverseParameterValue())`.",
        :get_reverse_parameter,
    )
    return JuMP.get_attribute(variable, ReverseParameterValue())
end

"""
    set_reverse_variable(model::JuMP.Model, variable::JuMP.VariableRef, value::Number)

Set the value of a variable input sensitivity for reverse mode.

Equivalent to `set_attribute(variable, DiffOpt.ReverseVariablePrimal(), value)`,
which is the preferred form going forward.
"""
function set_reverse_variable(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    Base.depwarn(
        "`DiffOpt.set_reverse_variable(model, variable, value)` is deprecated; use `set_attribute(variable, DiffOpt.ReverseVariablePrimal(), value)`.",
        :set_reverse_variable,
    )
    return JuMP.set_attribute(variable, ReverseVariablePrimal(), value)
end

"""
    get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a variable output sensitivity for forward mode.

Equivalent to `get_attribute(variable, DiffOpt.ForwardVariablePrimal())`,
which is the preferred form going forward.
"""
function get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)
    Base.depwarn(
        "`DiffOpt.get_forward_variable(model, variable)` is deprecated; use `get_attribute(variable, DiffOpt.ForwardVariablePrimal())`.",
        :get_forward_variable,
    )
    return JuMP.get_attribute(variable, ForwardVariablePrimal())
end

"""
    set_reverse_objective(model::JuMP.Model, value::Number)

Set the value of the objective input sensitivity for reverse mode.

Equivalent to `set_attribute(model, DiffOpt.ReverseObjectiveValue(), value)`,
which is the preferred form going forward.
"""
function set_reverse_objective(model::JuMP.Model, value::Number)
    Base.depwarn(
        "`DiffOpt.set_reverse_objective(model, value)` is deprecated; use `set_attribute(model, DiffOpt.ReverseObjectiveValue(), value)`.",
        :set_reverse_objective,
    )
    return JuMP.set_attribute(model, ReverseObjectiveValue(), value)
end

"""
    get_forward_objective(model::JuMP.Model)

Get the value of the objective output sensitivity for forward mode.

Equivalent to `get_attribute(model, DiffOpt.ForwardObjectiveSensitivity())`,
which is the preferred form going forward.
"""
function get_forward_objective(model::JuMP.Model)
    Base.depwarn(
        "`DiffOpt.get_forward_objective(model)` is deprecated; use `get_attribute(model, DiffOpt.ForwardObjectiveSensitivity())`.",
        :get_forward_objective,
    )
    return JuMP.get_attribute(model, ForwardObjectiveSensitivity())
end

"""
    set_forward_objective_function(model::JuMP.Model, func)

Set the function to be used for forward mode differentiation of the objective.

Equivalent to `set_attribute(model, DiffOpt.ForwardObjectiveFunction(), func)`,
which is the preferred form going forward.
"""
function set_forward_objective_function(
    model::JuMP.Model,
    func::JuMP.AbstractJuMPScalar,
)
    Base.depwarn(
        "`DiffOpt.set_forward_objective_function(model, func)` is deprecated; use `set_attribute(model, DiffOpt.ForwardObjectiveFunction(), func)`.",
        :set_forward_objective_function,
    )
    return JuMP.set_attribute(model, ForwardObjectiveFunction(), func)
end

function set_forward_objective_function(model::JuMP.Model, value::Number)
    Base.depwarn(
        "`DiffOpt.set_forward_objective_function(model, value)` is deprecated; use `set_attribute(model, DiffOpt.ForwardObjectiveFunction(), value)`.",
        :set_forward_objective_function,
    )
    return JuMP.set_attribute(model, ForwardObjectiveFunction(), value)
end

"""
    set_forward_constraint_function(model::JuMP.Model, con_ref::JuMP.ConstraintRef, func)

Set the function to be used for forward mode differentiation of a constraint.

Equivalent to `set_attribute(con_ref, DiffOpt.ForwardConstraintFunction(), func)`,
which is the preferred form going forward.
"""
function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        M,
        <:MOI.ConstraintIndex{<:MOI.AbstractScalarFunction},
    },
    func::JuMP.AbstractJuMPScalar,
) where {M}
    Base.depwarn(
        "`DiffOpt.set_forward_constraint_function(model, con_ref, func)` is deprecated; use `set_attribute(con_ref, DiffOpt.ForwardConstraintFunction(), func)`.",
        :set_forward_constraint_function,
    )
    return JuMP.set_attribute(con_ref, ForwardConstraintFunction(), func)
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        M,
        <:MOI.ConstraintIndex{<:MOI.AbstractScalarFunction},
    },
    value::Number,
) where {M}
    Base.depwarn(
        "`DiffOpt.set_forward_constraint_function(model, con_ref, value)` is deprecated; use `set_attribute(con_ref, DiffOpt.ForwardConstraintFunction(), value)`.",
        :set_forward_constraint_function,
    )
    return JuMP.set_attribute(con_ref, ForwardConstraintFunction(), value)
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        <:JuMP.AbstractModel,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
    },
    value::AbstractArray{<:JuMP.AbstractJuMPScalar},
)
    Base.depwarn(
        "`DiffOpt.set_forward_constraint_function(model, con_ref, value)` is deprecated; use `set_attribute(con_ref, DiffOpt.ForwardConstraintFunction(), value)`.",
        :set_forward_constraint_function,
    )
    return JuMP.set_attribute(con_ref, ForwardConstraintFunction(), value)
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        <:JuMP.AbstractModel,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
    },
    value::AbstractArray{<:Number},
)
    Base.depwarn(
        "`DiffOpt.set_forward_constraint_function(model, con_ref, value)` is deprecated; use `set_attribute(con_ref, DiffOpt.ForwardConstraintFunction(), value)`.",
        :set_forward_constraint_function,
    )
    return JuMP.set_attribute(con_ref, ForwardConstraintFunction(), value)
end

"""
    get_forward_constraint_dual(model::JuMP.Model, con_ref::JuMP.ConstraintRef)

Get the value of a constraint dual output sensitivity for forward mode.

Equivalent to `get_attribute(con_ref, DiffOpt.ForwardConstraintDual())`,
which is the preferred form going forward.
"""
function get_forward_constraint_dual(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef,
)
    Base.depwarn(
        "`DiffOpt.get_forward_constraint_dual(model, con_ref)` is deprecated; use `get_attribute(con_ref, DiffOpt.ForwardConstraintDual())`.",
        :get_forward_constraint_dual,
    )
    return JuMP.get_attribute(con_ref, ForwardConstraintDual())
end

"""
    get_reverse_objective_function(model::JuMP.Model)

Get the function to be used for reverse mode differentiation of the objective.

Equivalent to `get_attribute(model, DiffOpt.ReverseObjectiveFunction())`,
which is the preferred form going forward.
"""
function get_reverse_objective_function(model::JuMP.Model)
    Base.depwarn(
        "`DiffOpt.get_reverse_objective_function(model)` is deprecated; use `get_attribute(model, DiffOpt.ReverseObjectiveFunction())`.",
        :get_reverse_objective_function,
    )
    return JuMP.get_attribute(model, ReverseObjectiveFunction())
end

"""
    get_reverse_constraint_function(model::JuMP.Model, con_ref::JuMP.ConstraintRef)

Get the function to be used for reverse mode differentiation of a constraint.

Equivalent to `get_attribute(con_ref, DiffOpt.ReverseConstraintFunction())`,
which is the preferred form going forward.
"""
function get_reverse_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef,
)
    Base.depwarn(
        "`DiffOpt.get_reverse_constraint_function(model, con_ref)` is deprecated; use `get_attribute(con_ref, DiffOpt.ReverseConstraintFunction())`.",
        :get_reverse_constraint_function,
    )
    return JuMP.get_attribute(con_ref, ReverseConstraintFunction())
end
