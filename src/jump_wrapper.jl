"""
    diff_model(optimizer_constructor; with_parametric_opt_interface::Bool = true, with_bridge_type = Float64, with_cache::Bool = true)

Create a JuMP model with a differentiable optimizer. The optimizer is created
using `optimizer_constructor`. This model will try to select the proper
differentiable optimization method based on the problem structure.

See also: [`nonlinear_diff_model`](@ref), [`conic_diff_model`](@ref), [`quadratic_diff_model`](@ref).
"""
function diff_model(
    optimizer_constructor;
    with_parametric_opt_interface::Bool = true,
    with_bridge_type = Float64,
    with_cache::Bool = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_parametric_opt_interface = with_parametric_opt_interface,
        with_bridge_type = with_bridge_type,
        with_cache = with_cache,
    )
    return JuMP.direct_model(inner)
end

"""
    nonlinear_diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache::Bool = true)

Create a JuMP model with a differentiable optimizer for nonlinear programs.
The optimizer is created using `optimizer_constructor`.

See also: [`conic_diff_model`](@ref), [`quadratic_diff_model`](@ref), [`diff_model`](@ref).
"""
function nonlinear_diff_model(
    optimizer_constructor;
    with_bridge_type = Float64,
    with_cache::Bool = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_parametric_opt_interface = false,
        with_bridge_type = with_bridge_type,
        with_cache = with_cache,
    )
    MOI.set(inner, ModelConstructor(), NonLinearProgram.Model)
    return JuMP.direct_model(inner)
end

"""
    conic_diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache::Bool = true)

Create a JuMP model with a differentiable optimizer for conic programs.
The optimizer is created using `optimizer_constructor`.

See also: [`nonlinear_diff_model`](@ref), [`quadratic_diff_model`](@ref), [`diff_model`](@ref).
"""
function conic_diff_model(
    optimizer_constructor;
    with_bridge_type = Float64,
    with_cache::Bool = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_parametric_opt_interface = true,
        with_bridge_type = with_bridge_type,
        with_cache = with_cache,
    )
    MOI.set(inner, ModelConstructor(), ConicProgram.Model)
    return JuMP.direct_model(inner)
end

"""
    quadratic_diff_model(optimizer_constructor; with_bridge_type = Float64, with_cache::Bool = true)

Create a JuMP model with a differentiable optimizer for quadratic programs.
The optimizer is created using `optimizer_constructor`.

See also: [`nonlinear_diff_model`](@ref), [`conic_diff_model`](@ref), [`diff_model`](@ref).
"""
function quadratic_diff_model(
    optimizer_constructor;
    with_bridge_type = Float64,
    with_cache::Bool = true,
)
    inner = diff_optimizer(
        optimizer_constructor;
        with_parametric_opt_interface = true,
        with_bridge_type = with_bridge_type,
        with_cache = with_cache,
    )
    MOI.set(inner, ModelConstructor(), QuadraticProgram.Model)
    return JuMP.direct_model(inner)
end

"""
    set_forward_parameter(model::JuMP.Model, variable::JuMP.VariableRef, value::Number)

Set the value of a parameter input sensitivity for forward mode.
"""
function set_forward_parameter(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    return MOI.set(
        model,
        ForwardConstraintSet(),
        ParameterRef(variable),
        Parameter(value),
    )
end

"""
    get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a parameter output sensitivity for reverse mode.
"""
function get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)
    return MOI.get(model, ReverseConstraintSet(), ParameterRef(variable)).value
end

"""
    set_reverse_variable(model::JuMP.Model, variable::JuMP.VariableRef, value::Number)

Set the value of a variable input sensitivity for reverse mode.
"""
function set_reverse_variable(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    return MOI.set(model, ReverseVariablePrimal(), variable, value)
end

"""
    get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a variable output sensitivity for forward mode.
"""
function get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)
    return MOI.get(model, ForwardVariablePrimal(), variable)
end
