"""
"""
function diff_model(
    optimizer_constructor;
    with_parametric_opt_interface::Bool = false,
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

# nonlinear_diff_model
# conic_diff_model
# quadratic_diff_model

"""
"""
function set_forward_parameter(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    return MOI.set(
        model,
        DiffOpt.ForwardConstraintSet(),
        ParameterRef(variable),
        value,
    )
end

"""
"""
function get_reverse_parameter(model::JuMP.Model, variable::JuMP.VariableRef)
    return MOI.get(
        model,
        DiffOpt.ReverseConstraintSet(),
        ParameterRef(variable),
    )
end

"""
"""
function set_reverse_variable(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    return MOI.set(model, DiffOpt.ReverseVariablePrimal(), variable, value)
end

"""
"""
function get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)
    return MOI.get(model, DiffOpt.ForwardVariablePrimal(), variable)
end
