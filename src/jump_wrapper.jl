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
    get_forward_variable(model::JuMP.Model, variable::JuMP.VariableRef)

Get the value of a variable output sensitivity for forward mode.
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
"""
function set_reverse_objective(model::JuMP.Model, value::Number)
    return MOI.set(model, ReverseObjectiveSensitivity(), value)
end

"""
    get_forward_objective(model::JuMP.Model)

Get the value of the objective output sensitivity for forward mode.
"""
function get_forward_objective(model::JuMP.Model)
    return MOI.get(model, ForwardObjectiveSensitivity())
end

"""
    set_forward_objective_function(model::JuMP.Model, func)

Set the function to be used for forward mode differentiation of the objective.
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
    JuMP.check_belongs_to_model(con_ref, model)
    return MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        JuMP.moi_function(JuMP.AffExpr(value)),
    )
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        M,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
    },
    value::AbstractArray{<:JuMP.AbstractJuMPScalar},
) where {M}
    JuMP.check_belongs_to_model(con_ref, model)
    JuMP.check_belongs_to_model.(value, model)
    return MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        JuMP.moi_function(value),
    )
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        M,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
    },
    value::AbstractArray{<:Number},
) where {M}
    JuMP.check_belongs_to_model(con_ref, model)
    return MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        JuMP.moi_function(JuMP.AffExpr.(value)),
    )
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{
        <:JuMP.AbstractModel,
        <:MOI.ConstraintIndex{<:MOI.AbstractVectorFunction},
        S,
    },
    value::AbstractMatrix{<:Number},
) where {S<:Union{JuMP.SquareMatrixShape,JuMP.SymmetricMatrixShape}}
    if !LinearAlgebra.issymmetric(value)
        error(
            "ForwardConstraintFunction perturbation matrix must be " *
            "symmetric for PSD cone constraints.",
        )
    end
    JuMP.check_belongs_to_model(con_ref, model)
    v = JuMP.vectorize(value, con_ref.shape)
    func = JuMP.moi_function(JuMP.AffExpr.(v))
    MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        func,
    )
    return
end

function set_forward_constraint_function(
    model::JuMP.Model,
    con_ref::JuMP.ConstraintRef{<:JuMP.AbstractModel,<:MOI.ConstraintIndex,S},
    value::AbstractMatrix{<:JuMP.AbstractJuMPScalar},
) where {S<:Union{JuMP.SquareMatrixShape,JuMP.SymmetricMatrixShape}}
    if !LinearAlgebra.issymmetric(value)
        error(
            "ForwardConstraintFunction perturbation matrix must be " *
            "symmetric for PSD cone constraints.",
        )
    end
    JuMP.check_belongs_to_model(con_ref, model)
    JuMP.check_belongs_to_model.(value, model)
    v = JuMP.vectorize(value, con_ref.shape)
    func = JuMP.moi_function(v)
    MOI.set(
        JuMP.backend(model),
        ForwardConstraintFunction(),
        JuMP.index(con_ref),
        func,
    )
    return
end

"""
    get_forward_constraint_dual(model::JuMP.Model, con_ref::JuMP.ConstraintRef)

Get the value of a constraint dual output sensitivity for forward mode.
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
"""
function get_reverse_objective_function(model::JuMP.Model)
    func = MOI.get(JuMP.backend(model), ReverseObjectiveFunction())
    return JuMP.jump_function(model, func)
end

"""
    get_reverse_constraint_function(model::JuMP.Model, con_ref::JuMP.ConstraintRef)

Get the function to be used for reverse mode differentiation of a constraint.
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
