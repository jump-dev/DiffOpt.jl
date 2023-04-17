struct ObjectiveFunctionAttribute{A,F}
    attr::A
end

"""
    struct ObjectiveDualStart <: MOI.AbstractModelAttribute end

If the objective function had a dual, it would be `-1` for the Lagrangian
function to be the same.
When the `MOI.Bridges.Objective.SlackBridge` is used, it creates a constraint.
The dual of this constraint is therefore `-1` as well.
When setting this attribute, it allows to set the constraint dual of this
constraint.
"""
struct ObjectiveDualStart <: MOI.AbstractModelAttribute end
# Defining it for `MOI.set` leads to ambiguity
function MOI.throw_set_error_fallback(
    ::MOI.ModelLike,
    ::ObjectiveDualStart,
    value,
)
    return nothing
end

"""
    struct ObjectiveSlackGapPrimalStart <: MOI.AbstractModelAttribute end

If the objective function had a dual, it would be `-1` for the Lagrangian
function to be the same.
When the `MOI.Bridges.Objective.SlackBridge` is used, it creates a constraint.
The dual of this constraint is therefore `-1` as well.
When setting this attribute, it allows to set the constraint dual of this
constraint.
"""
struct ObjectiveSlackGapPrimalStart <: MOI.AbstractModelAttribute end
function MOI.throw_set_error_fallback(
    ::MOI.ModelLike,
    ::ObjectiveSlackGapPrimalStart,
    value,
)
    return nothing
end

function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::ObjectiveFunctionAttribute{A,F},
) where {A,F}
    obj_attr = MOI.ObjectiveFunction{F}()
    if MOI.Bridges.is_bridged(b, obj_attr)
        return MOI.get(
            MOI.Bridges.recursive_model(b),
            attr,
            MOI.Bridges.bridge(b, obj_attr),
        )
    else
        return MOI.get(b.model, attr.attr)
    end
end

function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::ReverseObjectiveFunction,
)
    if MOI.Bridges.is_objective_bridged(b)
        F = MOI.Bridges.Objective.function_type(
            MOI.Bridges.Objective.bridges(b),
        )
        return MOI.get(b, ObjectiveFunctionAttribute{typeof(attr),F}(attr))
    else
        return MOI.get(b.model, attr)
    end
end

function MOI.set(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::ObjectiveFunctionAttribute{A,F},
    value,
) where {A,F}
    obj_attr = MOI.ObjectiveFunction{F}()
    if MOI.Bridges.is_bridged(b, obj_attr)
        return MOI.set(
            MOI.Bridges.recursive_model(b),
            attr,
            MOI.Bridges.bridge(b, obj_attr),
            value,
        )
    else
        return MOI.set(b.model, attr.attr, value)
    end
end

function MOI.set(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::Union{
        ObjectiveDualStart,
        ObjectiveSlackGapPrimalStart,
        ForwardObjectiveFunction,
    },
    value,
)
    if MOI.Bridges.is_objective_bridged(b)
        F = MOI.Bridges.Objective.function_type(
            MOI.Bridges.Objective.bridges(b),
        )
        return MOI.set(
            b,
            ObjectiveFunctionAttribute{typeof(attr),F}(attr),
            value,
        )
    else
        return MOI.set(b.model, attr, value)
    end
end

function MOI.set(
    model::MOI.ModelLike,
    ::ObjectiveFunctionAttribute{ObjectiveDualStart},
    b::MOI.Bridges.Objective.SlackBridge,
    value,
)
    return MOI.set(model, MOI.ConstraintDualStart(), b.constraint, value)
end

function MOI.set(
    model::MOI.ModelLike,
    ::ObjectiveFunctionAttribute{ObjectiveSlackGapPrimalStart},
    b::MOI.Bridges.Objective.SlackBridge{T},
    value,
) where {T}
    # `f(x) - slack = value` so `slack = f(x) - value`
    fun = MOI.get(model, MOI.ConstraintFunction(), b.constraint)
    set = MOI.get(model, MOI.ConstraintSet(), b.constraint)
    MOI.Utilities.operate!(-, T, fun, MOI.constant(set))
    # `fun = f - slack` so we remove the term `-slack` to get `f`
    f = MOI.Utilities.remove_variable(fun, b.slack)
    f_val = MOI.Utilities.eval_variables(f) do v
        return MOI.get(model, MOI.VariablePrimalStart(), v)
    end
    MOI.set(model, MOI.VariablePrimalStart(), b.slack, f_val - value)
    return MOI.set(model, MOI.ConstraintPrimalStart(), b.constraint, value)
end

function _copy_dual(dest::MOI.ModelLike, src::MOI.ModelLike, index_map)
    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    MOI.set(
        dest,
        MOI.VariablePrimalStart(),
        getindex.(Ref(index_map), vis_src),
        MOI.get(src, MOI.VariablePrimal(), vis_src),
    )
    for (F, S) in MOI.get(dest, MOI.ListOfConstraintTypesPresent())
        _copy_constraint_start(
            dest,
            src,
            index_map.con_map[F, S],
            MOI.ConstraintPrimalStart(),
            MOI.ConstraintPrimal(),
        )
        _copy_constraint_start(
            dest,
            src,
            index_map.con_map[F, S],
            MOI.ConstraintDualStart(),
            MOI.ConstraintDual(),
        )
    end
    MOI.set(dest, ObjectiveDualStart(), -1.0)
    return MOI.set(dest, ObjectiveSlackGapPrimalStart(), 0.0)
end

function _copy_constraint_start(
    dest,
    src,
    index_map::MOIU.DoubleDicts.IndexDoubleDictInner{F,S},
    dest_attr,
    src_attr,
) where {F,S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        value = MOI.get(src, src_attr, ci)
        MOI.set(dest, dest_attr, index_map[ci], value)
    end
end
