"""
    struct ObjectiveFunctionAttribute{A,F} <: MOI.AbstractModelAttribute
        attr::A
    end

Objective function attribute `attr` for the function type `F`.
The type `F` is used by a `MOI.Bridges.AbstractBridgeOptimizer` to keep track
of its position in a chain of objective bridges.
"""
struct ObjectiveFunctionAttribute{A,F} <: MOI.AbstractModelAttribute
    attr::A
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
    attr::ForwardObjectiveFunction,
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
    # Same as in `JuMP.set_start_values`
    # Needed for models which bridge `min f(x)` into `min t; t >= f(x)`.
    MOI.set(dest, MOI.Bridges.Objective.SlackBridgePrimalDualStart(), nothing)
    return
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
