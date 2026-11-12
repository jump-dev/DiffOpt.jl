# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::ReverseObjectiveFunction,
)
    if MOI.Bridges.is_objective_bridged(b)
        F = MOI.Bridges.Objective.function_type(
            MOI.Bridges.Objective.bridges(b),
        )
        return MOI.get(
            b,
            MOI.Bridges.ObjectiveFunctionAttribute{typeof(attr),F}(attr),
        )
    else
        return MOI.get(b.model, attr)
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
            MOI.Bridges.ObjectiveFunctionAttribute{typeof(attr),F}(attr),
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
        if F <: MOI.VariableIndex && S <: MOI.Parameter
            continue
        end
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
    # Needed for models which bridge `min f(x)` into `min t such that t >= f(x)`.
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
