function MOI.set(model::JuMP.Model, ::ForwardInObjective, func::JuMP.AbstractJuMPScalar)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(model, ForwardInObjective(), JuMP.moi_function(func))
end


# extend caching optimizer
function MOI.set(
    m::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    value,
)
    if m.state == MOI.Utilities.ATTACHED_OPTIMIZER
        optimizer_index = m.model_to_optimizer_map[index]
        optimizer_value = MOI.Utilities.map_indices(m.model_to_optimizer_map, value)
        # if m.mode == AUTOMATIC
        #     try
        #         MOI.set(m.optimizer, attr,
        #             optimizer_index, optimizer_value)
        #     catch err
        #         if err isa MOI.NotAllowedError
        #             reset_optimizer(m)
        #         else
        #             rethrow(err)
        #         end
        #     end
        # else
            MOI.set(m.optimizer, attr,
                optimizer_index, optimizer_value)
        # end
    else
        error("Cannot set AbstractDiffAttribute $(attr) is the state is different from ATTACHED_OPTIMIZER")
    end
    # return MOI.set(m.model_cache, attr, index, value)
end
function MOI.set(
    m::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
    value,
)
    if m.state == MOI.Utilities.ATTACHED_OPTIMIZER
        optimizer_index = m.model_to_optimizer_map[index]
        optimizer_index2 = m.model_to_optimizer_map[index2]
        optimizer_value = MOI.Utilities.map_indices(m.model_to_optimizer_map, value)
        # if m.mode == AUTOMATIC
        #     try
        #         MOI.set(m.optimizer, attr,
        #             optimizer_index, optimizer_index2, optimizer_value)
        #     catch err
        #         if err isa MOI.NotAllowedError
        #             MOI.Utilities.reset_optimizer(m)
        #         else
        #             rethrow(err)
        #         end
        #     end
        # else
            MOI.set(m.optimizer, attr,
                optimizer_index, optimizer_index2, optimizer_value)
        # end
    else
        error("Cannot set AbstractDiffAttribute $(attr) is the state is different from ATTACHED_OPTIMIZER")
    end
    # return MOI.set(m.model_cache, attr, index, index2, value)
end

function MOI.get(
    model::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
)
    if MOI.Utilities.state(model) == MOI.Utilities.NO_OPTIMIZER
        error(
            "Cannot query $(attr) from caching optimizer because no " *
            "optimizer is attached.",
        )
    end
    return MOI.Utilities.map_indices(
        model.optimizer_to_model_map,
        MOI.get(model.optimizer, attr, model.model_to_optimizer_map[index]),
    )
end
function MOI.get(
    model::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
)
    if MOI.Utilities.state(model) == MOI.Utilities.NO_OPTIMIZER
        error(
            "Cannot query $(attr) from caching optimizer because no " *
            "optimizer is attached.",
        )
    end
    return MOI.Utilities.map_indices(
        model.optimizer_to_model_map,
        MOI.get(model.optimizer, attr,
            model.model_to_optimizer_map[index],
            model.model_to_optimizer_map[index2]),
    )
end


function MOI.get(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    v::Union{JuMP.VariableRef,JuMP.ConstraintRef},
)
    JuMP.check_belongs_to_model(v, model)
    return MOI.get(JuMP.backend(model), attr, JuMP.index(v))
end
function MOI.get(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    i1::JuMP.VariableRef,
    i2::Union{JuMP.VariableRef,JuMP.ConstraintRef},
)
    JuMP.check_belongs_to_model(i1, model)
    JuMP.check_belongs_to_model(i2, model)
    return MOI.get(JuMP.backend(model), attr, JuMP.index(i1), JuMP.index(i2))
end

function MOI.set(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    v::Union{JuMP.VariableRef,JuMP.ConstraintRef},
    value,
)
    JuMP.check_belongs_to_model(v, model)
    return MOI.set(JuMP.backend(model), attr, JuMP.index(v), value)
end
function MOI.set(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    i1::JuMP.VariableRef,
    i2::Union{JuMP.VariableRef,JuMP.ConstraintRef},
    value,
)
    JuMP.check_belongs_to_model(i1, model)
    JuMP.check_belongs_to_model(i2, model)
    return MOI.set(JuMP.backend(model), attr, JuMP.index(i1), JuMP.index(i2), value)
end

# JuMP
backward(model::JuMP.Model) = backward(JuMP.backend(model))
forward(model::JuMP.Model) = forward(JuMP.backend(model))

# MOIU
backward(model::MOI.Utilities.CachingOptimizer) = backward(model.optimizer)
forward(model::MOI.Utilities.CachingOptimizer) = forward(model.optimizer)

# MOIB
backward(model::MOI.Bridges.AbstractBridgeOptimizer) = backward(model.model)
forward(model::MOI.Bridges.AbstractBridgeOptimizer) = forward(model.model)

# bridges
# TODO: bridging is non-trivial
# there might be transformations that we are ignoring
# we should at least check for bridge and block if they are used
function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
)
    # if is_bridged(b, index)
    #     value = call_in_context(
    #         b,
    #         index,
    #         bridge -> MOI.get(b, attr, bridge, _index(b, index)...),
    #     )
    # else
    value = MOI.get(b.model, attr, index)
    # end
    # return unbridged_function(b, value)
end
function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
)
    value = MOI.get(b.model, attr, index, index2)
end
function MOI.set(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    value,
)
    MOI.set(b.model, attr, index, value)
end
function MOI.set(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
    value,
)
    MOI.set(b.model, attr, index, index2, value)
end
