function MOI.set(
    model::MOI.ModelLike,
    attr::ForwardInConstraint,
    bridge::MOI.Bridges.Constraint.VectorizeBridge{T},
    value,
) where {T}
    MOI.set(model, attr, bridge.vector_constraint, MOI.Utilities.operate(vcat, T, value))
end
function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.BackwardOutConstraint,
    bridge::MOI.Bridges.Constraint.AbstractFunctionConversionBridge,
)
    return MOI.get(model, attr, bridge.constraint)
end
function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.ForwardInConstraint,
    bridge::MOI.Bridges.Constraint.SetMapBridge,
)
    func = MOI.get(model, attr, bridge.constraint)
    return MOI.Bridges.map_function(typeof(bridge), func)
end
function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.BackwardOutConstraint,
    bridge::MOI.Bridges.Constraint.SetMapBridge,
)
    func = MOI.get(model, attr, bridge.constraint)
    return MOI.Bridges.adjoint_map_function(typeof(bridge), func)
end
