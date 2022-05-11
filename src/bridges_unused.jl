function MOI.set(
    model::MOI.ModelLike,
    attr::BackwardInVariablePrimal,
    bridge::MOI.Bridges.Variable.VectorizeBridge,
    value,
)
    MOI.set(model, attr, bridge.variable, value)
end
