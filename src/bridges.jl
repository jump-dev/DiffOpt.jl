function MOI.set(model::MOI.ModelLike, attr::ForwardInConstraint, bridge::MOI.Bridges.Constraint.VectorizeBridge{T}, value) where {T}
    MOI.set(model, attr, bridge.vector_constraint, MOI.Utilities.operate(vcat, T, value))
end
