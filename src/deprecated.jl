@deprecate backward(model) reverse_differentiate!(model) false
@deprecate forward(model) forward_differentiate!(model) false

@deprecate BackwardInVariablePrimal() ReverseVariablePrimal() false
@deprecate BackwardOutConstraint() ReverseConstraintFunction() false
@deprecate BackwardOutObjective() ReverseObjectiveFunction() false

@deprecate ForwardOutVariablePrimal() ForwardVariablePrimal() false
@deprecate ForwardInConstraint() ForwardConstraintFunction() false
@deprecate ForwardInObjective() ForwardObjective() false
