@deprecate backward(model) reverse_differentiate!(model) false
@deprecate forward(model) forward_differentiate!(model) false

@deprecate BackwardInVariablePrimal() ReverseVariablePrimal() false
@deprecate BackwardOutConstraint() ReverseConstraintPrimal() false
@deprecate BackwardOutObjective() ReverseObjective() false

@deprecate ForwardOutVariablePrimal() ForwardVariablePrimal() false
@deprecate ForwardInConstraint() ForwardConstraintFunction() false
@deprecate ForwardInObjective() ForwardObjective() false
