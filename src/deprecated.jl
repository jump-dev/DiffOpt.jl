@deprecate backward(model) reverse_differentiate!(model) false
@deprecate forward(model) forward_differentiate!(model) false

@deprecate BackwardInVariablePrimal() ReverseVariablePrimal() false
@deprecate BackwardOutConstraint() ReverseConstraintFunction() false
@deprecate BackwardOutObjective() ReverseObjectiveFunction() false

@deprecate ForwardOutVariablePrimal() ForwardVariablePrimal() false
@deprecate ForwardInConstraint() ForwardConstraintFunction() false
@deprecate ForwardInObjective() ForwardObjectiveFunction() false

@deprecate QPForwBackCache(args...) QuadraticForwardReverseCache(args...) false
@deprecate ConicBackCache(args...) ConicReverseCache(args...) false

@deprecate QPDiff(args...) QuadraticDiffProblem(args...) false
@deprecate ConicDiff(args...) ConicDiffProblem(args...) false
@deprecate QPForm(args...) QuadraticProblemForm(args...) false
