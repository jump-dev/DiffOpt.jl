"""
Generates a non-trivial random MOI linear program by adding variables
and constraints to MOI compatible Optimizer `optimizer`

minimize c'x
subject to Ax <= b, x >= 0 
where x in R^{n}, A in R^{m*n}, b in R^{m}, c in R^{n}

Note: Mutates the `optimizer` object
"""
function generate_lp(optimizer,n,m)
    s = rand(m)
    s = 2*s.-1
    λ = max.(-s, 0)
    s = max.(s, 0)
    x̂ = rand(n)
    A = rand(m, n)
    b = A*x̂ .+ s
    c = -A'*λ;

    x = MOI.add_variables(optimizer, n)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    for i in 1:m
        MOI.add_constraint(optimizer,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.),MOI.LessThan(b[i]))
    end

    for i in 1:n
        MOI.add_constraint(optimizer,MOI.SingleVariable(x[i]),MOI.GreaterThan(0.))
    end
end