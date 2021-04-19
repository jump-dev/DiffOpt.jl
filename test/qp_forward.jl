@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3

    model = diff_optimizer(Clp.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    v = MOI.add_variables(model, 1)

    # define objective
    objective_function = MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([1.0], v), 0.0)
    MOI.set(model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0], v), 0.),
        MOI.LessThan(0.0)
    )
    c2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0], v), 0.),
        MOI.LessThan(-3.0)
    )
    cg = [c1, c2]

    MOI.optimize!(model)

    MOI.set(model, DiffOpt.ForwardIn{DiffOpt.ConstraintConstant}(), c2, 1.0)

    DiffOpt.forward(model)

    dx = MOI.get(model, DiffOpt.ForwardOut{MOI.VariablePrimal}(), v[])
    @test dx ≈ 1.0  atol=ATOL rtol=RTOL
end
