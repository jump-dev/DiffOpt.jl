import HiGHS

@testset "FEASIBILITY_SENSE zeros objective" begin
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)

    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=ATOL rtol=RTOL

    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) == 0.0
end

@testset "Forward or reverse without optimizing throws" begin
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    # do not optimize, just try to differentiate
    @test_throws ErrorException DiffOpt.forward_differentiate!(model)
    @test_throws ErrorException DiffOpt.reverse_differentiate!(model)
    # impossible constraint
    MOI.add_constraint(model, x, MOI.LessThan(0.5))
    MOI.optimize!(model)
    @test_throws ErrorException DiffOpt.forward_differentiate!(model)
    @test_throws ErrorException DiffOpt.reverse_differentiate!(model)
end
