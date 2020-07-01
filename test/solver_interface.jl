@testset "Linear tests" begin
    MOIT.contlineartest(diff_optimizer(GLPK.Optimizer), MOIT.TestConfig(basis = true), [
        "partial_start",  # see below
        "linear1",        # see below
        "linear12",       # see below
        "linear15",       # refer https://github.com/AKS1996/DiffOpt.jl/issues/21
        "linear5",        # refer https://github.com/AKS1996/DiffOpt.jl/issues/20
        "linear7",        # vector issues
        "linear10"        # refer https://github.com/AKS1996/DiffOpt.jl/issues/22
    ])

    MOIT.partial_start_test(
        diff_optimizer(Ipopt.Optimizer),
        MOIT.TestConfig(basis = true, optimal_status=MOI.LOCALLY_SOLVED, atol=ATOL, rtol=RTOL)
    )

    MOIT.linear1test(diff_optimizer(GLPK.Optimizer), MOIT.TestConfig(basis = true, modify_lhs=false))

    # This requires an infeasiblity certificate for a variable bound.
    MOIT.linear12test(diff_optimizer(GLPK.Optimizer), MOIT.TestConfig(infeas_certificates=false))
end


@testset "FEASIBILITY_SENSE zeros objective" begin
    model = diff_optimizer(GLPK.Optimizer)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.SingleVariable}(), MOI.SingleVariable(x))

    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=ATOL rtol=RTOL
    
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) >= 1.0
end

@testset "Removing variables from model" begin
    model = diff_optimizer(GLPK.Optimizer)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)

    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.LessThan(0.))
    MOI.add_constraint(model, MOI.SingleVariable(y), MOI.LessThan(0.))
    MOI.add_constraint(model, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1., 1.], [x, y]), 0.), MOI.LessThan(0.))

    MOI.delete(model, x)

    @test size(model.con_idx)[1] == 1

    func = MOI.get(model, MOI.ConstraintFunction(), model.con_idx[1])
    @test func.variable == y
end