using Random

const MOIT = MathOptInterface.Test
const CONFIG = MOIT.TestConfig()

@testset "Linear tests" begin
    MOIT.contlineartest(diff_optimizer(GLPK.Optimizer), MOIT.TestConfig(basis = true), [
        # This requires an infeasiblity certificate for a variable bound.
        "linear12",
    ])
    MOIT.linear12test(diff_optimizer(Ipopt.Optimizer), MOIT.TestConfig(infeas_certificates=false))
end


@testset "FEASIBILITY_SENSE zeros objective" begin
    model = diff_optimizer(GLPK.Optimizer)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.SingleVariable}(), MOI.SingleVariable(x))

    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0
    
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) >= 1.0
end

# TODO: Has ZeroOne constraint. can differentiate such?
# @testset "Issue #140 https://github.com/jump-dev/Gurobi.jl" begin
#     m = diff_optimizer(GLPK.Optimizer)
#     N = 100
#     x = MOI.add_variables(m, N)
#     for xi in x
#         MOI.add_constraint(m, MOI.SingleVariable(xi), MOI.ZeroOne())
#         MOI.set(m, MOI.VariablePrimalStart(), xi, 0.0)
#     end
#     # Given a collection of items with individual weights and values,
#     # maximize the total value carried subject to the constraint that
#     # the total weight carried is less than 10.
#     Random.seed!(1)
#     item_weights = rand(N)
#     item_values = rand(N)
#     MOI.add_constraint(m,
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(item_weights, x), 0.0),
#         MOI.LessThan(10.0))
#     MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(.-item_values, x), 0.0))
#     MOI.optimize!(m)

#     @test MOI.get(m, MOI.TerminationStatus()) == MOI.SOLUTION_LIMIT
#     # We should have a primal feasible solution:
#     @test MOI.get(m, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
#     # But we have no dual status:
#     @test MOI.get(m, MOI.DualStatus()) == MOI.NO_SOLUTION
# end
