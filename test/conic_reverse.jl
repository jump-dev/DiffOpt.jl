import SCS
# min X[1,1] + X[2,2]    max y
#     X[2,1] = 1         [0   y/2     [ 1  0
#                         y/2 0    <=   0  1]
#     X >= 0              y free
# Optimal solution:
#
#     ⎛ 1   1 ⎞
# X = ⎜       ⎟           y = 2
#     ⎝ 1   1 ⎠
@testset "Differentiating simple PSD back" begin

    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    X = MOI.add_variables(model, 3)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{Float64}(vov),
        MOI.PositiveSemidefiniteConeTriangle(2)
    )

    c  = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, X[2]))],
            [-1.0]
        ),
        MOI.Zeros(1)
    )

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[1], X[end]]), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    x = MOI.get(model, MOI.VariablePrimal(), X)

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())])
    conic_form = DiffOpt.GeometricConicForm{Float64}()
    cones = conic_form.constraints.sets
    DiffOpt.set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)

    @test x ≈ ones(3) atol=ATOL rtol=RTOL

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), X[1], 1.0)

    DiffOpt.reverse_differentiate!(model)

    db = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))

    @test db ≈ [-1.0]  atol=ATOL rtol=RTOL
end
