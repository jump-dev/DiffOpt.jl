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

    model = diff_optimizer(SCS.Optimizer)
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

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    @test x ≈ ones(3) atol=ATOL rtol=RTOL

    MOI.set(model, DiffOpt.BackwardIn{MOI.VariablePrimal}(), X[1], 1.0)

    DiffOpt.backward(model)

    db = MOI.get(model, DiffOpt.BackwardOut{DiffOpt.ConstraintConstant}(), c)

    @test db ≈ [-1.0]  atol=ATOL rtol=RTOL
    # TODO: unifor sign
end
