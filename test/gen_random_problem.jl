@testset "Creating an LP" begin
    model = GLPK.Optimizer()
    x̂ = generate_lp(model,10,5)
    
    MOI.optimize!(model)

    @test MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
end


@testset "Creating a convex QP" begin
    model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
    x̂ = generate_qp(model,10,5,5)
    
    MOI.optimize!(model)

    @test MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
end
