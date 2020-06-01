using GLPK

@testset "Creating an LP" begin
    model = GLPK.Optimizer()
    x̂ = generate_lp(model,10,5)
    
    MOI.optimize!(model)

    @test MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
end
