@testset "Test equality method" begin
    model = GLPK.Optimizer()
    x = MOI.add_variables(model, 2)

    con = MOI.add_constraint(
            model, 
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0; 1.0], x), 0.0),
            MOI.LessThan(1.0)
          )

    con_set = MOI.get(model, MOI.ConstraintSet(), con)

    @test is_equality(con_set) == false
end
