@testset "Linear tests" begin
    MOIT.contlineartest(diff_optimizer(GLPK.Optimizer), MOIT.TestConfig(basis = true), [
        "partial_start",  # see below
        "linear12",       # see below
    ])

    MOIT.partial_start_test(
        diff_optimizer(Ipopt.Optimizer),
        MOIT.TestConfig(basis = true, optimal_status=MOI.LOCALLY_SOLVED, atol=ATOL, rtol=RTOL)
    )

    # This requires an infeasiblity certificate for a variable bound.
    MOIT.linear12test(
        diff_optimizer(GLPK.Optimizer),
        MOIT.TestConfig(infeas_certificates=false)
    )
end

@testset "Convex Quadratic tests" begin
    MOIT.qp1test(diff_optimizer(OSQP.Optimizer), MOIT.TestConfig(atol=1e-2, rtol=1e-2))
    MOIT.qp2test(diff_optimizer(OSQP.Optimizer), MOIT.TestConfig(atol=1e-2, rtol=1e-2))
    MOIT.qp3test(
        diff_optimizer(Ipopt.Optimizer), 
        MOIT.TestConfig(optimal_status=MOI.LOCALLY_SOLVED, atol=1e-3)
    )
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

@testset "ModelLike" begin
    for opt in [GLPK.Optimizer]
        MODEL = diff_optimizer(opt)
        @testset "default_objective_test" begin
            MOIT.default_objective_test(MODEL)
        end
        @testset "default_status_test" begin
            MOIT.default_status_test(MODEL)
        end
        @testset "nametest" begin
            MOIT.nametest(MODEL)
        end
        @testset "validtest" begin
            MOIT.validtest(MODEL)
        end
        @testset "emptytest" begin
            MOIT.emptytest(MODEL)
        end
        @testset "orderedindicestest" begin
            MOIT.orderedindicestest(MODEL)
        end
        @testset "copytest" begin
            # Requires VectorOfVariables
            # MOIT.copytest(MODEL, MOIU.CachingOptimizer(
            #     diff_optimizer(GLPK.Optimizer),
            #     GLPK.Optimizer()
            # ))
        end
    end
end


@testset "Unit" begin
    MOIT.unittest(diff_optimizer(GLPK.Optimizer), MOIT.TestConfig(), [
        "number_threads", # might not work on all solvers
            
        # not testing integer constraints
        "solve_zero_one_with_bounds_1",  
        "solve_zero_one_with_bounds_2",
        "solve_zero_one_with_bounds_3",  
        "solve_integer_edge_cases",  
            
        "delete_soc_variables", 
        "solve_qcp_edge_cases",  # currently only affine or conic constraints
        "solve_objbound_edge_cases",
        "solve_qp_edge_cases",  # No quadratics
        "update_dimension_nonnegative_variables", # TODO: fix this

        # see below
        "solve_duplicate_terms_vector_affine"
    ])

    MOIT.solve_duplicate_terms_obj(diff_optimizer(SCS.Optimizer), MOIT.TestConfig())
end

@testset "basic_constraint_tests" begin
    # it contains SOCP constraints
    MOIT.basic_constraint_tests(diff_optimizer(SCS.Optimizer), MOIT.TestConfig())
end

@testset "contconic.jl tests" begin
    MODEL = diff_optimizer(SCS.Optimizer)

    # linear tests
    for (name, test) in MOI.Test.lintests
        test(MODEL, MOIT.TestConfig())
    end

    CONFIG_LOW_TOL = MOIT.TestConfig(atol = 1e-3, rtol = 1e-2, duals = false, infeas_certificates = false)
    # SOCP tests
    for (name, test) in MOI.Test.soctests
        test(MODEL, CONFIG_LOW_TOL)
    end
end
