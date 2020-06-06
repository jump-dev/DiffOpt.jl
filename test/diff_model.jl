using OSQP


@testset "Testing forward on trivial QP" begin
    # using example on https://osqp.org/docs/examples/setup-and-solve.html
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0; 1.0]
    G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
    h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0];

    model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
    x = MOI.add_variables(model, 2)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms, 
                MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j])
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for i in 1:6
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
            MOI.LessThan(h[i])
        )
    end

    diff = diff_model(model)

    ẑ = diff.forward()
    
    @test maximum(abs.(ẑ - [0.3; 0.7])) <= 1e-4
end



@testset "Testing backward on trivial QP 1" begin
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0; 1.0]
    G = [1.0 1.0;]
    h = [-1.0;]

    model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
    x = MOI.add_variables(model, 2)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms, 
                MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j])
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraint
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1])
    )

    diff = diff_model(model)

    ẑ = diff.forward()
    
    @test maximum(abs.(ẑ - [-0.25; -0.75])) <= 1e-4

    grad_wrt_h = diff.backward(["h"])[1]

    @test maximum(abs.(grad_wrt_h - [0.25; 0.75; -1.75])) <= 1e-4
end
