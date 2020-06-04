using OSQP


@testset "Testing forward on trivial QP" begin
    # using example on https://osqp.org/docs/examples/setup-and-solve.html
    Q = [4. 1.;1. 2.]
    q = [1.; 1.]
    G = [1. 1.; 1. 0.; 0. 1.; -1. -1.; -1. 0.; 0. -1.]
    h = [1.; 0.7; 0.7; -1.; 0.;0.];

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
                            0.
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    con_idx = []

    # add constraints
    for i in 1:6
        push!(con_idx, 
            MOI.add_constraint(
                model,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.),
                MOI.LessThan(h[i])
            )
        )
    end

    diff = DiffModel(model, con_idx)

    ẑ = diff.forward()
    
    @test maximum(abs.(ẑ - [.3; 0.7])) <= 1e-4
end
