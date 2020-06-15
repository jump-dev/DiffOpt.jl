@testset "Testing forward on trivial QP" begin
    # using example on https://osqp.org/docs/examples/setup-and-solve.html
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0; 1.0]
    G = [1.0 1.0; 1.0 0.0; 0.0 1.0; -1.0 -1.0; -1.0 0.0; 0.0 -1.0]
    h = [1.0; 0.7; 0.7; -1.0; 0.0; 0.0];

    model = MOI.instantiate(Ipopt.Optimizer, with_bridge_type=Float64)
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

    z, λ, ν = diff.forward()
    
    @test z ≈ [0.3; 0.7] atol=ATOL rtol=RTOL
end



@testset "Differentiating trivial QP 1" begin
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

    z, λ, ν = diff.forward()
    
    @test z ≈ [-0.25; -0.75] atol=ATOL rtol=RTOL

    grad_wrt_h = diff.backward(["h"], [1.0 1.0])[1]

    @test grad_wrt_h ≈ [1.0] atol=ATOL rtol=RTOL
end


@testset "Differentiating a non-convex QP" begin
    Q = [0.0 0.0; 1.0 2.0]
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
                MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j]),
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0,
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraint
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1]),
    )

    diff = diff_model(model)

    @test_throws ErrorException diff.forward() # should break
end


# TODO: differentiate this and compare with qpth
@testset "Differentiating QP with inequality and equality constraints" begin
    # refered from: https://www.mathworks.com/help/optim/ug/quadprog.html#d120e113424
    Q = [1.0 -1.0 1.0; 
        -1.0  2.0 -2.0;
        1.0 -2.0 4.0]
    q = [2.0; -3.0; 1.0]
    G = [0.0 0.0 1.0;
         0.0 1.0 0.0;
         1.0 0.0 0.0;
         0.0 0.0 -1.0;
         0.0 -1.0 0.0;
         -1.0 0.0 0.0;]
    h = [1.0; 1.0; 1.0; 0.0; 0.0; 0.0;]
    A = [1.0 1.0 1.0;]
    b = [0.5;]

    model = MOI.instantiate(Ipopt.Optimizer, with_bridge_type=Float64)
    x = MOI.add_variables(model, 3)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:3
        for j in i:3 # indexes (i,j), (j,i) will be mirrored. specify only one kind
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
    for i in 1:6
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i, :], x), 0.0),
            MOI.LessThan(h[i])
        )
    end

    for i in 1:1
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.0),
            MOI.EqualTo(b[i])
        )
    end

    diff = diff_model(model)

    z, λ, ν  = diff.forward()
    
    @test z ≈ [0.0; 0.5; 0.0] atol=ATOL rtol=RTOL

    grads = diff.backward(["Q","q","G","h","A","b"], [1.0 1.0 1.0])
end



# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_1_py.ipynb
@testset "Differentiating MOI examples 1" begin
    # homogeneous quadratic objective
    # Min x^2 + xy + y^2 + yz + z^2
    # st  x + 2y + 3z >= 4 (c1)
    #     x +  y      >= 1 (c2)
    #     x, y, z \in R

    model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
    v = MOI.add_variables(model, 3)
    @test MOI.get(model, MOI.NumberOfVariables()) == 3

    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([-1.0, -2.0, -3.0], v),
            0.0),
        MOI.LessThan(-4.0)
    )
    c2 = MOI.add_constraint(
        model, 
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([-1.0, -1.0, 0.0], v),
            0.0),
        MOI.LessThan(-1.0)
    )

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE

    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm{Float64}[], 
        MOI.ScalarQuadraticTerm.(
            [2.0, 1.0, 2.0, 1.0, 2.0],
            v[[1, 1, 2, 2, 3]],
            v[[1, 2, 2, 3, 3]]
        ),
        0.0
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj)

    diff = diff_model(model)

    z, λ, ν = diff.forward()

    @test z ≈ [4/7, 3/7, 6/7] atol=ATOL rtol=RTOL

    # obtain gradients
    grads = diff.backward(["Q","q","G","h"], [1.0 1.0 1.0])

    dl_dQ = grads[1]
    dl_dq = grads[2]
    dl_dG = grads[3]
    dl_dh = grads[4]

    @test dl_dQ ≈ [-0.12244895  0.01530609 -0.11224488;
                    0.01530609  0.09183674  0.07653058;
                   -0.11224488  0.07653058 -0.06122449]  atol=ATOL rtol=RTOL

    @test dl_dq ≈ [-0.2142857;  0.21428567; -0.07142857] atol=ATOL rtol=RTOL

    @test dl_dG ≈ [0.05102035   0.30612245  0.255102;
                   0.06122443   0.36734694  0.3061224] atol=ATOL rtol=RTOL

    @test dl_dh ≈ [-0.35714284; -0.4285714] atol=ATOL rtol=RTOL
end



# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_2_py.ipynb
@testset "Differentiating MOI examples 2 - non trivial backward pass vector" begin
    # non-homogeneous quadratic objective
    #    minimize 2 x^2 + y^2 + xy + x + y
    #       s.t.  x, y >= 0
    #             x + y = 1

    model = MOI.instantiate(Ipopt.Optimizer, with_bridge_type=Float64)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)

    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], [x,y]), 0.0),
        MOI.EqualTo(1.0)
    )

    vc1 = MOI.add_constraint(
        model, 
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0,0.0], [x,y]), 0.0), 
        MOI.LessThan(0.0)
    )
    @test vc1.value == x.value

    vc2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0,-1.0], [x,y]), 0.0), 
        MOI.LessThan(0.0)
    )
    @test vc2.value == y.value


    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x, y]),
        MOI.ScalarQuadraticTerm.([4.0, 2.0, 1.0], [x, y, x], [x, y, y]),
        0.0
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj)

    diff = diff_model(model)

    z, λ, ν = diff.forward()

    @test z ≈ [0.25, 0.75] atol=ATOL rtol=RTOL
    @test λ ≈ [0.0, 0.0]   atol=ATOL rtol=RTOL
    @test ν ≈ [11/4]       atol=ATOL rtol=RTOL

    # obtain gradients
    dl_dz = [1.3 0.5]   # choosing a non trivial backward pass vector
    grads = diff.backward(["Q", "q", "G", "h", "A", "b"], dl_dz)

    dl_dQ = grads[1]
    dl_dq = grads[2]
    dl_dG = grads[3]
    dl_dh = grads[4]
    dl_dA = grads[5]
    dl_db = grads[6]

    @test dl_dQ ≈ [-0.05   -0.05;
                   -0.05    0.15]  atol=ATOL rtol=RTOL

    @test dl_dq ≈ [-0.2; 0.2] atol=ATOL rtol=RTOL

    @test dl_dG ≈ [1e-8  1e-8; 1e-8 1e-8] atol=ATOL rtol=RTOL

    @test dl_dh ≈ [1e-8; 1e-8] atol=ATOL rtol=RTOL

    @test dl_dA ≈ [0.375 -1.075] atol=ATOL rtol=RTOL

    @test dl_db ≈ [0.7] atol=ATOL rtol=RTOL
end