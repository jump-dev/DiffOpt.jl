"""
Generates a non-trivial random MOI linear program by adding variables
and constraints to MOI-compatible Optimizer `optimizer`

minimize `c' * x`
subject to `Ax <= b, x >= 0`
where `x in R^{n}, A in R^{m*n}, b in R^{m}, c in R^{n}`

Note: Mutates the `optimizer` object
"""
function generate_lp(optimizer, n, m)
    s = rand(m)
    s = 2*s .- 1
    λ = max.(-s, 0)
    s = max.(s, 0)
    x̂ = rand(n)
    A = rand(m, n)
    b = A * x̂ .+ s
    c = -A' * λ

    x = MOI.add_variables(optimizer, n)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    for i in 1:m
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.), MOI.LessThan(b[i]),
        )
    end

    for i in 1:n
        MOI.add_constraint(optimizer, x[i], MOI.GreaterThan(0.0))
    end
end


"""
Generates a non-trivial random MOI convex quadratic program 
by adding variables and constraints to MOI compatible Optimizer `optimizer`

minimize `0.5 * x' * Q * x  + q' * x`
subject to `Gx <= h, Ax == b`
where `x in R^{n}, Q in R^{n*n}, q in R^{n}, G in R^{m,n}, h in R^{m}, A in R^{p*n}, b in R^{p}`

Note: (1) Mutates the `optimizer` object
      (2) Matrix `Q` is Positive Semidefinite
"""
function generate_qp(optimizer,n,m,p)
    x̂ = rand(n)
    Q = rand(n, n)
    Q = Q' * Q # ensure PSD
    q = rand(n)
    G = rand(m, n)
    h = G * x̂ + rand(m)
    A = rand(p, n)
    b = A * x̂

    x = MOI.add_variables(optimizer, n)

    # define objective
    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:n
        for j in i:n # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quadratic_terms, MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j]))
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(q, x), quadratic_terms, 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    for i in 1:m
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.),MOI.LessThan(h[i])
        )
    end

    for i in 1:p
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.),MOI.EqualTo(b[i])
        )
    end
end


@testset "Creating an LP" begin
    model = GLPK.Optimizer()
    x̂ = generate_lp(model,10,5)

    MOI.set(model, MOI.Silent(), true)    
    MOI.optimize!(model)

    @test MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
end


@testset "Creating a convex QP" begin
    model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
    x̂ = generate_qp(model,10,5,5)
    
    MOI.set(model, MOI.Silent(), true)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
end
