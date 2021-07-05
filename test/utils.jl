using LinearAlgebra, Test

const ATOL = 1e-4
const RTOL = 1e-4

using DiffOpt

import MathOptInterface
const MOI = MathOptInterface

macro _test(computed, expected::Symbol)
    exp = esc(expected)
    com = esc(computed)
    return :(
        if $exp === nothing
            $exp = $com
        else
            @test $com ≈ $exp atol = ATOL rtol = RTOL
        end
    )
end

"""
    qp_test(solver; kws...)

Check our results for QPs using the notations of [AK17].

[AK17] Amos, Brandon, and J. Zico Kolter. "Optnet: Differentiable optimization as a layer in neural networks." International Conference on Machine Learning. PMLR, 2017. https://arxiv.org/pdf/1703.00443.pdf
"""
function qp_test(
    solver,
    lt::Bool,
    set_zero::Bool,
    canonicalize::Bool;
    dzb = nothing,
    n = length(dzb),
    q = nothing,
    dqf = zeros(n),
    dqb = nothing,
    Q = zeros(n, n),
    dQf = zeros(n, n),
    dQb = nothing,
    h = zeros(0),
    nle = length(h),
    dhf = zeros(nle),
    dhb = nothing,
    G = zeros(0, n),
    dGf = zeros(nle, n),
    dGb = nothing,
    b = zeros(0),
    neq = length(b),
    dbf = zeros(neq),
    dbb = nothing,
    A = zeros(0, n),
    dAf = zeros(neq, n),
    dAb = nothing,
    z = nothing,
    dzf = nothing,
    ∇zf = nothing,
    ∇zb = nothing,
    λ = nothing,
    dλf = zeros(nle),
    dλb = zeros(nle),
    ∇λf = nothing,
    ∇λb = nothing,
    ν = nothing,
    dνf = nothing,
    dνb = zeros(neq),
    ∇νf = nothing,
    ∇νb = nothing,
)
    n = length(q)
    @assert n == LinearAlgebra.checksquare(Q)
    @assert n == size(A, 2)
    @assert n == size(G, 2)
    model = diff_optimizer(solver)
    MOI.set(model, MOI.Silent(), true)

    v = MOI.add_variables(model, n)
    fv = MOI.SingleVariable.(v)

    _sign(x, a) = a == lt ? -x : x

    if lt
        cle = MOI.add_constraint.(model, G * fv, MOI.LessThan.(h))
    else
        cle = MOI.add_constraint.(model, -G * fv, MOI.GreaterThan.(-h))
    end
    ceq = MOI.add_constraint.(model, A * fv, MOI.EqualTo.(b))

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    if iszero(Q)
        obj = q ⋅ fv
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    else
        obj = fv' * (Q / 2) * fv + q ⋅ fv
        MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    end

    MOI.optimize!(model)

    @_test(MOI.get(model, MOI.VariablePrimal(), v), z)

    # The sign as reversed as [AK17]
    # use a different convention for the dual
    @_test(convert(Vector{Float64}, -MOI.get.(model, MOI.ConstraintDual(), ceq)), ν)
    @_test(convert(Vector{Float64}, _sign(MOI.get.(model, MOI.ConstraintDual(), cle), true)), λ)

    #dobjb = fv' * (dQb / 2.0) * fv + dqb' * fv
    # TODO, it should .-
    #dleb = dGb * fv .+ dhb
    #deqb = dAb * fv .+ dbb
    @assert dzb !== nothing
    @testset "Backward pass" begin
        MOI.set.(model, DiffOpt.BackwardInVariablePrimal(), v, dzb)

        DiffOpt.backward(model)

        dobjb = MOI.get(model, DiffOpt.BackwardOutObjective())
        spb = DiffOpt.sparse_array_representation(
            DiffOpt.standard_form(dobjb),
            n,
            dobjb.index_map,
        )
        @_test(spb.quadratic_terms, dQb)
        @_test(spb.affine_terms, dqb)

        funcs = MOI.get.(model, DiffOpt.BackwardOutConstraint(), cle)
        @_test(convert(Vector{Float64}, MOI.constant.(funcs)), dhb)
        @_test(Float64[JuMP.coefficient(funcs[i], vi) for i in eachindex(cle), vi in v], dGb)
        funcs = MOI.get.(model, DiffOpt.BackwardOutConstraint(), ceq)
        @_test(convert(Vector{Float64}, MOI.constant.(funcs)), dbb)
        @_test(Float64[JuMP.coefficient(funcs[i], vi) for i in eachindex(ceq), vi in v], dAb)
    end

    # Test against [AK17, eq. (8)]
    @_test(dqb, ∇zb)
    @_test((∇zb * z' + z * ∇zb') / 2, dQb)
    @_test(-dbb, ∇νb)
    @_test(∇νb * z' + ν * ∇zb', dAb)
    if all(i -> abs(λ[i]) > ATOL, 1:nle)
        @_test(-dhb ./ λ, ∇λb)
    end
    if ∇λb !== nothing
        @_test(Diagonal(λ) * (∇λb * z' + λ * ∇zb'), dGb)
    end

    # Test against [AK17, eq. (7)]
    if ∇λb !== nothing
        @_test(-(Q * ∇zb + G' * (λ .* ∇λb) + A' * ∇νb), dzb)
    end
    if ∇λb !== nothing
        @_test(-(G * ∇zb + (G * z - h) .* ∇λb), dλb)
    end
    @_test(-A * ∇zb, dνb)

    dobjf = fv' * (dQf / 2.0) * fv + dqf' * fv
    # TODO, it should .-
    dlef = dGf * fv .+ dhf
    deqf = dAf * fv .+ dbf

    @testset "Forward pass" begin
        MOI.set(model, DiffOpt.ForwardInObjective(), dobjf)
        for (j, jc) in enumerate(cle)
            func = dlef[j]
            canonicalize && MOI.Utilities.canonicalize!(func)
            if set_zero || !MOI.iszero(dlef[j])
                MOI.set(model, DiffOpt.ForwardInConstraint(), jc, func)
            end
        end
        for (j, jc) in enumerate(ceq)
            func = deqf[j]
            canonicalize && MOI.Utilities.canonicalize!(func)
            if set_zero || !MOI.iszero(deqf[j])
                MOI.set(model, DiffOpt.ForwardInConstraint(), jc, func)
            end
        end

        DiffOpt.forward(model)

        @_test(MOI.get.(model, DiffOpt.ForwardOutVariablePrimal(), v), dzf)
    end

    # Test against [AK17, eq. (6)]
    @_test(dQf * z + dqf + dAf' * ν + dGf' * λ, ∇zf)
    if dλf !== nothing && dνf !== nothing
        @test Q * dzf + G' * dλf + A' * dνf ≈ ∇zf atol = ATOL rtol = RTOL
    end
    @_test(λ .* (dGf * z - dhf), ∇λf)
    @test (G * z - h) .* dλf + λ .* (G * dzf) ≈ -∇λf atol = ATOL rtol = RTOL
    @_test(dAf * z - dbf, ∇νf)
    @test A * dzf ≈ -∇νf atol = ATOL rtol = RTOL

    # As a kind of integration test, we check that the scalar product is the same whether it is don at the level of
    # 1) (dz, dλ, dν) (dλb and dνb are zero so we ignore their product (appropriate since we have not yet
    #    implemented the getter for dνf))
    dprod = dzf ⋅ dzb # ignored as it is zero : + dλf ⋅ dλb + dνf ⋅ dνb
    # 2) (∇z, ∇λ, ∇ν) which are the LHS of (6) and (7) (which are differentiation
    #    of the gradient of the laplacian with respect to z, λ and ∇ν hence the variable names)
    if ∇λb !== nothing
        ∇prod = ∇zf ⋅ ∇zb + ∇λf ⋅ ∇λb + ∇νf ⋅ ∇νb
        @test dprod ≈ ∇prod atol = ATOL rtol = RTOL
    end
    # 3) the problem data (here we made it so that they are the same for the forward
    #    and backward pass but we could have picked any other dQ, dq, ... for the forward pass
    #    and we would still have pprod = ∇prod = dprod
    pprod = dQf ⋅ dQb + dqf ⋅ dqb + dGf ⋅ dGb + dhf ⋅ dhb + dAf ⋅ dAb + dbf ⋅ dbb
    @test pprod ≈ pprod atol = ATOL rtol = RTOL
end

function qp_test(solver; kws...)
    @testset "With $(lt ? "LessThan" : "GreaterThan") constraints" for lt in [true, false]
        @testset "With$(set_zero ? "" : "out") setting zero tangents" for set_zero in [true, false]
            @testset "With$(canonicalize ? "" : "out") canonicalization" for canonicalize in [true, false]
                qp_test(solver, lt, set_zero, canonicalize; kws...)
            end
        end
    end
end

function qp_test_with_solutions(solver;
    dzb = nothing,
    n = length(dzb),
    q = nothing,
    dqf = zeros(n),
    Q = zeros(n, n),
    dQf = zeros(n, n),
    h = zeros(0),
    nle = length(h),
    dhf = zeros(nle),
    G = zeros(0, n),
    dGf = zeros(nle, n),
    b = zeros(0),
    neq = length(b),
    dbf = zeros(neq),
    A = zeros(0, n),
    dAf = zeros(neq, n),
    kws...
)
    @testset "Without known solutions" begin
        qp_test(solver;
            dzb = dzb,
            q = q,
            dqf = dqf,
            Q = Q,
            dQf = dQf,
            h = h,
            dhf = dhf,
            G = G,
            dGf = dGf,
            b = b,
            dbf = dbf,
            A = A,
            dAf = dAf,
        )
    end
    @testset "With known solutions" begin
        qp_test(solver;
            dzb = dzb,
            q = q,
            dqf = dqf,
            Q = Q,
            dQf = dQf,
            h = h,
            dhf = dhf,
            G = G,
            dGf = dGf,
            b = b,
            dbf = dbf,
            A = A,
            dAf = dAf,
            kws...
        )
    end
end
