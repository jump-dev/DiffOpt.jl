# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test
using JuMP
import DiffOpt
import MathOptInterface as MOI
import LinearAlgebra: ⋅
import LinearAlgebra
import SparseArrays

macro _test(computed, expected::Symbol)
    exp = esc(expected)
    com = esc(computed)
    ato = esc(:atol)
    rto = esc(:rtol)
    return :(
        if $exp === nothing
            @test ($exp = $com) isa Any
        else
            @test $com ≈ $exp atol = $ato rtol = $rto
        end
    )
end

"""
    qp_test(solver; kws...)

Check our results for QPs using the notations of [AK17].

[AK17] Amos, Brandon, and J. Zico Kolter. "Optnet: Differentiable optimization
as a layer in neural networks." International Conference on Machine Learning.
PMLR, 2017. https://arxiv.org/pdf/1703.00443.pdf
"""
function qp_test(
    solver,
    diff_model,
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
    fix_indices = Int[],
    fix_values = Float64[],
    nfix = length(fix_values),
    ub_indices = Int[],
    ub_values = Float64[],
    nub = length(ub_values),
    lb_indices = Int[],
    lb_values = Float64[],
    nlb = length(lb_values),
    h = zeros(0),
    nle = length(h),
    dhf = zeros(nle + nub + nlb),
    dhb = nothing,
    G = zeros(0, n),
    dGf = zeros(nle + nub + nlb, n),
    dGb = nothing,
    b = zeros(0),
    neq = length(b),
    dbf = zeros(neq + nfix),
    dbb = nothing,
    A = zeros(0, n),
    dAf = zeros(neq + nfix, n),
    dAb = nothing,
    z = nothing,
    dzf = nothing,
    ∇zf = nothing,
    ∇zb = nothing,
    λ = nothing,
    dλf = zeros(nle + nub + nlb),
    dλb = zeros(nle + nub + nlb),
    ∇λf = nothing,
    ∇λb = nothing,
    ν = nothing,
    dνf = nothing,
    dνb = zeros(neq + nfix),
    ∇νf = nothing,
    ∇νb = nothing,
    atol = ATOL,
    rtol = RTOL,
)
    if !all(iszero, Q) && diff_model == DiffOpt.ConicProgram.Model
        return # TODO https://github.com/jump-dev/DiffOpt.jl/pull/231
    end
    n = length(q)
    @assert n == LinearAlgebra.checksquare(Q)
    @assert n == size(A, 2)
    @assert n == size(G, 2)
    @assert length(fix_values) == length(fix_indices)
    model = DiffOpt.diff_optimizer(solver)
    MOI.set(model, MOI.Silent(), true)

    v = MOI.add_variables(model, n)

    _sign(x, a) = a == lt ? -x : x

    if lt
        cle = MOI.add_constraint.(model, G * v, MOI.LessThan.(h))
    else
        cle = MOI.add_constraint.(model, -G * v, MOI.GreaterThan.(-h))
    end
    if !iszero(nub)
        cub =
            MOI.add_constraint.(model, v[ub_indices], MOI.LessThan.(ub_values))
        G = vcat(G, SparseArrays.sparse(1:nub, ub_indices, ones(nub), nub, n))
        h = vcat(h, ub_values)
    end
    if !iszero(nlb)
        clb =
            MOI.add_constraint.(
                model,
                v[lb_indices],
                MOI.GreaterThan.(lb_values),
            )
        G = vcat(G, SparseArrays.sparse(1:nlb, lb_indices, -ones(nlb), nlb, n))
        h = vcat(h, -lb_values)
    end

    ceq = MOI.add_constraint.(model, A * v, MOI.EqualTo.(b))
    if !iszero(nfix)
        cfix =
            MOI.add_constraint.(model, v[fix_indices], MOI.EqualTo.(fix_values))
        A = vcat(
            A,
            SparseArrays.sparse(1:nfix, fix_indices, ones(nfix), nfix, n),
        )
        b = vcat(b, fix_values)
    end

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    if iszero(Q)
        obj = LinearAlgebra.dot(q, v)
    else
        obj = LinearAlgebra.dot(v, Q / 2, v) + LinearAlgebra.dot(q, v)
    end
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)

    MOI.optimize!(model)

    @_test(MOI.get(model, MOI.VariablePrimal(), v), z)

    # The sign as reversed as [AK17]
    # use a different convention for the dual
    _ν = -MOI.get.(model, MOI.ConstraintDual(), ceq)
    if !iszero(nfix)
        _ν = vcat(_ν, -MOI.get.(model, MOI.ConstraintDual(), cfix))
    end
    @_test(convert(Vector{Float64}, _ν), ν)
    _λ = _sign(MOI.get.(model, MOI.ConstraintDual(), cle), true)
    if !iszero(nub)
        _λ = vcat(_λ, -MOI.get.(model, MOI.ConstraintDual(), cub))
    end
    if !iszero(nlb)
        _λ = vcat(_λ, MOI.get.(model, MOI.ConstraintDual(), clb))
    end
    @_test(convert(Vector{Float64}, _λ), λ)

    MOI.set(model, DiffOpt.ModelConstructor(), diff_model)

    #dobjb = v' * (dQb / 2.0) * v + dqb' * v
    # TODO, it should .-
    #dleb = dGb * v .+ dhb
    #deqb = dAb * v .+ dbb
    @assert dzb !== nothing
    @testset "Backward pass" begin
        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), v, dzb)

        DiffOpt.reverse_differentiate!(model)
        @test !isnan(MOI.get(model, DiffOpt.DifferentiateTimeSec()))

        dobjb = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
        spb = DiffOpt.sparse_array_representation(
            DiffOpt.standard_form(dobjb),
            n,
            dobjb.index_map,
        )
        @_test(spb.quadratic_terms, dQb)
        @_test(spb.affine_terms, dqb)

        # FIXME should multiply by -1 if lt is false
        funcs = MOI.get.(model, DiffOpt.ReverseConstraintFunction(), cle)
        if !iszero(nub)
            funcs = vcat(
                funcs,
                MOI.get.(model, DiffOpt.ReverseConstraintFunction(), cub),
            )
        end
        if !iszero(nlb)
            # FIXME should multiply by -1
            funcs = vcat(
                funcs,
                MOI.get.(model, DiffOpt.ReverseConstraintFunction(), clb),
            )
        end
        @_test(convert(Vector{Float64}, _sign(MOI.constant.(funcs), true)), dhb)
        @_test(
            Float64[
                _sign(JuMP.coefficient(funcs[i], vi), false) for
                i in eachindex(funcs), vi in v
            ],
            dGb
        )

        funcs = MOI.get.(model, DiffOpt.ReverseConstraintFunction(), ceq)
        if !iszero(nfix)
            funcs = vcat(
                funcs,
                MOI.get.(model, DiffOpt.ReverseConstraintFunction(), cfix),
            )
        end
        @_test(convert(Vector{Float64}, -MOI.constant.(funcs)), dbb)
        @_test(
            Float64[
                JuMP.coefficient(funcs[i], vi) for i in eachindex(funcs),
                vi in v
            ],
            dAb
        )
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
        @_test(LinearAlgebra.Diagonal(λ) * ∇λb * z' + λ * ∇zb', dGb)
    end

    # Test against [AK17, eq. (7)]
    if ∇λb !== nothing
        @_test(-(Q * ∇zb + G' * (λ .* ∇λb) + A' * ∇νb), dzb)
    end
    if ∇λb !== nothing
        @_test(-(G * ∇zb + (G * z - h) .* ∇λb), dλb)
    end
    @_test(-A * ∇zb, dνb)

    dobjf = v' * (dQf / 2.0) * v + dqf' * v
    dlef = dGf * v .- dhf
    deqf = dAf * v .- dbf

    @testset "Forward pass" begin
        MOI.set(model, DiffOpt.ForwardObjectiveFunction(), dobjf)
        for (j, jc) in enumerate(cle)
            func = dlef[j]
            canonicalize && MOI.Utilities.canonicalize!(func)
            if set_zero || !MOI.iszero(dlef[j])
                MOI.set(
                    model,
                    DiffOpt.ForwardConstraintFunction(),
                    jc,
                    _sign(func, false),
                )
            end
        end
        for (j, jc) in enumerate(ceq)
            func = deqf[j]
            canonicalize && MOI.Utilities.canonicalize!(func)
            if set_zero || !MOI.iszero(func)
                MOI.set(model, DiffOpt.ForwardConstraintFunction(), jc, func)
            end
        end
        if !iszero(nfix)
            for (j, jc) in enumerate(cfix)
                func = deqf[length(ceq)+j]
                canonicalize && MOI.Utilities.canonicalize!(func)
                if set_zero || !MOI.iszero(func)
                    # TODO FIXME should work if we drop support for `VariableIndex` and we let the Functionize bridge do the work
                    @test_throws MOI.UnsupportedAttribute MOI.set(
                        model,
                        DiffOpt.ForwardConstraintFunction(),
                        jc,
                        func,
                    )
                end
            end
        end

        DiffOpt.forward_differentiate!(model)
        @test !isnan(MOI.get(model, DiffOpt.DifferentiateTimeSec()))

        @_test(MOI.get.(model, DiffOpt.ForwardVariablePrimal(), v), dzf)
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
    pprod =
        dQf ⋅ dQb + dqf ⋅ dqb + dGf ⋅ dGb + dhf ⋅ dhb + dAf ⋅ dAb + dbf ⋅ dbb
    @test pprod ≈ pprod atol = ATOL rtol = RTOL
    return
end

function qp_test(solver, lt, set_zero, canonicalize; kws...)
    @testset "With $diff_model" for diff_model in [
        DiffOpt.ConicProgram.Model,
        DiffOpt.QuadraticProgram.Model,
    ]
        qp_test(solver, diff_model, lt, set_zero, canonicalize; kws...)
    end
    return
end

function qp_test(solver; kws...)
    @testset "With $(lt ? "LessThan" : "GreaterThan") constraints" for lt in [
        true,
        false,
    ]
        @testset "With$(set_zero ? "" : "out") setting zero tangents" for set_zero in
                                                                          [
            true,
            false,
        ]
            @testset "With$(canonicalize ? "" : "out") canonicalization" for canonicalize in
                                                                             [
                true,
                false,
            ]
            end
        end
    end
    return
end

function qp_test_with_solutions(
    solver;
    dzb = nothing,
    n = length(dzb),
    q = nothing,
    dqf = zeros(n),
    Q = zeros(n, n),
    dQf = zeros(n, n),
    fix_indices = Int[],
    fix_values = Float64[],
    nfix = length(fix_values),
    ub_indices = Int[],
    ub_values = Float64[],
    nub = length(ub_values),
    lb_indices = Int[],
    lb_values = Float64[],
    nlb = length(lb_values),
    h = zeros(0),
    nle = length(h),
    dhf = zeros(nle + nub + nlb),
    G = zeros(0, n),
    dGf = zeros(nle + nub + nlb, n),
    b = zeros(0),
    neq = length(b),
    dbf = zeros(neq + nfix),
    A = zeros(0, n),
    dAf = zeros(neq + nfix, n),
    kws...,
)
    @testset "Without known solutions" begin
        qp_test(
            solver;
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
            fix_indices = fix_indices,
            fix_values = fix_values,
            ub_indices = ub_indices,
            ub_values = ub_values,
            lb_indices = lb_indices,
            lb_values = lb_values,
        )
    end
    @testset "With known solutions" begin
        qp_test(
            solver;
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
            fix_indices = fix_indices,
            fix_values = fix_values,
            ub_indices = ub_indices,
            ub_values = ub_values,
            lb_indices = lb_indices,
            lb_values = lb_values,
            kws...,
        )
    end
    return
end
