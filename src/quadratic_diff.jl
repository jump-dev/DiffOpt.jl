function build_quad_diff_cache!(model)
    problem_data = get_problem_data(model.optimizer)
    (
        Q, q, G, h, A, b, nz, var_list,
        nineq, ineq_con_idx, nineq_sv_le, ineq_con_sv_le_idx, neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = problem_data

    # separate λ, ν

    λ = MOI.get.(model.optimizer, MOI.ConstraintDual(), ineq_con_idx)
    append!(
        λ,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), ineq_con_sv_le_idx)
    )
    ν = MOI.get.(model.optimizer, MOI.ConstraintDual(), eq_con_idx)
    append!(
        ν,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), eq_con_sv_idx)
    )

    LHS = create_LHS_matrix(z, λ, Q, G, h, A)
    model.gradient_cache = QPCache(
        problem_data,
        λ,
        ν,
        LHS,
    )
end