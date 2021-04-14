function build_quad_diff_cache!(model)
    problem_data, index_map = get_problem_data(model.optimizer)
    (
        Q, q, G, h, A, b,
        nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = problem_data

    z = model.primal_optimal

    # separate λ, ν

    λ = MOI.get.(model.optimizer, MOI.ConstraintDual(), le_con_idx)
    append!(
        λ,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), ge_con_idx),
    )
    append!(
        λ,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), le_con_sv_idx),
    )
    append!(
        λ,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), ge_con_sv_idx),
    )
    ν = MOI.get.(model.optimizer, MOI.ConstraintDual(), eq_con_idx)
    append!(
        ν,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), eq_con_sv_idx),
    )

    LHS = create_LHS_matrix(z, λ, Q, G, h, A)
    model.gradient_cache = QPCache(
        problem_data,
        λ,
        ν,
        z,
        LHS,
        index_map,
    )
    return nothing
end
