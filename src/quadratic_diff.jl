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

    z = MOI.get(model.optimizer, MOI.VariablePrimal(), var_list)

    # separate λ, ν

    λ = -MOI.get.(model.optimizer, MOI.ConstraintDual(), le_con_idx)
    append!(
        λ,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), ge_con_idx),
    )
    append!(
        λ,
        -MOI.get.(model.optimizer, MOI.ConstraintDual(), le_con_sv_idx),
    )
    append!(
        λ,
        MOI.get.(model.optimizer, MOI.ConstraintDual(), ge_con_sv_idx),
    )
    # We want to stay consistent with the variable `ν` defined in (3) of
    # Left hand side of eq. (6) in https://arxiv.org/pdf/1703.00443.pdf
    # However, in eq. (6), they put it in the lagrangian as
    # `+ ν ⋅ (Az - b)`
    # while in MOI, we put it as
    # `- ν ⋅ (Az - b)`
    # so the we should reverse the sign if we want to use the same equations
    # as in the paper.
    ν = -MOI.get.(model.optimizer, MOI.ConstraintDual(), eq_con_idx)
    append!(
        ν,
        -MOI.get.(model.optimizer, MOI.ConstraintDual(), eq_con_sv_idx),
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


# TODO: create test functions for the methods

# """
#     Left hand side of eqn(6) in https://arxiv.org/pdf/1703.00443.pdf
# """
# function create_LHS_matrix(z, λ, Q, G, h, A=nothing)
#     if A == nothing || size(A)[1] == 0
#         return [Q                G';
#                 Diagonal(λ) * G    Diagonal(G * z - h)]
#     else
#         @assert size(A)[2] == size(G)[2]
#         p, n = size(A)
#         m    = size(G)[1]
#         return [Q                  G'                    A';
#                 Diagonal(λ) * G    Diagonal(G * z - h)   zeros(m, p);
#                 A                  zeros(p, m)           zeros(p, p)]
#     end
# end


"""
    create_LHS_matrix(z, λ, Q, G, h, A=nothing)

Inverse matrix specified on RHS of eqn(7) in https://arxiv.org/pdf/1703.00443.pdf

Helper method while calling `_backward_quad`
"""
function create_LHS_matrix(z, λ, Q, G, h, A=nothing)::AbstractMatrix{Float64}
    if A === nothing || size(A)[1] == 0
        return [Q         G' * Diagonal(λ);
                G         Diagonal(G * z - h)]
    else
        p, n = size(A)
        m    = size(G, 1)
        if n != size(G, 2)
            throw(DimensionError("Sizes of $A and $G do not match"))
        end
        return [Q         G' * Diagonal(λ)       A';
                G         Diagonal(G * z - h)    spzeros(m, p);
                A         spzeros(p, m)          spzeros(p, p)]
    end
end
# TODO: this is the transpose, check back for usage

# """
#     Right hand side of eqn(6) in https://arxiv.org/pdf/1703.00443.pdf
# """
# function create_RHS_matrix(z, dQ, dq, λ, dG, dh, ν=nothing, dA=nothing, db=nothing)
#     if dA == nothing || size(dA)[1] == 0
#         return -[dQ * z + dq + dG' * λ      ;
#                  Diagonal(λ) * (dG * z - dh)]
#     else
#         return -[dQ * z + dq + dG' * λ + dA' * ν;
#                  Diagonal(λ) * (dG * z - dh)    ;
#                  dA * z - db                    ]
#     end
# end

const _QP_SET_TYPES = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    # MOI.Interval{Float64},
}

const _QP_FUNCTION_TYPES = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
}

_qp_supported(::Type{F}, ::Type{S}) where {F <: _QP_FUNCTION_TYPES, S <: _QP_SET_TYPES} = true
_qp_supported(::Type{F}, ::Type{S}) where {F, S} = false
function _qp_supported(model::MOI.AbstractOptimizer)
    return all(FS -> _qp_supported(FS...), MOI.get(model, MOI.ListOfConstraints()))
end

"""
    get_problem_data(model::MOI.AbstractOptimizer)

Return problem parameters as matrices along with other program info such as number of constraints, variables, etc
"""
function get_problem_data(model::MOI.AbstractOptimizer)
    for (F, S) in MOI.get(model, MOI.ListOfConstraints())
        if !_qp_supported(F, S)
            throw(MOI.UnsupportedConstraint{F,S}("DiffOpt does not support this constraint type for its Quadratic Programming differentiation. Maybe try the Conic Programming differentiation ? For this, do `MOI.set(model, DiffOpt.ProgramClass(), DiffOpt.CONIC)`."))
        end
    end
    var_list = MOI.get(model, MOI.ListOfVariableIndices())
    nz = length(var_list)

    index_map = MOIU.IndexMap(nz)
    for (i,vi) in enumerate(var_list)
        index_map[vi] = VI(i)
    end

    # handle inequality constraints
    le_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.LessThan{Float64},
                        }())
    ge_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.GreaterThan{Float64},
                        }())
    nineq_le = length(le_con_idx)
    nineq_ge = length(ge_con_idx)
    le_con_sv_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.SingleVariable,
                            MOI.LessThan{Float64},
                        }())
    ge_con_sv_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.SingleVariable,
                            MOI.GreaterThan{Float64},
                        }())
    nineq_sv_le = length(le_con_sv_idx)
    nineq_sv_ge = length(ge_con_sv_idx)

    G = spzeros(nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge, nz)
    h = spzeros(nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge)

    ineq_cont = 0
    eq_cont = 0

    for i in 1:nineq_le
        con = le_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for (j, var_idx) in enumerate(var_list)
            for term in func.terms
                if term.variable_index == var_idx
                    G[i,j] = MOI.coefficient(term)
                end
            end
        end
        h[i] = set.upper - func.constant

        ineq_cont += 1
        index_map[con] =
            CI{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(ineq_cont)
    end
    for i in 1:nineq_ge
        # note: ax >= b needs to be converted in Gx <= h form
        con = ge_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for (j, var_idx) in enumerate(var_list)
            for term in func.terms
                if term.variable_index == var_idx
                    G[i+nineq_le,j] = -MOI.coefficient(term)
                end
            end
        end
        h[i] = func.constant - set.lower
        ineq_cont += 1
        index_map[con] =
            CI{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(ineq_cont)
    end
    for i in eachindex(le_con_sv_idx)
        con = le_con_sv_idx[i]
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        vidx = findfirst(v -> v == func.variable, var_list)
        G[i+nineq_le+nineq_ge,vidx] = 1
        h[i+nineq_le+nineq_ge] = MOI.constant(set)
        ineq_cont += 1
        index_map[con] =
            CI{MOI.SingleVariable, MOI.LessThan{Float64}}(ineq_cont)
    end
    for i in eachindex(ge_con_sv_idx)
        # note: x >= b needs to be converted in Gx <= h form
        con = ge_con_sv_idx[i]
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        vidx = findfirst(v -> v == func.variable, var_list)
        G[i+nineq_le+nineq_ge+nineq_sv_le,vidx] = -1
        h[i+nineq_le+nineq_ge+nineq_sv_le] = -MOI.constant(set)
        ineq_cont += 1
        index_map[con] =
            CI{MOI.SingleVariable, MOI.GreaterThan{Float64}}(ineq_cont)
    end

    # handle equality constraints
    eq_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.EqualTo{Float64}
                        }())
    neq = length(eq_con_idx)

    eq_con_sv_idx = MOI.get(
        model,
        MOI.ListOfConstraintIndices{
            MOI.SingleVariable,
            MOI.EqualTo{Float64}
        }())
    neq_sv = length(eq_con_sv_idx)

    A = spzeros(neq + neq_sv, nz)
    b = spzeros(neq + neq_sv)

    for i in 1:neq
        con = eq_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for x in func.terms
            # never nothing, variable is present
            vidx = findfirst(v -> v == x.variable_index, var_list)
            A[i, vidx] = x.coefficient
        end
        b[i] = set.value - func.constant

        eq_cont += 1
        index_map[con] =
            CI{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(eq_cont)
    end
    for i in 1:neq_sv
        con = eq_con_sv_idx[i]
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        vidx = findfirst(v -> v == func.variable, var_list)
        A[i+neq,vidx] = 1
        b[i+neq] = set.value
        eq_cont += 1
        index_map[con] =
            CI{MOI.SingleVariable, MOI.EqualTo{Float64}}(eq_cont)
    end


    # handle objective
    # works both for any objective function convertible to a ScalarQuadraticFunction.
    # So in particular SingleVariable, ScalarAffineFunction and ScalarQuadraticFunction should work.
    objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    # TODO Remove, this is a temporary workaround for https://github.com/jump-dev/Ipopt.jl/pull/275
    objective_function = convert(MOI.ScalarQuadraticFunction{Float64}, objective_function)
    sparse_array_obj = sparse_array_representation(objective_function, nz, index_map)

    return (
        sparse_array_obj.quadratic_terms, sparse_array_obj.affine_terms,
        G, h, A, b,
        nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ), index_map
end
