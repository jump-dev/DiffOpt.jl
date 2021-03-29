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
function create_LHS_matrix(z, λ, Q, G, h, A=nothing)
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

# TODO: use MatOI for building the matrix
"""
    get_problem_data(model::MOI.AbstractOptimizer)

Return problem parameters as matrices along with other program info such as number of constraints, variables, etc
"""
function get_problem_data(model::MOI.AbstractOptimizer)
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
    # works both for affine and quadratic objective functions
    objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    Q = spzeros(nz, nz)
    q = spzeros(nz)

    if objective_function isa MathOptInterface.ScalarAffineFunction{Float64}
        for x in objective_function.terms
            vidx = findfirst(v -> v == x.variable_index, var_list)
            q[vidx] = x.coefficient
        end
    elseif objective_function isa MathOptInterface.ScalarQuadraticFunction{Float64}

        var_to_id = Dict(var_list .=> 1:nz)

        for quad in objective_function.quadratic_terms
            i = var_to_id[quad.variable_index_1]
            j = var_to_id[quad.variable_index_2]
            Q[i,j] = quad.coefficient
            Q[j,i] = quad.coefficient
        end

        q = MOI.coefficient.(objective_function.affine_terms)
    end

    return (
        Q, q, G, h, A, b,
        nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ), index_map
end

# used for testing mostly
# computes the tuple `(s, y)`, vectorized forms of the slack and duals of inequality constraints
function _slack_dual_vectors(model)
    cone_types = unique!([S for (_, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    # fix optimization sense
    if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        conic_form.sense = MOI.MIN_SENSE
        conic_form.c = -conic_form.c
    end    
    s = map_rows(model.optimizer, conic_form, index_map, Flattened{Float64}()) do (ci, r)
        MOI.get(model, MOI.ConstraintPrimal(), ci)
    end
    y = map_rows(model.optimizer, conic_form, index_map, Flattened{Float64}()) do (ci, r)
        MOI.get(model, MOI.ConstraintDual(), ci)
    end
    return (s, y)
end
