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

Helper method while calling [`backward_quad`](@ref)
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


is_equality(set::MOI.AbstractSet) = false
is_equality(set::MOI.EqualTo) = true


"""
    get_problem_data(model::MOI.AbstractOptimizer)

Return problem parameters as matrices along with other program info such as number of constraints, variables, etc
"""
function get_problem_data(model::MOI.AbstractOptimizer)
    var_list = MOI.get(model, MOI.ListOfVariableIndices())
    nz = size(var_list, 1)

    # handle inequality constraints
    ineq_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.LessThan{Float64}
                        }())
    nineq = length(ineq_con_idx)

    G = spzeros(nineq, nz)
    h = spzeros(nineq)

    for i in 1:nineq
        con = ineq_con_idx[i]

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
    end

    # handle equality constraints
    eq_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.EqualTo{Float64}
                        }())
    neq = length(eq_con_idx)

    A = spzeros(neq, nz)
    b = spzeros(neq)

    for i in 1:neq
        con = eq_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for x in func.terms
            A[i, x.variable_index.value] = x.coefficient
        end
        b[i] = set.value - func.constant
    end


    # handle objective
    # works both for affine and quadratic objective functions
    objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    Q = spzeros(nz, nz)
    q = spzeros(nz)

    if objective_function isa MathOptInterface.ScalarAffineFunction{Float64}
        for x in objective_function.terms
            q[x.variable_index.value] = x.coefficient
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

    return Q, q, G, h, A, b, nz, var_list, nineq, ineq_con_idx, neq, eq_con_idx
end

# might slow down computation
# need to find a faster way
function CSRToCSC(B::MatOI.SparseMatrixCSRtoCSC{T, Int}) where {T}
    A = spzeros(T, B.m, B.n)
    last = 0
    for i in 1:B.n
        rnge = (last+1):B.colptr[i]
        A[(1 .+ B.rowval[rnge]), i] = B.nzval[rnge]
        last = B.colptr[i]
    end
    return A
end
