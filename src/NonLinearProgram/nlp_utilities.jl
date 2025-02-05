# Copyright (c) 2025: Andrew Rosemberg and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    fill_off_diagonal(H)

Filling the off-diagonal elements of a sparse matrix to make it symmetric.
"""
function fill_off_diagonal(H)
    ret = H + H'
    row_vals = SparseArrays.rowvals(ret)
    non_zeros = SparseArrays.nonzeros(ret)
    for col in 1:size(ret, 2)
        for i in SparseArrays.nzrange(ret, col)
            if col == row_vals[i]
                non_zeros[i] /= 2
            end
        end
    end
    return ret
end

"""
    compute_optimal_hessian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{JuMP.ConstraintRef}, x::Vector{JuMP.VariableRef})

Compute the Hessian of the Lagrangian calculated at the optimal solution.
"""
function compute_optimal_hessian(
    model::Model,
    rows::Vector{MOI.Nonlinear.ConstraintIndex},
)
    sense_multiplier = sense_mult(model)
    evaluator = model.cache.evaluator
    y = [model.y[model.model.nlp_index_2_constraint[row].value] for row in rows]
    hessian_sparsity = MOI.hessian_lagrangian_structure(evaluator)
    I = [i for (i, _) in hessian_sparsity]
    J = [j for (_, j) in hessian_sparsity]
    V = zeros(length(hessian_sparsity))
    # The signals are being adjusted to match the Ipopt convention (inner.mult_g)
    # but we don't know if we need to adjust the objective function multiplier
    MOI.eval_hessian_lagrangian(
        evaluator,
        V,
        model.x,
        1.0,
        -sense_multiplier * y,
    )
    num_vars = length(model.x)
    H = SparseArrays.sparse(I, J, V, num_vars, num_vars)
    return fill_off_diagonal(H)
end

"""
    compute_optimal_jacobian(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{JuMP.ConstraintRef}, x::Vector{JuMP.VariableRef})

Compute the Jacobian of the constraints calculated at the optimal solution.
"""
function compute_optimal_jacobian(
    model::Model,
    rows::Vector{MOI.Nonlinear.ConstraintIndex},
)
    evaluator = model.cache.evaluator
    jacobian_sparsity = MOI.jacobian_structure(evaluator)
    I = [i for (i, _) in jacobian_sparsity]
    J = [j for (_, j) in jacobian_sparsity]
    V = zeros(length(jacobian_sparsity))
    MOI.eval_constraint_jacobian(evaluator, V, model.x)
    A = SparseArrays.sparse(I, J, V, length(rows), length(model.x))
    return A
end

"""
    compute_optimal_hess_jac(evaluator::MOI.Nonlinear.Evaluator, rows::Vector{JuMP.ConstraintRef}, x::Vector{JuMP.VariableRef})

Compute the Hessian of the Lagrangian and Jacobian of the constraints calculated at the optimal solution.
"""
function compute_optimal_hess_jac(
    model::Model,
    rows::Vector{MOI.Nonlinear.ConstraintIndex},
)
    hessian = compute_optimal_hessian(model, rows)
    jacobian = compute_optimal_jacobian(model, rows)

    return hessian, jacobian
end

"""
    create_evaluator(form::Form)

Create the evaluator for the NLP.
"""
function create_evaluator(form::Form)
    nlp = form.model
    backend = MOI.Nonlinear.SparseReverseMode()
    evaluator = MOI.Nonlinear.Evaluator(
        nlp,
        backend,
        MOI.VariableIndex.(1:form.num_variables),
    )
    MOI.initialize(evaluator, [:Hess, :Jac])
    return evaluator
end

"""
    is_less_inequality(con::MOI.ConstraintIndex{F, S}) where {F, S}

Check if the constraint is a less than inequality.
"""
function is_less_inequality(
    ::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarNonlinearFunction,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
    },
    S<:MOI.LessThan,
}
    return true
end

function is_less_inequality(::MOI.ConstraintIndex{F,S}) where {F,S}
    return false
end

function is_greater_inequality(::MOI.ConstraintIndex{F,S}) where {F,S}
    return false
end

"""
    is_greater_inequality(con::MOI.ConstraintIndex{F, S}) where {F, S}

Check if the constraint is a greater than inequality.
"""
function is_greater_inequality(
    ::MOI.ConstraintIndex{F,S},
) where {
    F<:Union{
        MOI.ScalarNonlinearFunction,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
    },
    S<:MOI.GreaterThan,
}
    return true
end

"""
    find_inequalities(cons::Vector{JuMP.ConstraintRef})

Find the indices of the inequality constraints.
"""
function find_inequalities(model::Form)
    num_cons = length(model.list_of_constraint)
    leq_locations = zeros(num_cons)
    geq_locations = zeros(num_cons)
    for con in keys(model.list_of_constraint)
        if is_less_inequality(con)
            leq_locations[model.constraints_2_nlp_index[con].value] = true
        end
        if is_greater_inequality(con)
            geq_locations[model.constraints_2_nlp_index[con].value] = true
        end
    end
    return findall(x -> x == 1, leq_locations),
    findall(x -> x == 1, geq_locations)
end

"""
    compute_solution_and_bounds(model::Model; tol=1e-6)

Compute the solution and bounds of the primal variables.
"""
function compute_solution_and_bounds(model::Model; tol = 1e-6)
    sense_multiplier = sense_mult(model)
    num_vars = get_num_primal_vars(model)
    form = model.model
    leq_locations = model.cache.leq_locations
    geq_locations = model.cache.geq_locations
    ineq_locations = vcat(geq_locations, leq_locations)
    num_leq = length(leq_locations)
    num_geq = length(geq_locations)
    num_ineq = num_leq + num_geq
    has_up = model.cache.has_up
    has_low = model.cache.has_low
    primal_vars = model.cache.primal_vars
    cons = model.cache.cons
    # Primal solution: value.([primal_vars; slack_vars])
    model_cons_leq =
        [form.nlp_index_2_constraint[con] for con in cons[leq_locations]]
    model_cons_geq =
        [form.nlp_index_2_constraint[con] for con in cons[geq_locations]]
    s_leq =
        [model.s[con.value] for con in model_cons_leq] - [form.leq_values[con] for con in model_cons_leq]
    s_geq =
        [model.s[con.value] for con in model_cons_geq] - [form.geq_values[con] for con in model_cons_geq]
    primal_idx = [i.value for i in model.cache.primal_vars]
    X = [model.x[primal_idx]; s_leq; s_geq]

    # value and dual of the lower bounds
    V_L = spzeros(num_vars + num_ineq)
    X_L = spzeros(num_vars + num_ineq)
    for (i, j) in enumerate(has_low)
        V_L[j] =
            model.y[form.constraint_lower_bounds[primal_vars[j].value].value] *
            sense_multiplier
        #dual.(LowerBoundRef(primal_vars[j])) * sense_multiplier
        #
        if sense_multiplier == 1.0
            V_L[j] <= -tol &&
                @info "Dual of lower bound must be positive" i V_L[j]
        else
            V_L[j] >= tol &&
                @info "Dual of lower bound must be negative" i V_L[j]
        end
        #
        X_L[j] = form.lower_bounds[primal_vars[j].value]
    end
    for (i, con) in enumerate(cons[geq_locations])
        # By convention jump dual will allways be positive for geq constraints
        # but for ipopt it will be positive if min problem and negative if max problem
        V_L[num_vars+i] =
            model.y[form.nlp_index_2_constraint[con].value] * sense_multiplier
        #
        if sense_multiplier == 1.0
            V_L[num_vars+i] <= -tol &&
                @info "Dual of geq constraint must be positive" i V_L[num_vars+i]
        else
            V_L[num_vars+i] >= tol &&
                @info "Dual of geq constraint must be negative" i V_L[num_vars+i]
        end
    end
    # value and dual of the upper bounds
    V_U = spzeros(num_vars + num_ineq)
    X_U = spzeros(num_vars + num_ineq)
    for (i, j) in enumerate(has_up)
        V_U[j] =
            model.y[form.constraint_upper_bounds[primal_vars[j].value].value] *
            (-sense_multiplier)
        # dual.(UpperBoundRef(primal_vars[j])) * (- sense_multiplier)
        if sense_multiplier == 1.0
            V_U[j] <= -tol &&
                @info "Dual of upper bound must be positive" i V_U[i]
        else
            V_U[j] >= tol &&
                @info "Dual of upper bound must be negative" i V_U[i]
        end
        #
        X_U[j] = form.upper_bounds[primal_vars[j].value]
    end
    for (i, con) in enumerate(cons[leq_locations])
        # By convention jump dual will allways be negative for leq constraints
        # but for ipopt it will be positive if min problem and negative if max problem
        V_U[num_vars+num_geq+i] =
            model.y[form.nlp_index_2_constraint[con].value] *
            (-sense_multiplier)
        # dual.(con) * (- sense_multiplier)
        if sense_multiplier == 1.0
            V_U[num_vars+num_geq+i] <= -tol &&
                @info "Dual of leq constraint must be positive" i V_U[num_vars+i]
        else
            V_U[num_vars+num_geq+i] >= tol &&
                @info "Dual of leq constraint must be negative" i V_U[num_vars+i]
        end
    end
    return X,
    V_L,
    X_L,
    V_U,
    X_U,
    leq_locations,
    geq_locations,
    ineq_locations,
    vcat(has_up, collect(num_vars+num_geq+1:num_vars+num_geq+num_leq)),
    vcat(has_low, collect(num_vars+1:num_vars+num_geq)),
    cons
end

"""
    build_sensitivity_matrices(model::Model, cons::Vector{MOI.Nonlinear.ConstraintIndex}, _X::AbstractVector, _V_L::AbstractVector, _X_L::AbstractVector, _V_U::AbstractVector, _X_U::AbstractVector, leq_locations::Vector{Z}, geq_locations::Vector{Z}, ineq_locations::Vector{Z}, has_up::Vector{Z}, has_low::Vector{Z})

Build the M (KKT Jacobian w.r.t. solution) and N (KKT Jacobian w.r.t. parameters) matrices for the sensitivity analysis.
"""
function build_sensitivity_matrices(
    model::Model,
    cons::Vector{MOI.Nonlinear.ConstraintIndex},
    _X::AbstractVector,
    _V_L::AbstractVector,
    _X_L::AbstractVector,
    _V_U::AbstractVector,
    _X_U::AbstractVector,
    leq_locations::Vector{Z},
    geq_locations::Vector{Z},
    ineq_locations::Vector{Z},
    has_up::Vector{Z},
    has_low::Vector{Z},
) where {Z<:Integer}
    # Setting
    num_vars = get_num_primal_vars(model)
    num_parms = get_num_params(model)
    num_cons = get_num_constraints(model)
    num_ineq = length(ineq_locations)
    num_low = length(has_low)
    num_up = length(has_up)

    # Primal solution
    X_lb = spzeros(num_low, num_low)
    X_ub = spzeros(num_up, num_up)
    V_L = spzeros(num_low, num_vars + num_ineq)
    V_U = spzeros(num_up, num_vars + num_ineq)
    I_L = spzeros(num_vars + num_ineq, num_low)
    I_U = spzeros(num_vars + num_ineq, num_up)

    # value and dual of the lower bounds
    for (i, j) in enumerate(has_low)
        V_L[i, j] = _V_L[j]
        X_lb[i, i] = _X[j] - _X_L[j]
        I_L[j, i] = -1
    end
    # value and dual of the upper bounds
    for (i, j) in enumerate(has_up)
        V_U[i, j] = _V_U[j]
        X_ub[i, i] = _X_U[j] - _X[j]
        I_U[j, i] = 1
    end

    # Function Derivatives
    hessian, jacobian = compute_optimal_hess_jac(model, cons)
    primal_idx = [i.value for i in model.cache.primal_vars]
    params_idx = [i.value for i in model.cache.params]
    # Hessian of the lagrangian wrt the primal variables
    W = spzeros(num_vars + num_ineq, num_vars + num_ineq)
    W[1:num_vars, 1:num_vars] = hessian[primal_idx, primal_idx]
    # Jacobian of the constraints
    A = spzeros(num_cons, num_vars + num_ineq)
    # A is the Jacobian of: c(x) = b and c(x) <= b and c(x) >= b, possibly all mixed up.
    # Each of the will be re-written as:
    # c(x) - b = 0
    # c(x) - b - su = 0, su <= 0
    # c(x) - b - sl = 0, sl >= 0
    # Jacobian of the constraints wrt the primal variables
    A[:, 1:num_vars] = jacobian[:, primal_idx]
    # Jacobian of the constraints wrt the slack variables
    for (i, j) in enumerate(geq_locations)
        A[j, num_vars+i] = -1
    end
    for (i, j) in enumerate(leq_locations)
        A[j, num_vars+i] = -1
    end
    # Partial second derivative of the lagrangian wrt primal solution and parameters
    ∇ₓₚL = spzeros(num_vars + num_ineq, num_parms)
    ∇ₓₚL[1:num_vars, :] = hessian[primal_idx, params_idx]
    # Partial derivative of the equality constraintswith wrt parameters
    ∇ₚC = jacobian[:, params_idx]

    # M matrix
    # M = [
    #     [W A' -I I];
    #     [A 0 0 0];
    #     [V_L 0 (X - X_L) 0]
    #     [V_U 0 0 0 (X_U - X)]
    # ]
    len_w = num_vars + num_ineq
    M = spzeros(
        len_w + num_cons + num_low + num_up,
        len_w + num_cons + num_low + num_up,
    )

    M[1:len_w, 1:len_w] = W
    M[1:len_w, len_w+1:len_w+num_cons] = A'
    M[len_w+1:len_w+num_cons, 1:len_w] = A
    M[1:len_w, len_w+num_cons+1:len_w+num_cons+num_low] = I_L
    M[len_w+num_cons+1:len_w+num_cons+num_low, 1:len_w] = V_L
    M[
        len_w+num_cons+1:len_w+num_cons+num_low,
        len_w+num_cons+1:len_w+num_cons+num_low,
    ] = X_lb
    M[len_w+num_cons+num_low+1:len_w+num_cons+num_low+num_up, 1:len_w] = V_U
    M[
        len_w+num_cons+num_low+1:len_w+num_cons+num_low+num_up,
        len_w+num_cons+num_low+1:len_w+num_cons+num_low+num_up,
    ] = X_ub
    M[1:len_w, len_w+num_cons+num_low+1:end] = I_U

    # N matrix
    # N = [∇ₓₚL ; ∇ₚC; zeros(num_low + num_up, num_parms)]
    N = spzeros(len_w + num_cons + num_low + num_up, num_parms)
    N[1:len_w, :] = ∇ₓₚL
    N[len_w+1:len_w+num_cons, :] = ∇ₚC

    return M, N
end

"""
    inertia_corrector_factorization(M::SparseMatrixCSC, num_w, num_cons; st=1e-6, max_corrections=50)

Inertia correction for the factorization of the KKT matrix. Sparse version.
"""
function inertia_corrector_factorization(
    M::SparseMatrixCSC,
    num_w,
    num_cons;
    st = 1e-6,
    max_corrections = 50,
)
    # Factorization
    K = lu(M; check = false)
    # Inertia correction
    status = K.status
    num_c = 0
    diag_mat = ones(size(M, 1))
    diag_mat[num_w+1:num_w+num_cons] .= -1
    diag_mat = SparseArrays.spdiagm(diag_mat)
    while status == 1 && num_c < max_corrections
        println("Inertia correction")
        M = M + st * diag_mat
        K = lu(M; check = false)
        status = K.status
        num_c += 1
    end
    if status != 0
        @warn "Inertia correction failed"
        return nothing
    end
    return K
end

"""
    inertia_corrector_factorization(M::Matrix, num_w, num_cons; st=1e-6, max_corrections=50)

Inertia correction for the factorization of the KKT matrix. Dense version.
"""
function inertia_corrector_factorization(M; st = 1e-6, max_corrections = 50)
    num_c = 0
    if cond(M) > 1 / st
        @warn "Inertia correction"
        M = M + st * I(size(M, 1))
        num_c += 1
    end
    while cond(M) > 1 / st && num_c < max_corrections
        M = M + st * I(size(M, 1))
        num_c += 1
    end
    if num_c == max_corrections
        @warn "Inertia correction failed"
        return nothing
    end
    return lu(M)
end

"""
    compute_derivatives_no_relax(model::Model, cons::Vector{MOI.Nonlinear.ConstraintIndex},
        _X::AbstractVector, _V_L::AbstractVector, _X_L::AbstractVector, _V_U::AbstractVector, _X_U::AbstractVector, leq_locations::Vector{Z}, geq_locations::Vector{Z}, ineq_locations::Vector{Z},
        has_up::Vector{Z}, has_low::Vector{Z}
    )

Compute the derivatives of the solution w.r.t. the parameters without accounting for active set changes.
"""
function compute_derivatives_no_relax(
    model::Model,
    cons::Vector{MOI.Nonlinear.ConstraintIndex},
    _X::AbstractVector,
    _V_L::AbstractVector,
    _X_L::AbstractVector,
    _V_U::AbstractVector,
    _X_U::AbstractVector,
    leq_locations::Vector{Z},
    geq_locations::Vector{Z},
    ineq_locations::Vector{Z},
    has_up::Vector{Z},
    has_low::Vector{Z},
) where {Z<:Integer}
    M, N = build_sensitivity_matrices(
        model,
        cons,
        _X,
        _V_L,
        _X_L,
        _V_U,
        _X_U,
        leq_locations,
        geq_locations,
        ineq_locations,
        has_up,
        has_low,
    )

    # Sensitivity of the solution (primal-dual_constraints-dual_bounds) w.r.t. the parameters
    num_vars = get_num_primal_vars(model)
    num_cons = get_num_constraints(model)
    num_ineq = length(ineq_locations)
    K = inertia_corrector_factorization(M, num_vars + num_ineq, num_cons) # Factorization
    if isnothing(K)
        return zeros(size(M, 1), size(N, 2)), K, N
    end
    ∂s = zeros(size(M, 1), size(N, 2))
    # ∂s = - (K \ N) # Sensitivity
    ldiv!(∂s, K, N)
    ∂s = -∂s # multiply by -1 since we used ∂s as an auxilary variable to calculate K \ N

    return ∂s, K, N
end

sense_mult(model::Model) = objective_sense(model) == MOI.MIN_SENSE ? 1.0 : -1.0

"""
    compute_sensitivity(model::Model; tol=1e-6)

Compute the sensitivity of the solution given sensitivity of the parameters (Δp).
"""
function compute_sensitivity(model::Model; tol = 1e-6)
    # Solution and bounds
    X,
    V_L,
    X_L,
    V_U,
    X_U,
    leq_locations,
    geq_locations,
    ineq_locations,
    has_up,
    has_low,
    cons = compute_solution_and_bounds(model; tol = tol)
    # Compute derivatives
    # ∂s = [∂x; ∂λ; ∂ν_L; ∂ν_U]
    ∂s, K, N = compute_derivatives_no_relax(
        model,
        cons,
        X,
        V_L,
        X_L,
        V_U,
        X_U,
        leq_locations,
        geq_locations,
        ineq_locations,
        has_up,
        has_low,
    )
    ## Adjust signs based on JuMP convention
    num_vars = get_num_primal_vars(model)
    num_cons = get_num_constraints(model)
    num_ineq = length(ineq_locations)
    num_w = num_vars + num_ineq
    num_lower = length(has_low)
    sense_multiplier = sense_mult(model)
    # Duals
    ∂s[num_w+1:num_w+num_cons, :] *= -sense_multiplier
    # Dual bounds lower
    ∂s[num_w+num_cons+1:num_w+num_cons+num_lower, :] *= sense_multiplier
    # Dual bounds upper
    ∂s[num_w+num_cons+num_lower+1:end, :] *= -sense_multiplier
    return ∂s
end
