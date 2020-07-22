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
    Inverse matrix specified on RHS of eqn(7) in https://arxiv.org/pdf/1703.00443.pdf
"""
function create_LHS_matrix(z, λ, Q, G, h, A=nothing)
    if A == nothing || size(A)[1] == 0
        return [Q         G' * Diagonal(λ);
                G         Diagonal(G * z - h)]
    else
        @assert size(A)[2] == size(G)[2]
        p, n = size(A)
        m    = size(G)[1]
        return [Q         G' * Diagonal(λ)       A';  
                G         Diagonal(G * z - h)    zeros(m, p);
                A         zeros(p, m)            zeros(p, p)]
    end
end


"""
    Right hand side of eqn(6) in https://arxiv.org/pdf/1703.00443.pdf
"""
function create_RHS_matrix(z, dQ, dq, λ, dG, dh, ν=nothing, dA=nothing, db=nothing)
    if dA == nothing || size(dA)[1] == 0
        return -[dQ * z + dq + dG' * λ      ;
                 Diagonal(λ) * (dG * z - dh)]
    else
        return -[dQ * z + dq + dG' * λ + dA' * ν;
                 Diagonal(λ) * (dG * z - dh)    ;
                 dA * z - db                    ]
    end
end


is_equality(set::MOI.AbstractSet) = false
is_equality(set::MOI.EqualTo) = true

coefficient(t::MOI.ScalarAffineTerm) = t.coefficient


"""
    Return problem parameters as matrices along with other problem info
"""
function get_problem_data(model::MOI.AbstractOptimizer)
    var_idx = MOI.get(model, MOI.ListOfVariableIndices())
    nz = size(var_idx)[1]

    # handle inequality constraints
    ineq_con_idx = MOI.get(
                        model, 
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64}, 
                            MOI.LessThan{Float64}
                        }())
    nineq = size(ineq_con_idx)[1]

    G = zeros(nineq, nz)
    h = zeros(nineq)

    for i in 1:nineq
        con = ineq_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        G[i, :] = coefficient.(func.terms)'
        h[i] = set.upper - func.constant
    end
    
    # handle equality constraints
    eq_con_idx   = MOI.get(
                        model, 
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64}, 
                            MOI.EqualTo{Float64}
                        }())
    neq   = size(eq_con_idx)[1]
    
    A = zeros(neq, nz)
    b = zeros(neq)
    
    for i in 1:neq
        con = eq_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        A[i, :] = coefficient.(func.terms)'
        b[i] = set.value - func.constant
    end

    
    # handle objective
    # works both for affine and quadratic objective functions
    objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    Q = zeros(nz, nz)
    
    if typeof(objective_function) == MathOptInterface.ScalarAffineFunction{Float64}
        q = coefficient.(objective_function.terms)
    elseif typeof(objective_function) == MathOptInterface.ScalarQuadraticFunction{Float64}
        # @assert size(objective_function.quadratic_terms)[1] == (nz * (nz + 1)) / 2    
        
        var_to_id = Dict(var_idx .=> 1:nz)
        
        for quad in objective_function.quadratic_terms
            i = var_to_id[quad.variable_index_1]
            j = var_to_id[quad.variable_index_2]
            Q[i,j] = quad.coefficient
            Q[j,i] = quad.coefficient
        end
        
        q = coefficient.(objective_function.affine_terms)
    end
    
    return Q, q, G, h, A, b, nz, var_idx, nineq, ineq_con_idx, neq, eq_con_idx
end


# find expression of projections on cones and their derivatives here:
#   https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf


"""
    projection of vector `z` on zero cone i.e. K = {0} or its dual
"""
function π(::MOI.Zeros, z::Array{Float64}, dual=true)
    return dual ? z : zeros(Float64, size(z))
end

function π(::MOI.EqualTo, z::Array{Float64}, dual=true)
    return dual ? z : zeros(Float64, size(z))
end

function π(::MOI.EqualTo, z::Float64, dual=true)
    return dual ? [z] : zeros(Float64, 1)
end

"""
    projection of vector `z` on Nonnegative cone i.e. K = R+
"""
function π(::MOI.Nonnegatives, z::Array{Float64})
    return max.(z,0.0)
end

"""
    projection of vector `z` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function π(::MOI.SecondOrderCone, z::Array{Float64})
    t = z[1]
    x = z[2:length(z)]
    norm_x = norm(x)
    if norm_x <= t
        return copy(z)
    elseif norm_x <= -t
        return zeros(Float64, size(z))
    else
        result = zeros(Float64, size(z))
        result[1] = 1.0
        result[2:length(z)] = x / norm_x
        result *= (norm_x + t) / 2.0
        return result
    end
end

"""
    projection of vector `z` on positive semidefinite cone i.e. K = S^n⨥
"""
function π(::MOI.PositiveSemidefiniteConeTriangle, z::Array{Float64})
    dim = Int64(floor(√(2*length(z))))
    X = unvec_symm(z, dim)
    λ, U = eigen(X)
    return vec_symm(U * Diagonal(max.(λ,0)) * U')
end

"""
    Returns a dim-by-dim symmetric matrix corresponding to `x`.

    `x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
    matrix; the correspondence is as in SCS.
    X = [ X11 X12 ... X1k
          X21 X22 ... X2k
          ...
          Xk1 Xk2 ... Xkk ],
    where
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
"""
function unvec_symm(x, dim)
    X = zeros(dim, dim)
    for i in 1:dim
        for j in i:dim
            X[j,i] = x[(i-1)*dim-Int(((i-1)*i)/2)+j]
        end
    end
    X = X + X'
    X /= √2
    for i in 1:dim
        X[i,i] /= √2
    end
    return X
end

"""
    Returns a vectorized representation of a symmetric matrix `X`.
    Vectorization (including scaling) as per SCS.
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
"""
function vec_symm(X)
    X = copy(X)
    X *= √2
    for i in 1:size(X)[1]
        X[i,i] /= √2
    end
    return X[tril(trues(size(X)))]
end

"""
    Projection onto R^n x K^* x R_+
    `cones` represents a convex cone K, and K^* is its dual cone
"""
function π(cones::Array{<:MOI.AbstractSet}, z)
    @assert length(cones) == length(z)
    return vcat([π(cones[i], z[i]) for i in 1:length(cones)]...)
end


#  Derivative of the projection of vector `z` on MOI set `cone`
#  Dπ[i,j] = ∂π[i] / ∂z[j]   where `π` denotes projection of `z` on `cone`

"""
    derivative of projection of vector `z` on zero cone i.e. K = {0}
"""
function Dπ(::MOI.Zeros, z::Array{Float64})
    y = ones(Float64, size(z))
    return reshape(y, length(y), 1)
end

function Dπ(::MOI.EqualTo, z::Array{Float64})
    y = ones(Float64, size(z))
    return reshape(y, length(y), 1)
end

function Dπ(::MOI.EqualTo, ::Float64)
    y = ones(Float64, 1)
    return reshape(y, length(y), 1)
end

"""
    derivative of projection of vector `z` on Nonnegative cone i.e. K = R+
"""
function Dπ(::MOI.Nonnegatives, z::Array{Float64})
    y = (sign.(z) .+ 1.0)/2
    return reshape(y, length(y), 1)
end

"""
    derivative of projection of vector `z` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function Dπ(::MOI.SecondOrderCone, z::Array{Float64})
    n = length(z)
    t = z[1]
    x = z[2:n]
    norm_x = norm(x)
    if norm_x <= t
        return Matrix{Float64}(I,n,n)
    elseif norm_x <= -t
        return zeros(n,n)
    else
        result = [
            norm_x     x';
            x          (norm_x + t)*Matrix{Float64}(I,n-1,n-1) - (t/(norm_x^2))*(x*x')
        ]
        result /= (2.0 * norm_x)
        return result
    end
end

"""
    derivative of projection of vector `z` on positive semidefinite cone i.e. K = S^n⨥
"""
function Dπ(cone::MOI.PositiveSemidefiniteConeTriangle, z::Array{Float64})
    n = length(z)
    y = zeros(n)
    D = zeros(n,n)

    for i in 1:n
        y[i] = 1.0
        D[i, 1:n] = Dπ(cone, z, y)
        y[i] = 0.0
    end

    return D
end

function Dπ(::MOI.PositiveSemidefiniteConeTriangle, z::Array{Float64}, y::Array{Float64})
    n = length(z)
    dim = Int64(floor(√(2*n)))
    X = unvec_symm(z, dim)
    λ, U = eigen(X)

    # if all the eigenvalues are >= 0
    if max.(λ, 0) == λ
        return Matrix{Float64}(I, n, n)
    end

    # k is the number of negative eigenvalues in X minus ONE
    k = count(λ .< 1e-4)

    # defining matrix B
    X̃ = unvec_symm(y, dim)
    B = U' * X̃ * U
    
    for i in 1:size(B)[1] # do the hadamard product
        for j in 1:size(B)[2]
            if (i <= k && j <= k)
                B[i, j] = 0
            elseif (i > k && j <= k)
                λpi = max(λ[i], 0.0)
                λmj = -min(λ[j], 0.0)
                B[i, j] *= λpi / (λmj + λpi)
            elseif (i <= k && j > k) 
                λmi = -min(λ[i], 0.0)
                λpj = max(λ[j], 0.0)
                B[i, j] *= λpj / (λmi + λpj)
            end
        end
    end

    return vec_symm(U * B * U')
end

"""
    derivative of projection of vector `z` on a product of cones
"""
function Dπ(cones::Array{<:MOI.AbstractSet}, z)
    @assert length(cones) == length(z)
    return BlockDiagonal([Dπ(cones[i], z[i]) for i in 1:length(cones)])
end
