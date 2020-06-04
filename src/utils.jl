# TODO: create test functions for the methods

"""
    Left hand side of eqn(6) in https://arxiv.org/pdf/1703.00443.pdf
"""
function create_LHS_matrix(z, λ, Q, G, h, A=nothing)
    if A == nothing || size(A)[1] == 0
        return [Q                G';
                Diagonal(λ) * G    Diagonal(G * z - h)]
    else
        @assert size(A)[2] == size(G)[2]
        p, n = size(A)
        m    = size(G)[1]
        return [Q                  G'                    A';  
                Diagonal(λ) * G    Diagonal(G * z - h)   zeros(m, p);
                A                  zeros(p, m)           zeros(p, p)]
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



is_equality(set::S) where {S<:MOI.AbstractSet} = false
is_equality(set::MOI.EqualTo{T}) where T       = true
