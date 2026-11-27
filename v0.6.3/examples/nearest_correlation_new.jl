# # Nearest correlation

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/nearest_correlation.jl)

# This example illustrates the sensitivity analysis of the nearest correlation problem studied in [H02].
#
# Higham, Nicholas J.
# *Computing the nearest correlation matrix—a problem from finance.*
# IMA journal of Numerical Analysis 22.3 (2002): 329-343.

using DiffOpt, JuMP, SCS, LinearAlgebra
solver = SCS.Optimizer

function proj(A, dH = Diagonal(ones(size(A, 1))), H_data = ones(size(A)))
    n = LinearAlgebra.checksquare(A)
    model = Model(() -> DiffOpt.diff_optimizer(solver))
    @variable(model, X[1:n, 1:n] in PSDCone())
    @variable(model, H[1:n, 1:n] in Parameter.(H_data))
    @variable(model, E[1:n, 1:n])
    @constraint(model, [i in 1:n], X[i, i] == 1)
    @constraint(model, E .== (H .* (X .- A)))
    @objective(model, Min, sum(E .^ 2))
    for i in 1:n
        DiffOpt.set_forward_parameter(model, H[i, i], dH[i, i])
    end
    optimize!(model)
    DiffOpt.forward_differentiate!(model)
    dX = DiffOpt.get_forward_variable.(model, X)
    return value.(X), dX
end

# Example from [H02, p. 334-335]:

A = LinearAlgebra.Tridiagonal(ones(2), ones(3), ones(2))

# The projection is computed as follows:

X, dX = proj(A)
nothing # hide

# The projection of `A` is:

X

# The derivative of the projection with respect to a uniform increase of the weights
# of the diagonal entries is:

dX

# Example from [H02, Section 4, p. 340]:

A = LinearAlgebra.Tridiagonal(-ones(3), 2ones(4), -ones(3))

# The projection is computed as follows:

X, dX = proj(A)
nothing # hide

# The projection of `A` is:

X

# The derivative of the projection with respect to a uniform increase of the weights
# of the diagonal entries is:

dX
