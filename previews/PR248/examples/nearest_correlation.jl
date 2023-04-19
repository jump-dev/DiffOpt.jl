# # Nearest correlation

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/nearest_correlation.jl)

# This example illustrates the sensitivity analysis of the nearest correlation problem studied in [H02].
#
# Higham, Nicholas J.
# *Computing the nearest correlation matrix—a problem from finance.*
# IMA journal of Numerical Analysis 22.3 (2002): 329-343.

using DiffOpt, JuMP, SCS, LinearAlgebra
solver = SCS.Optimizer

function proj(A, dH = Diagonal(ones(size(A, 1))), H = ones(size(A)))
    n = LinearAlgebra.checksquare(A)
    model = Model(() -> DiffOpt.diff_optimizer(solver))
    @variable(model, X[1:n, 1:n] in PSDCone())
    @constraint(model, [i in 1:n], X[i, i] == 1)
    @objective(model, Min, sum((H .* (X - A)).^2))
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), sum((dH .* (X - A)).^2))
    optimize!(model)
    DiffOpt.forward_differentiate!(model)
    dX = MOI.get.(model, DiffOpt.ForwardVariablePrimal(), X)
    return value.(X), dX
end

# Example from [H02, p. 334-335]:

A = LinearAlgebra.Tridiagonal(ones(2), ones(3), ones(2))

# The projection is computed as follows:

X, dX = proj(A)
nothing # hide

# The projection of `A` is:

X

# The derivative of the projection with respect to increase uniformly the weights
# of the diagonal entries is:

dX

# Example from [H02, Section 4, p. 340]:

A = LinearAlgebra.Tridiagonal(-ones(3), 2ones(4), -ones(3))

# The projection is

X, dX = proj(A)
nothing # hide

# The projection of `A` is:

X

# The derivative of the projection with respect to increase uniformly the weights
# of the diagonal entries is:

dX
