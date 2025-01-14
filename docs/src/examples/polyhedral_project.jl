# # Polyhedral QP layer

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/polyhedral_project.jl)

# We use DiffOpt to define a custom network layer which, given an input matrix `y`,
# computes its projection onto a polytope defined by a fixed number of inequalities:
# `a_i^T x ≥ b_i`.
# A neural network is created using Flux.jl and trained on the MNIST dataset,
# integrating this quadratic optimization layer.
#
# The QP is solved in the forward pass, and its DiffOpt derivative is used in the backward pass expressed with `ChainRulesCore.rrule`.

# This example is similar to the custom ReLU layer, except that the layer is parameterized
# by the hyperplanes `(w,b)` and not a simple stateless function.
# This also means that `ChainRulesCore.rrule` must return the derivatives of the output with respect to the
# layer parameters to allow for backpropagation.

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Flux
import MLDatasets
import Statistics
using Base.Iterators: repeated
using LinearAlgebra
using Random

Random.seed!(42)

# ## The Polytope representation and its derivative

struct Polytope{N}
    w::NTuple{N,Vector{Float64}}
    b::Vector{Float64}
end

Polytope(w::NTuple{N}) where {N} = Polytope{N}(w, randn(N))

# We define a "call" operation on the polytope, making it a so-called functor.
# Calling the polytope with a matrix `y` operates an Euclidean projection of this matrix onto the polytope.
function (polytope::Polytope{N})(
    y::AbstractMatrix;
    model = direct_model(DiffOpt.diff_optimizer(Ipopt.Optimizer)),
) where {N}
    layer_size, batch_size = size(y)
    empty!(model)
    set_silent(model)
    @variable(model, x[1:layer_size, 1:batch_size])
    @constraint(
        model,
        greater_than_cons[idx in 1:N, sample in 1:batch_size],
        dot(polytope.w[idx], x[:, sample]) ≥ polytope.b[idx]
    )
    @objective(model, Min, dot(x - y, x - y))
    optimize!(model)
    return Float32.(JuMP.value.(x))
end

# The `@functor` macro from Flux implements auxiliary functions for collecting the parameters of
# our custom layer and operating backpropagation.
Flux.@functor Polytope

# Define the reverse differentiation rule, for the function we defined above.
# Flux uses ChainRules primitives to implement reverse-mode differentiation of the whole network.
# To learn the current layer (the polytope the layer contains),
# the gradient is computed with respect to the `Polytope` fields in a ChainRulesCore.Tangent type
# which is used to represent derivatives with respect to structs.
# For more details about backpropagation, visit [Introduction, ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/dev/).

function ChainRulesCore.rrule(
    polytope::Polytope{N},
    y::AbstractMatrix,
) where {N}
    model = direct_model(DiffOpt.diff_optimizer(Ipopt.Optimizer))
    xv = polytope(y; model = model)
    function pullback_matrix_projection(dl_dx)
        layer_size, batch_size = size(dl_dx)
        dl_dx = ChainRulesCore.unthunk(dl_dx)
        ##  `dl_dy` is the derivative of `l` wrt `y`
        x = model[:x]
        ## grad wrt input parameters
        dl_dy = zeros(size(dl_dx))
        ## grad wrt layer parameters
        dl_dw = zero.(polytope.w)
        dl_db = zero(polytope.b)
        ## set sensitivities
        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, dl_dx)
        ## compute grad
        DiffOpt.reverse_differentiate!(model)
        ## compute gradient wrt objective function parameter y
        obj_expr = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
        dl_dy .= -2 * JuMP.coefficient.(obj_expr, x)
        greater_than_cons = model[:greater_than_cons]
        for idx in 1:N, sample in 1:batch_size
            cons_expr = MOI.get(
                model,
                DiffOpt.ReverseConstraintFunction(),
                greater_than_cons[idx, sample],
            )
            dl_db[idx] -= JuMP.constant(cons_expr) / batch_size
            dl_dw[idx] .+=
                JuMP.coefficient.(cons_expr, x[:, sample]) / batch_size
        end
        dself = ChainRulesCore.Tangent{Polytope{N}}(; w = dl_dw, b = dl_db)
        return (dself, dl_dy)
    end
    return xv, pullback_matrix_projection
end

# ## Define the Network

layer_size = 20
m = Flux.Chain(
    Flux.Dense(784, layer_size), # 784 being image linear dimension (28 x 28)
    Polytope((randn(layer_size), randn(layer_size), randn(layer_size))),
    Flux.Dense(layer_size, 10), # 10 being the number of outcomes (0 to 9)
    Flux.softmax,
)

# ## Prepare data

M = 500 # batch size
## Preprocessing train data
imgs = MLDatasets.MNIST(split=:train).features[:,:,1:M]
labels = MLDatasets.MNIST(split=:train).targets[1:M]
train_X = float.(reshape(imgs, size(imgs, 1) * size(imgs, 2), M)) # stack images
train_Y = Flux.onehotbatch(labels, 0:9);
## Preprocessing test data
test_imgs = MLDatasets.MNIST(split=:test).features[:,:,1:M]
test_labels = MLDatasets.MNIST(split=:test).targets[1:M]
test_X = float.(reshape(test_imgs, size(test_imgs, 1) * size(test_imgs, 2), M))
test_Y = Flux.onehotbatch(test_labels, 0:9);

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 50
dataset = repeated((train_X, train_Y), epochs);

# ## Network training

# training loss function, Flux optimizer
custom_loss(m, x, y) = Flux.crossentropy(m(x), y)
opt = Flux.setup(Flux.Adam(), m)

# Train to optimize network parameters

@time Flux.train!(custom_loss, m, dataset, opt);

# Although our custom implementation takes time, it is able to reach similar
# accuracy as the usual ReLU function implementation.

# ## Accuracy results

# Average of correct guesses
accuracy(x, y) = Statistics.mean(Flux.onecold(m(x)) .== Flux.onecold(y));

# Training accuracy

accuracy(train_X, train_Y)

# Test accuracy

accuracy(test_X, test_Y)

# Note that the accuracy is low due to simplified training.
# It is possible to increase the number of samples `N`,
# the number of epochs `epoch` and the connectivity `inner`.
