# # Custom ReLU layer

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/custom-relu.jl)

# We demonstrate how DiffOpt can be used to generate a simple neural network
# unit - the ReLU layer. A neural network is created using Flux.jl and
# trained on the MNIST dataset.

# This tutorial uses the following packages

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Flux
import MLDatasets
import Statistics
import Base.Iterators: repeated
using LinearAlgebra

# ## The ReLU and its derivative

# Define a relu through an optimization problem solved by a quadratic solver.
# Return the solution of the problem.
function matrix_relu(
    y::Matrix;
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer)),
)
    layer_size, batch_size = size(y)
    empty!(model)
    set_silent(model)
    @variable(model, x[1:layer_size, 1:batch_size] >= 0)
    @objective(model, Min, x[:]'x[:] - 2y[:]'x[:])
    optimize!(model)
    return Float32.(value.(x))
end

# Define the reverse differentiation rule, for the function we defined above.
function ChainRulesCore.rrule(::typeof(matrix_relu), y::Matrix{T}) where {T}
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    pv = matrix_relu(y; model = model)
    function pullback_matrix_relu(dl_dx)
        ## some value from the backpropagation (e.g., loss) is denoted by `l`
        ## so `dl_dy` is the derivative of `l` wrt `y`
        x = model[:x]::Matrix{JuMP.VariableRef} # load decision variable `x` into scope
        dl_dy = zeros(T, size(x))
        dl_dq = zeros(T, size(x))
        ## set sensitivities
        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x[:], dl_dx[:])
        ## compute grad
        DiffOpt.reverse_differentiate!(model)
        ## return gradient wrt objective function parameters
        obj_exp = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
        ## coeff of `x` in q'x = -2y'x
        dl_dq[:] .= JuMP.coefficient.(obj_exp, x[:])
        dq_dy = -2 # dq/dy = -2
        dl_dy[:] .= dl_dq[:] * dq_dy
        return (ChainRulesCore.NoTangent(), dl_dy)
    end
    return pv, pullback_matrix_relu
end

# For more details about backpropagation, visit [Introduction, ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/dev/).

# ## Define the network

layer_size = 10
m = Flux.Chain(
    Flux.Dense(784, layer_size), # 784 being image linear dimension (28 x 28)
    matrix_relu,
    Flux.Dense(layer_size, 10), # 10 being the number of outcomes (0 to 9)
    Flux.softmax,
)

# ## Prepare data

N = 1000 # batch size
## Preprocessing train data
imgs = MLDatasets.MNIST(; split = :train).features[:, :, 1:N]
labels = MLDatasets.MNIST(; split = :train).targets[1:N]
train_X = float.(reshape(imgs, size(imgs, 1) * size(imgs, 2), N)) # stack images
train_Y = Flux.onehotbatch(labels, 0:9);
## Preprocessing test data
test_imgs = MLDatasets.MNIST(; split = :test).features[:, :, 1:N]
test_labels = MLDatasets.MNIST(; split = :test).targets[1:N];
test_X = float.(reshape(test_imgs, size(test_imgs, 1) * size(test_imgs, 2), N))
test_Y = Flux.onehotbatch(test_labels, 0:9);

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 50 # ~1 minute (i7 8th gen with 16gb RAM)
## epochs = 100 # leads to 77.8% in about 2 minutes
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
