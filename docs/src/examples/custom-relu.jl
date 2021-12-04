# # Custom ReLU layer

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/docs/src/examples/custom-relu.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/custom-relu.ipynb)

# We demonstrate how DiffOpt can be used to generate a simple neural network
# unit - the ReLU layer. A neural network is created using Flux.jl which is
# trained on the MNIST dataset.

# This tutorial uses the following packages

using Statistics
using DiffOpt
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
import OSQP
using JuMP
using ChainRulesCore


## prepare data
imgs = Flux.Data.MNIST.images()
labels = Flux.Data.MNIST.labels();

# Preprocessing
X = hcat(float.(reshape.(imgs, :))...) #stack all the images
Y = onehotbatch(labels, 0:9); # just a common way to encode categorical variables

test_X = hcat(float.(reshape.(Flux.Data.MNIST.images(:test), :))...)
test_Y = onehotbatch(Flux.Data.MNIST.labels(:test), 0:9)

X = X[:, 1:1000]
Y = Y[:, 1:1000]

# Define a relu through an optimization problem solved by a quadratic solver.
# Return the solution of the problem.
function matrix_relu(
    y::AbstractArray{T};
    model = Model(() -> diff_optimizer(OSQP.Optimizer))
) where T
    _x = zeros(size(y))
    N = length(y[:, 1])
    empty!(model)
    set_silent(model)
    @variable(model, x[1:N] >= 0)
    for i in 1:size(y)[2]
        @objective(
            model,
            Min,
            x'x -2x'y[:, i]
        )
        optimize!(model)
        _x[:, i] = value.(x)
    end
    return _x
end


# Define the backward differentiation rule, for the function we defined above.
function ChainRulesCore.rrule(
    ::typeof(matrix_relu),
    y::AbstractArray;
    model = Model(() -> diff_optimizer(OSQP.Optimizer))
) where T
    pv = matrix_relu(y, model = model)
    function pullback_matrix_relu(dx)
        x = model[:x]
        dy = zeros(T, size(dx))
        for i in 1:size(y)[2]
            MOI.set.(
                model,
                DiffOpt.BackwardInVariablePrimal(),
                x,
                dx[:, i]
            ) # set sensitivities
            DiffOpt.backward(model) # compute grad
            obj_exp = MOI.get(
                model,
                DiffOpt.BackwardOutObjective()
            ) # return grdiente wrt objective function parameters
            dy[:, i] = JuMP.coefficient.(obj_exp, x) # coeff of `x` in -2x'y
            dy[:, i] = -2 * dy[:, i]
        end
        return (NO_FIELDS, dy)
    end
    return pv, pullback_matrix_relu
end

## Define the Network

# Network structure

m = Chain(
    Dense(784, 64),
    matrix_relu,
    Dense(64, 10),
    softmax,
)

# Define input data

dataset = repeated((X,Y), 20) # repeat the data set, very low accuracy on the orig dataset

# Parameters for the network training

custom_loss(x, y) = crossentropy(m(x), y) # training loss function
opt = ADAM(); # stochastic gradient descent variant to optimize weights of the neral network
evalcb = () -> @show(custom_loss(X, Y)) # callback to show loss

# Train to optimize network parameters

Flux.train!(custom_loss, params(m), dataset, opt, cb = throttle(evalcb, 5));

# Although our custom implementation takes time, it is able to reach similar
# accuracy as the usual ReLU function implementation.

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y)) # average of correct guesses
@show accuracy(X,Y)
@show accuracy(test_X, test_Y)
