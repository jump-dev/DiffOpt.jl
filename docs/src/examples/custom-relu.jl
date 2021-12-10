# # Custom ReLU layer

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/custom-relu.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/custom-relu.ipynb)

# We demonstrate how DiffOpt can be used to generate a simple neural network
# unit - the ReLU layer. A neural network is created using Flux.jl which is
# trained on the MNIST dataset.

# This tutorial uses the following packages

using JuMP
import DiffOpt
import OSQP
import ChainRulesCore
import Flux
import Statistics
import Base.Iterators: repeated

# ## The ReLU and its derivative

# Define a relu through an optimization problem solved by a quadratic solver.
# Return the solution of the problem.
function matrix_relu(
    y::AbstractArray{T};
    model = Model(() -> DiffOpt.diff_optimizer(OSQP.Optimizer))
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
            x'x -2y[:, i]'x  # x' Q x + q'x with Q = I, q = -2y
        )
        optimize!(model)
        _x[:, i] = value.(x)
    end
    return _x
end


# Define the backward differentiation rule, for the function we defined above.
function ChainRulesCore.rrule(
    ::typeof(matrix_relu),
    y::AbstractArray{T};
    model = Model(() -> DiffOpt.diff_optimizer(OSQP.Optimizer))
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
            ) # return gradient wrt objective function parameters
            dq[:, i] = JuMP.coefficient.(obj_exp, x) # coeff of `x` in q'x = -2x'y
            dy[:, i] = -2 * dq[:, i]  # ∵ dq/dy = -2
        end
        return (ChainRulesCore.NoTangent(), dy,)
    end
    return pv, pullback_matrix_relu
end

# ## prepare data
imgs = Flux.Data.MNIST.images()
labels = Flux.Data.MNIST.labels();

# Preprocessing
X = hcat(float.(reshape.(imgs, :))...) #stack all the images
Y = Flux.onehotbatch(labels, 0:9); # just a common way to encode categorical variables

N = 1000

train_X = X[:, 1:N]
train_Y = Y[:, 1:N]

test_X = hcat(float.(reshape.(Flux.Data.MNIST.images(:test), :))...)
test_Y = Flux.onehotbatch(Flux.Data.MNIST.labels(:test), 0:9);

# ## Define the Network

# Network structure

inner = 15

m = Flux.Chain(
    Flux.Dense(784, inner), #784 being image linear dimension (28 x 28)
    matrix_relu,
    Flux.Dense(inner, 10), # 10 beinf the number of outcomes (0 to 9)
    Flux.softmax,
)

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 5

dataset = repeated((train_X, train_Y), epochs);

# Parameters for the network training

custom_loss(x, y) = Flux.crossentropy(m(x), y) # training loss function
opt = Flux.ADAM(); # stochastic gradient descent variant to optimize weights of the neral network
evalcb = () -> @show(custom_loss(X, Y)); # callback to show loss

# Train to optimize network parameters

@time Flux.train!(custom_loss, Flux.params(m), dataset, opt, cb = Flux.throttle(evalcb, 5));

# Although our custom implementation takes time, it is able to reach similar
# accuracy as the usual ReLU function implementation.

accuracy(x, y) = Statistics.mean(Flux.onecold(m(x)) .== Flux.onecold(y)); # average of correct guesses

# Train accuracy

accuracy(train_X, train_Y)

# Test accuracy

accuracy(test_X, test_Y)

# Note that the accuracy is low due to simplified training.
# It is possible to increase the number of samples `N`,
# the number of epochs `epoch` and teh conectivity `inner`.
